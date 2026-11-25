"""render_mask (shaped_text.py) with a glyph cache, for the training-line renderer only.

Same HarfBuzz shaping and FreeType rasterising as shaped_text.render_mask (that one stays unchanged for the
benchmark). Each glyph bitmap is turned into a numpy array once per (font, size, glyph) and reused: converting
FreeType's bitmap buffer through a Python list was 40 % of the renderer's time.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..shaped_text import _faces

_GLYPHS: dict = {}


@lru_cache(maxsize=512)
def _hb_font(path, index, size):
    import uharfbuzz as hb
    hface, _ = _faces(path, index)
    f = hb.Font(hface)
    f.scale = (size * 64, size * 64)
    return f


def _glyph(path, index, size, gid):
    key = (path, size, gid)
    g = _GLYPHS.get(key)
    if g is None:
        import freetype
        _, ft = _faces(path, index)
        ft.set_char_size(size * 64)
        ft.load_glyph(gid, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_RENDER)
        gl = ft.glyph
        bm = gl.bitmap
        if bm.width and bm.rows:
            arr = np.frombuffer(bytes(bm.buffer), np.uint8).reshape(bm.rows, bm.pitch)[:, :bm.width].copy()
        else:
            arr = None
        g = (arr, gl.bitmap_left, gl.bitmap_top)
        if len(_GLYPHS) > 200_000:
            _GLYPHS.clear()
        _GLYPHS[key] = g
    return g


def render_mask(text: str, font_path: str, size: int, index: int = 0) -> np.ndarray:
    import uharfbuzz as hb
    font = _hb_font(font_path, index, size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf, {})
    width = sum(p.x_advance for p in buf.glyph_positions) // 64 + 4 * size
    height = 4 * size
    canvas = np.zeros((height, width), np.uint8)
    pen_x, base = 2 * size * 64, int(2.6 * size)
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        arr, left, top = _glyph(font_path, index, size, info.codepoint)
        if arr is not None:
            x = (pen_x + pos.x_offset) // 64 + left
            y = base - (pos.y_offset // 64) - top
            rows, cols = arr.shape
            y0, x0 = max(y, 0), max(x, 0)
            y1, x1 = min(y + rows, height), min(x + cols, width)
            if y1 > y0 and x1 > x0:
                sub = arr[y0 - y:y1 - y, x0 - x:x1 - x]
                np.maximum(canvas[y0:y1, x0:x1], sub, out=canvas[y0:y1, x0:x1])
        pen_x += pos.x_advance
    rows = np.flatnonzero(canvas.any(1))
    cols = np.flatnonzero(canvas.any(0))
    if len(rows) == 0:
        return np.zeros((size, size), np.uint8)
    return canvas[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
