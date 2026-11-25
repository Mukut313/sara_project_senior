"""OCR v5: scanner-style text reading, Bangla + English, fully offline (config.OCR_PIPELINE = "v5").

    picture --> page found? flatten it (doc_scanner.find_page / flatten)
            --> even out light and shadow (doc_scanner.even_light), for the text finder
            --> find text lines: PP-OCRv5 detector, on the AI HAT or the CPU (ppocr_det.TextFinder)
            --> straighten: when no page outline, turn the picture by the lines' common tilt and find again
            --> cut every line along its own tilt (doc_scanner.crop_line)
            --> read the lines: SARA line recogniser, ONNX (sara_rec.SaraRecognizer), in reading order;
                a line read with low confidence is also tried upside down
            --> main text block, reading order, clean-up (ocr_text_region)

Nothing here downloads anything or needs the internet. Compared with the EasyOCR path (v2) it does not load PyTorch
or EasyOCR at all (about 600 MB less memory on the Pi).

read(frame, on_line=None) -> dict(text, lines, boxes, lang, timings, page)
  on_line(text) is called for each line as soon as it is read (pages are spoken while the rest is being read).
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable, Optional

import cv2
import numpy as np

import config
import doc_scanner as ds

logger = logging.getLogger("smart_glass.ocr")

_CONTENT = re.compile(r"[^\W_]", re.UNICODE)


def _upright(q: np.ndarray) -> np.ndarray:
    """The line box turned level around its centre: same length and height, no tilt. The main-block rule compares
    line heights; with the tilted boxes' upright extents, lines on a tilted page looked taller than level lines
    and the level ones were dropped as 'background' (validation pages, 2026-11-25)."""
    q = np.asarray(q, np.float32)
    c = q.mean(0)
    w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3])) / 2
    h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])) / 2
    return np.float32([[c[0] - w, c[1] - h], [c[0] + w, c[1] - h], [c[0] + w, c[1] + h], [c[0] - w, c[1] + h]])


def _looks_like_text(t: str) -> bool:
    s = t.replace(" ", "")
    return bool(s) and len(_CONTENT.findall(s)) / len(s) >= 0.5


class OcrV5:
    def __init__(self):
        from ppocr_det import TextFinder
        from sara_rec import make_recognizer
        self.finder = TextFinder(getattr(config, "OCR5_DET_BACKEND", "auto"))
        self.rec = make_recognizer()

    def close(self):
        self.finder.close()

    # -------------------------------------------------------------- quick look (auto capture)
    def quick_lines(self, frame: np.ndarray, width: int = 640) -> list:
        """Text-line outlines on a small copy of the frame, in frame pixels; for the auto-capture check."""
        h, w = frame.shape[:2]
        s = min(1.0, width / float(w))
        small = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else frame
        boxes, _ = self.finder.find(small, box_thresh=0.6)
        return [b / s for b, _ in boxes]

    # -------------------------------------------------------------- full read
    def _find(self, img):
        kw = dict(thresh=float(getattr(config, "OCR5_DB_THRESH", 0.3)),
                  box_thresh=float(getattr(config, "OCR5_BOX_THRESH", 0.5)),
                  unclip=float(getattr(config, "OCR5_UNCLIP", 1.6)))
        boxes, _ = self.finder.find(img, **kw)
        # very large letters (camera close to a sign): the finder then returns one box around most of the picture.
        # Look again at a smaller scale, where the letters have the size the finder was trained on.
        H = img.shape[0]
        big = float(getattr(config, "OCR5_BIG_TEXT_FRAC", 0.3))
        if big and any(min(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1])) > big * H for q, _ in boxes):
            s = float(getattr(config, "OCR5_BIG_TEXT_SCALE", 0.35))
            small = cv2.resize(img, (max(32, int(img.shape[1] * s)), max(32, int(H * s))), interpolation=cv2.INTER_AREA)
            pad = int(0.15 * max(small.shape[:2]))
            small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            b2, _ = self.finder.find(small, **kw)
            if b2:
                boxes = [((q - pad) / s, sc) for q, sc in b2]
        min_h = float(getattr(config, "OCR5_MIN_LINE_PX", 7))
        out = []
        for q, sc in boxes:
            hgt = min(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
            wid = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
            if min(hgt, wid) >= min_h:
                out.append((q, sc))
        return out

    def read(self, frame: np.ndarray, on_line: Optional[Callable[[str], None]] = None, stream_min_rows: int = 4) -> dict:
        """on_line is used only when the picture has at least stream_min_rows text rows (a page)."""
        t = {"start": time.perf_counter()}
        rotated = False
        img = frame
        page = None
        page_margin = 0                     # margin flatten() adds around a page (px), so a view can crop it
        # The text finder may also look at a cleaned-up copy (OCR5_ENHANCE_FIND, e.g. "even" = shadows removed); lines
        # are always cut from the picture itself. The cleaned copy is used only for document-like pictures: the
        # finder first looks at the plain picture, and only if it finds at least OCR5_FIND_ENHANCE_MIN_LINES lines does
        # it look at the cleaned copy too and keep whichever found more. Shadow removal assumes dark text on light
        # paper; on photos of signs (light text on dark, single words) it hid the text (validation real words: nothing
        # read 11 % -> 47 % when it was always on, 2026-11-25).
        def find_best(im):
            b0 = self._find(im)
            mode = getattr(config, "OCR5_ENHANCE_FIND", "none")
            if getattr(config, "OCR5_EVEN_LIGHT_FIND", False):
                mode = "even"
            if not mode or mode == "none" or len(b0) < int(getattr(config, "OCR5_FIND_ENHANCE_MIN_LINES", 3)):
                return b0
            b1 = self._find(cv2.cvtColor(ds.enhance(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), mode), cv2.COLOR_GRAY2BGR))
            return b1 if len(b1) >= len(b0) else b0
        boxes = find_best(img)
        t["page"] = time.perf_counter()
        # A page is flattened only when it is a document: an outline with at least OCR5_PAGE_MIN_LINES text lines
        # inside it. A sign or a word photo that merely has a rectangular outline is read as it is (flattening those
        # made validation words worse, 2026-11-25).
        if getattr(config, "OCR5_FLATTEN_PAGE", True) and len(boxes) >= int(getattr(config, "OCR5_PAGE_MIN_LINES", 3)):
            q = ds.find_page(frame)
            if q is not None:
                poly = q.reshape(-1, 1, 2).astype(np.float32)
                inside = sum(cv2.pointPolygonTest(poly, (float(b[:, 0].mean()), float(b[:, 1].mean())), False) >= 0
                             for b, _ in boxes)
                # a page holds most of the text in view and is lighter than what is around it (paper); an outline
                # in a photo scene (a window, a building) that happens to hold a few text boxes is not one
                # (validation 2026-11-25: such scenes read 75 % wrong when flattened, 4.8 % when not)
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                m = np.zeros(g.shape, np.uint8)
                cv2.fillPoly(m, [q.astype(np.int32)], 1)
                paper = float(np.median(g[m > 0])) if m.any() else 0.0
                around = float(np.median(g[m == 0])) if (m == 0).any() else 0.0
                if (inside >= int(getattr(config, "OCR5_PAGE_MIN_LINES", 3)) and inside >= 0.7 * len(boxes)
                        and paper > around + 15):
                    img = ds.flatten(frame, q)
                    page_margin = ds.LAST_FLATTEN_MARGIN
                    page = q.tolist()
                    boxes = find_best(img)
        # No page edge in view (the page fills the picture or runs off it): estimate the page's four corners from the
        # text lines themselves and warp them flat with a homography, as a scanner app does (OCR5_TEXT_WARP).
        text_warp = None
        if page is None and getattr(config, "OCR5_TEXT_WARP", True) and len(boxes) >= 3:
            from ocr_text_region import dominant_block
            blk = dominant_block([(_upright(q), i, 1.0) for i, (q, _) in enumerate(boxes)], 0.0)
            keep = {b["t"] for b in blk}
            tq = ds.text_quad([q for i, (q, _) in enumerate(boxes) if i in keep], img.shape)
            if tq is not None and ds.perspective_amount(tq) > float(getattr(config, "OCR5_TEXT_WARP_MIN_DEG", 1.5)):
                img, _ = ds.warp_quad(img, tq)
                boxes = find_best(img)
                text_warp = tq.tolist()
                rotated = True
        # tilted picture without a page outline: turn it level and look again (lines then come out straighter)
        ang = ds.deskew_angle([b for b, _ in boxes]) if boxes else 0.0
        if page is None and text_warp is None and abs(ang) > float(getattr(config, "OCR5_DESKEW_MIN_DEG", 4.0)):
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
            M[0, 2] += nw / 2 - w / 2
            M[1, 2] += nh / 2 - h / 2
            img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            boxes = self._find(img)
            rotated = True
        t["find"] = time.perf_counter()
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if getattr(config, "OCR5_EVEN_LIGHT_READ", False):
            grey = ds.even_light(grey, 0.7)
        # scanner-style clean-up of the picture the lines are cut from (doc_scanner.enhance; OCR5_ENHANCE)
        grey = ds.enhance(grey, getattr(config, "OCR5_ENHANCE", "none"))
        # Main text block (ocr_text_region.dominant_block, as in v4): in a scene, only the lines of similar size around
        # the most confidently read large text are spoken; other signs, number plates and stray marks in the picture
        # are not. Not applied to a flattened page, whose headings and body text differ in size.
        # config.OCR5_MAIN_BLOCK: "auto" (no page outline only), "always", "off".
        # Pages read line by line while speaking (streaming) need the choice before reading, so there it is made on
        # geometry alone; otherwise it is made after reading, with each line's confidence (validation, 2026-11-25:
        # the geometry-only rule spoke background signs in photo scenes).
        mb = getattr(config, "OCR5_MAIN_BLOCK", "auto")
        # a flattened "page" with only a few lines is a sign or label: the main-block rule applies to it as well
        few = len(ds.reading_order([(q, sc) for q, sc in boxes])) < int(getattr(config, "OCR5_PAGE_BLOCK_BELOW_ROWS", 5))
        use_block = bool(boxes) and (mb == "always" or (mb == "auto" and (page is None or few)))
        rows = ds.reading_order([(q, sc) for q, sc in boxes])
        if len(rows) < stream_min_rows:
            on_line = None
        if use_block and on_line is not None:
            from ocr_text_region import dominant_block
            blk = dominant_block([(_upright(q), i, 1.0) for i, (q, _) in enumerate(boxes)], 0.0)
            keep = {b["t"] for b in blk}
            boxes = [b for i, b in enumerate(boxes) if i in keep]
            rows = ds.reading_order([(q, sc) for q, sc in boxes])
            use_block = False
        items = [it for row in rows for it in row]
        crops = [ds.crop_line(grey, q, float(getattr(config, "OCR5_CROP_PAD", 0.0))) for q, _ in items]
        min_conf = float(getattr(self.rec, "min_conf", getattr(config, "OCR5_MIN_CONF", 0.45)))
        results = [None] * len(items)
        spoken = 0
        first_t = None
        # read in small groups in reading order, so the first lines can be spoken before the rest is read
        group = int(getattr(config, "OCR5_GROUP", 6))
        for g0 in range(0, len(items), group):
            idx = list(range(g0, min(len(items), g0 + group)))
            res = self.rec.read([crops[i] for i in idx])
            low = [k for k, (txt, c) in enumerate(res) if c < getattr(self.rec, "flip_below", 0.6)]
            if low and getattr(config, "OCR5_TRY_FLIP", True):          # upside down? read it turned round
                flips = self.rec.read([cv2.rotate(crops[idx[k]], cv2.ROTATE_180) for k in low])
                for k, f in zip(low, flips):
                    if f[1] > res[k][1] + 0.1:
                        res[k] = f
            # Test-time clean-up (OCR5_TTA): a line read with less than OCR5_TTA_BELOW confidence is read again in
            # cleaned-up versions (doc_scanner.enhance modes, e.g. "sharp,clahe"); the most confident reading is kept.
            # A filter is then used only where it helps that line.
            variants = [v.strip() for v in str(getattr(config, "OCR5_TTA", "")).split(",") if v.strip()]
            if variants:
                below = float(getattr(config, "OCR5_TTA_BELOW", 0.9))
                gain = float(getattr(config, "OCR5_TTA_MARGIN", 0.02))
                cand = [k for k, (txt, c) in enumerate(res) if c < below]
                for v in variants:
                    if not cand:
                        break
                    alt = self.rec.read([ds.enhance(crops[idx[k]], v) for k in cand])
                    for k, a in zip(cand, alt):
                        if a[1] > res[k][1] + gain:
                            res[k] = a
            for k, i in enumerate(idx):
                results[i] = res[k]
            if on_line is not None:
                # speak whole rows that are complete
                done = g0 + len(idx)
                acc = 0
                for r_i, row in enumerate(rows):
                    acc += len(row)
                    if acc > done:
                        break
                    if r_i < spoken:
                        continue
                    line = self._row_text(row, items, results, min_conf)
                    spoken = r_i + 1
                    if line:
                        if first_t is None:
                            first_t = time.perf_counter()
                        on_line(line)
        t["read"] = time.perf_counter()
        if use_block:
            from ocr_text_region import dominant_block
            det = [(_upright(q), i, c) for i, ((q, _), (txt, c)) in enumerate(zip(items, results))
                   if txt and c >= min_conf and _looks_like_text(txt)]
            keep = {b["t"] for b in dominant_block(det, min_conf)}
            results = [r if i in keep else ("", 0.0) for i, r in enumerate(results)]
        lines, kept_boxes = [], []
        for row in rows:
            line = self._row_text(row, items, results, min_conf)
            if line:
                lines.append(line)
        for (q, sc), (txt, c) in zip(items, results):
            kept_boxes.append((q, txt, c))
        text = self._clean("\n".join(lines))
        from utils import detect_language
        lang = detect_language(text) if text else "en"
        t["end"] = time.perf_counter()
        tm = {"page_ms": 1000 * (t["page"] - t["start"]), "find_ms": 1000 * (t["find"] - t["page"]),
              "read_ms": 1000 * (t["read"] - t["find"]), "total_ms": 1000 * (t["end"] - t["start"]),
              "first_line_ms": None if first_t is None else 1000 * (first_t - t["start"]),
              "lines": len(items), "backend": self.finder.backend}
        logger.info("OCR v5: %d lines, %.0f ms (find %.0f, read %.0f)%s", len(items), tm["total_ms"], tm["find_ms"],
                    tm["read_ms"], ", page flattened" if page else "")
        return {"text": text, "lines": lines, "boxes": kept_boxes, "lang": lang, "timings": tm, "page": page,
                "image": img, "rotated": rotated, "text_warp": text_warp, "scan": grey, "page_margin": page_margin}

    @staticmethod
    def _row_text(row, items, results, min_conf):
        words = []
        pos = {id(x): k for k, x in enumerate(items)}
        for it in row:
            i = pos[id(it)]
            txt, c = results[i] if results[i] else ("", 0.0)
            if txt and c >= min_conf and _looks_like_text(txt):
                words.append(txt)
        return " ".join(words).strip()

    @staticmethod
    def _clean(text: str) -> str:
        try:
            from ocr_text_region import clean_text
            return "\n".join(clean_text(l) for l in text.split("\n") if clean_text(l))
        except Exception:  # noqa: BLE001
            return text
