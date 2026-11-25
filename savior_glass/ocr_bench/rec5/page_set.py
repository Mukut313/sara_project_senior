"""Camera-page test set: whole pages of text photographed the way the glass sees them, with the typed text.

Each page: 4-10 lines of Bangla, English or mixed text in a TEST font (bench.BN_FONT / EN_FONT "test": Nirmala,
Noto Sans Bengali, Tiro Bangla, Arial, Times, Calibri - none of them is a training font), printed dark on a light
sheet, laid on a table or held up, and seen by a 1280 x 720 camera:
  flat     page roughly facing the camera, small tilt
  angle    page seen at an angle (perspective), as when the head is not straight over it
  shadow   a hand / head shadow and uneven light across the page
  blur     slight defocus and motion, JPEG
Text: Bangla sentences from Bangla Wikipedia articles that are NOT in the training corpus (the last row groups of
the parquet file, checked against bn_sentences.txt); English from the 5,000 most frequent English words (the
renderer's word list), so English pages test letters, not language.
Pages are drawn once (seed fixed) and cached in rec5_runs/page_set.npz.

    python -m ocr_bench.rec5.page_set            (builds and prints a summary)
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from .. import bench as B
from .charset import encodable, normalize
from .fastshape import render_mask
from .paths import CORPUS, GLASS, RUNS

N_PAGES = 40
CONDS = ("flat", "angle", "shadow", "blur")


def _held_out_bn(n=400):
    import pyarrow.parquet as pq
    seen = set((CORPUS / "bn_sentences.txt").read_text(encoding="utf-8").splitlines())
    pf = pq.ParquetFile(sorted(CORPUS.glob("bnwiki_*.parquet"))[-1])
    out = []
    for g in range(pf.num_row_groups - 1, -1, -1):
        for doc in pf.read_row_group(g, columns=["text"]).column(0).to_pylist():
            for s in doc.replace("\n", " ").split("।"):
                s = normalize(s)
                if 15 <= len(s) <= 140 and encodable(s) and s + "।" not in seen and s not in seen:
                    if sum("ঀ" <= c <= "৿" for c in s) > 0.6 * len(s.replace(" ", "")):
                        out.append(s + "।")
            if len(out) >= n:
                return out
    return out


def _en_words():
    words = []
    for line in (GLASS / "ocr_bench" / "lexicon" / "en_50k.txt").read_text(encoding="utf-8").splitlines()[:5000]:
        w = line.rsplit(" ", 1)[0]
        if w.isalpha() and w.isascii() and len(w) > 1:
            words.append(w)
    return words


def _wrap(tokens, font, size, max_w):
    lines, cur = [], []
    for t in tokens:
        trial = " ".join(cur + [t])
        if cur and render_mask(trial, font, size).shape[1] > max_w:
            lines.append(" ".join(cur))
            cur = [t]
        else:
            cur.append(t)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _page(rng, lang, bn_sents, en_words):
    fonts = B.BN_FONT["test"] if lang in ("bn", "mixed") else B.EN_FONT["test"]
    font, idx = fonts[rng.randrange(len(fonts))]
    size = rng.choice((26, 30, 34))
    W = 900
    if lang == "bn":
        toks = " ".join(rng.sample(bn_sents, 3)).split()
    elif lang == "en":
        toks = [rng.choice(en_words) for _ in range(60)]
        toks[0] = toks[0].capitalize()
    else:
        toks = []
        for s in rng.sample(bn_sents, 2):
            ws = s.split()
            for _ in range(3):
                ws.insert(rng.randrange(len(ws) + 1), rng.choice(en_words))
            toks += ws
    lines = _wrap(toks, font, size, W - 120)[: rng.randint(4, 10)]
    masks = [render_mask(l, font, size, idx) for l in lines]
    lh = int(size * 1.9)
    H = 120 + lh * len(lines)
    page = np.full((H, W), rng.uniform(225, 250), np.float32)
    ink = rng.uniform(10, 50)
    y = 60
    for m in masks:
        mm = m.astype(np.float32) / 255
        h, w = mm.shape
        page[y:y + h, 60:60 + w] = page[y:y + h, 60:60 + w] * (1 - mm) + ink * mm
        y += lh
    return page, lines


def _scene(rng, page, cond, close=False):
    H, W = 720, 1280
    table = np.full((H, W, 3), [rng.uniform(40, 140) for _ in range(3)], np.float32)
    table += cv2.GaussianBlur(np.random.default_rng(rng.randrange(1 << 30)).normal(0, 12, (H, W)).astype(np.float32),
                              (0, 0), 3)[..., None]
    ph, pw = page.shape
    s = rng.uniform(0.85, 1.15) * min(0.9 * W / pw, 0.9 * H / ph)
    if close:                                     # held close: the page is larger than the view and runs off it
        s *= rng.uniform(1.25, 1.6)
    cw, ch = pw * s, ph * s
    cx, cy = W / 2 + rng.uniform(-60, 60), H / 2 + rng.uniform(-30, 30)
    dst = np.float32([[cx - cw / 2, cy - ch / 2], [cx + cw / 2, cy - ch / 2], [cx + cw / 2, cy + ch / 2], [cx - cw / 2, cy + ch / 2]])
    a = math.radians(rng.uniform(-4, 4) if cond != "angle" else rng.uniform(-10, 10))
    R = np.float32([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    dst = (dst - [cx, cy]) @ R.T + [cx, cy]
    if cond == "angle":                                       # one side further away
        k = rng.uniform(0.08, 0.16) * cw
        if rng.random() < 0.5:
            dst[0, 0] += k; dst[1, 0] -= k                     # top edge shorter
        else:
            dst[1, 1] += k * 0.5; dst[2, 1] -= k * 0.5          # right edge shorter
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), dst.astype(np.float32))
    pg = cv2.warpPerspective(np.repeat(page[..., None], 3, 2), M, (W, H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.ones((ph, pw), np.float32), M, (W, H))[..., None]
    img = table * (1 - mask) + pg * mask
    light = np.ones((H, W), np.float32)
    if cond == "shadow":
        gx = np.linspace(rng.uniform(0.45, 0.7), 1.0, W, dtype=np.float32)
        if rng.random() < 0.5:
            gx = gx[::-1]
        light = np.repeat(gx[None], H, 0)
        poly = np.int32([[rng.randrange(0, W // 2), 0], [rng.randrange(W // 2, W), 0], [rng.randrange(W // 3, W), H // 2],
                         [rng.randrange(0, W // 2), H // 3]])
        sh = np.ones((H, W), np.float32)
        cv2.fillPoly(sh, [poly], rng.uniform(0.55, 0.75))
        light *= cv2.GaussianBlur(sh, (0, 0), 25)
    else:
        light *= np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.85, 1.0), W, dtype=np.float32)[None]
    img = img * light[..., None]
    sigma = rng.uniform(0.6, 1.0) if cond != "blur" else rng.uniform(1.2, 1.8)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    if cond == "blur":
        k = rng.randint(3, 6)
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, ker)
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 5), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 88) if cond == "blur" else 85])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def build(split: str = "test"):
    """split "test" (the 40 pages of the report, seed 2026), "val" (40 other pages, seed 7) or "val_close" (40 pages
    held close, larger than the view, mostly at an angle, seed 11). Settings are chosen on the two validation splits.
    Text cut off by the picture edge stays in the reference: reading it is impossible, so CER there has a floor."""
    f = RUNS / ("page_set.npz" if split == "test" else f"page_set_{split}.npz")
    if f.exists():
        return list(np.load(f, allow_pickle=True)["items"])
    rng = random.Random({"test": 2026, "val": 7, "val_close": 11}[split])
    close = split == "val_close"
    bn, en = _held_out_bn(), _en_words()
    items = []
    for i in range(N_PAGES):
        lang = ("bn", "en", "mixed", "bn")[i % 4]
        cond = CONDS[(i // 4) % len(CONDS)]
        page, lines = _page(rng, lang, bn, en)
        if close and cond == "flat":
            cond = "angle"                         # close-ups: flat / angle / shadow / blur -> angle-heavy
        img = _scene(rng, page, cond, close)
        ok, enc = cv2.imencode(".png", img)
        items.append({"lang": lang, "cond": cond, "lines": lines, "png": enc.tobytes()})
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, items=np.array(items, dtype=object))
    return items


def decode(it):
    return cv2.imdecode(np.frombuffer(it["png"], np.uint8), cv2.IMREAD_COLOR)


def page_cer(gt_lines, hyp_text):
    return B.cer(" ".join(gt_lines), " ".join(hyp_text.split()))


if __name__ == "__main__":
    items = build()
    print(len(items), "pages;", {c: sum(i["cond"] == c for i in items) for c in CONDS},
          {l: sum(i["lang"] == l for i in items) for l in ("bn", "en", "mixed")})
    cv2.imwrite(str(RUNS / "page_example.jpg"), decode(items[5]))
    print(items[5]["lines"])
