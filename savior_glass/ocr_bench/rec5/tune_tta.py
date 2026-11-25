"""Choose the test-time clean-up (OCR5_TTA) on VALIDATION data.

    python -m ocr_bench.rec5.tune_tta
Every candidate is scored on all four validation sets, so a choice that helps pages but hurts signs is seen:
  pages      page_set "val" and "val_close"
  rendered   bench validation set (432 images)
  real       the 300 real MLT validation words in a 640 x 480 frame
Rule (written before the run): lowest mean of the four CERs; a candidate may not be worse than "off" by more than
0.3 points on any of them. Output: results/ocr_bench/tta_tune.json
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("OCR_PIPELINE", "v5")
os.environ.setdefault("OCR5_THREADS", "4")

import numpy as np  # noqa: E402

from .. import bench as B  # noqa: E402
from ..page_bench import OUT  # noqa: E402
from .page_set import build, decode, page_cer  # noqa: E402

CANDIDATES = {
    "off": ("", 0.9),
    "sharp (unsure lines)": ("sharp", 0.9),
    "clahe (unsure lines)": ("clahe", 0.9),
    "sharp, clahe (unsure lines)": ("sharp,clahe", 0.9),
    "sharp, clahe, even (unsure lines)": ("sharp,clahe,even", 0.9),
    "denoise+sharp, clahe (unsure lines)": ("denoise+sharp,clahe", 0.9),
    "sharp, clahe (every line)": ("sharp,clahe", 1.01),
}


def main():
    import config
    from ocr_text_region import clean_text
    from ocr_v5 import OcrV5
    from ..ocr_v3_eval import score
    from ..real_words_tune import fetch
    from ..train_real_words import split
    eng = OcrV5()
    pages = {s: build(s) for s in ("val", "val_close")}
    _, va, _ = split()
    words = fetch(va)
    read = lambda f: clean_text(" ".join(eng.read(f)["lines"]))   # noqa: E731
    res = {}
    for name, (tta, below) in CANDIDATES.items():
        config.OCR5_TTA, config.OCR5_TTA_BELOW = tta, below
        t0 = time.time()
        r = {s: round(100 * float(np.mean([page_cer(it["lines"], " ".join(eng.read(decode(it))["lines"])) for it in items])), 3)
             for s, items in pages.items()}
        r["rendered"] = round(100 * B.score(read, "val")["overall"]["cer"], 3)
        rw = score(read, words)
        r["real"] = round(100 * rw["mean_cer"], 3)
        r["real_bn"], r["real_en"] = round(100 * rw["bn"]["cer"], 2), round(100 * rw["en"]["cer"], 2)
        r["mean"] = round((r["val"] + r["val_close"] + r["rendered"] + r["real"]) / 4, 3)
        r["minutes"] = round((time.time() - t0) / 60, 1)
        res[name] = r
        print(name, r, flush=True)
    base = res["off"]
    ok = {n: r for n, r in res.items() if all(r[k] <= base[k] + 0.3 for k in ("val", "val_close", "rendered", "real"))}
    choice = min(ok, key=lambda n: ok[n]["mean"])
    (OUT / "tta_tune.json").write_text(json.dumps({"results": res, "admissible": list(ok), "chosen": choice,
                                                   "chosen_settings": CANDIDATES[choice]}, indent=1), encoding="utf-8")
    print("CHOSEN:", choice, res[choice])


if __name__ == "__main__":
    main()
