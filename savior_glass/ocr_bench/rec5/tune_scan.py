"""Choose the scanner settings on VALIDATION data: the four-point text warp and the enhancement filters.

    python -m ocr_bench.rec5.tune_scan
Scored per candidate (lower is better):
  pages      page_set "val" (40 pages that fit the view) and "val_close" (40 pages held close, mostly at an angle,
             running off the view: the case the text warp is for)
  rendered   the rendered benchmark's validation set (432 images: clean / wild / scene), so a filter that helps pages
             but hurts signs is seen
Rule (written before the run): keep the candidate with the lowest mean of (val pages, val_close pages, rendered val);
a candidate must not be worse than the baseline by more than 0.3 points on any of the three. The test pages are not
read here. Output: results/ocr_bench/scan_tune.json
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
    "A baseline (no text warp, no filter)": {"OCR5_TEXT_WARP": False, "OCR5_ENHANCE": "none"},
    "B text warp": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "none"},
    "C warp + even light": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "even"},
    "D warp + soft even light": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "evensoft"},
    "E warp + CLAHE": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "clahe"},
    "F warp + sharpen": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "sharp"},
    "G warp + denoise + sharpen": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "denoise+sharp"},
    "H warp + even light + sharpen": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "even+sharp"},
    "I warp + finder sees even light": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "none", "OCR5_ENHANCE_FIND": "even"},
    # second round (added after the first run, still validation only): the winner I with the best reading filter G
    "J = I + read denoise + sharpen": {"OCR5_TEXT_WARP": True, "OCR5_ENHANCE": "denoise+sharp", "OCR5_ENHANCE_FIND": "even"},
}


def main():
    import config
    from ocr_text_region import clean_text
    from ocr_v5 import OcrV5
    eng = OcrV5()
    pages = {s: build(s) for s in ("val", "val_close")}
    base = {k: getattr(config, k) for k in ("OCR5_TEXT_WARP", "OCR5_ENHANCE", "OCR5_ENHANCE_FIND")}
    res = {}
    for name, cfg in CANDIDATES.items():
        for k, v in base.items():
            setattr(config, k, v)
        for k, v in cfg.items():
            setattr(config, k, v)
        t0 = time.time()
        r = {}
        for s, items in pages.items():
            cs = [page_cer(it["lines"], " ".join(eng.read(decode(it))["lines"])) for it in items]
            r[s] = round(100 * float(np.mean(cs)), 3)
        rv = B.score(lambda f: clean_text(" ".join(eng.read(f)["lines"])), "val")
        r["rendered"] = round(100 * rv["overall"]["cer"], 3)
        r["rendered_scene"] = round(100 * rv["by_cond"]["scene"]["all"]["cer"], 3)
        r["mean"] = round((r["val"] + r["val_close"] + r["rendered"]) / 3, 3)
        r["minutes"] = round((time.time() - t0) / 60, 1)
        res[name] = r
        print(name, r, flush=True)
    a = res["A baseline (no text warp, no filter)"]
    ok = {n: r for n, r in res.items() if all(r[k] <= a[k] + 0.3 for k in ("val", "val_close", "rendered"))}
    choice = min(ok, key=lambda n: ok[n]["mean"])
    out = {"rule": __doc__.split("Rule (written before the run): ")[1].split(" The test pages")[0], "results": res,
           "admissible": list(ok), "chosen": choice, "chosen_settings": CANDIDATES[choice]}
    (OUT / "scan_tune.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("CHOSEN:", choice, res[choice])


if __name__ == "__main__":
    main()
