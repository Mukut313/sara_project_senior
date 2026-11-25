"""Read the camera-page test set (page_set.py) with one reading option and score it.

    python -m ocr_bench.rec5.page_eval v4            the glass's reader before OCR v5 (EasyOCR pipeline v2/v4)
    python -m ocr_bench.rec5.page_eval v5_easyocr    OCR v5 front end + the EasyOCR letter reader
    python -m ocr_bench.rec5.page_eval v5_sara       OCR v5 front end + the SARA line reader (models/ocr5_rec)

CPU only, 4 threads (like the Pi's 4 cores; laptop times are NOT Pi times). Score: character error of the whole page
(lines joined in reading order, whitespace collapsed), per page, averaged; also per condition and language.
Output: results/ocr_bench/page_eval_<option>.json
"""
from __future__ import annotations

import json
import os
import sys
import time

opt = sys.argv[1] if len(sys.argv) > 1 else "v5_sara"
os.environ["OCR_GPU"] = "0"
os.environ["OCR5_THREADS"] = "4"
if opt == "v4":
    os.environ["OCR_PIPELINE"] = "v2"
else:
    os.environ["OCR_PIPELINE"] = "v5"
    os.environ["OCR5_RECOGNIZER"] = "easyocr" if opt == "v5_easyocr" else "sara"

import numpy as np  # noqa: E402

from ..page_bench import OUT  # noqa: E402
from .page_set import build, decode, page_cer  # noqa: E402


def main():
    import torch
    torch.set_num_threads(4)
    from ocr_text_region import clean_text
    from modes.ocr_mode import OCRMode
    m = OCRMode()
    m.speak_line = None
    m.activate()
    if opt == "v4":
        def read(frame):
            return clean_text(" ".join(m._read_v2(frame)))
    else:
        def read(frame):
            return " ".join(m._v5.read(frame)["lines"])
    read(decode(build()[0]))                                   # warm up
    rows = []
    for it in build():
        img = decode(it)
        t0 = time.perf_counter()
        hyp = read(img)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({"lang": it["lang"], "cond": it["cond"], "cer": page_cer(it["lines"], hyp), "ms": ms,
                     "gt": " ".join(it["lines"]), "hyp": hyp})
        print(f"{it['lang']:5s} {it['cond']:6s} CER {rows[-1]['cer']:.3f}  {ms:6.0f} ms", flush=True)
    agg = lambda rs: round(float(np.mean([r["cer"] for r in rs])), 4) if rs else None   # noqa: E731
    res = {"option": opt, "pages": len(rows), "cer": agg(rows),
           "by_lang": {l: agg([r for r in rows if r["lang"] == l]) for l in ("bn", "en", "mixed")},
           "by_cond": {c: agg([r for r in rows if r["cond"] == c]) for c in ("flat", "angle", "shadow", "blur")},
           "ms_median": round(float(np.median([r["ms"] for r in rows]))), "rows": rows}
    if opt != "v4":
        import config
        res["settings"] = {k: getattr(config, k) for k in dir(config) if k.startswith("OCR5_")}
    (OUT / f"page_eval_{opt}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "settings")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
