"""OCR v5 through the glass's own reading function, on the same data and scoring as OCR v4 (ocr_v4_eval.py).

    python -m ocr_bench.rec5.eval_pipeline val  [label]     settings are chosen here
    python -m ocr_bench.rec5.eval_pipeline test [label]     read once, after the choice

  rendered   bench.score on the phrase set ("val" or "test"; 48 phrases x 3 conditions x 3 seeds, Bangla + English,
             clean / wild / scene) - val uses the validation fonts, test the test fonts; neither is a training font
  real words ICDAR 2019 MLT words placed in a 640 x 480 grey frame (real_words.frame_of), 150 + 150 validation words
             or the 500 + 500 locked test words (train_real_words.split)
OCR v4 (the glass's reader before this change, candidate SV) on the same sets: results/ocr_bench/ocr_v4_eval.json.
The ONNX Runtime CPU threads are limited to 4, as on the Pi (config.OCR5_THREADS). Settings come from config / the
environment (OCR5_* variables), so a candidate is scored by setting its variables before the run.
Output: results/ocr_bench/ocr_v5_<split>[_label].json
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("OCR5_THREADS", "4")
os.environ.setdefault("OCR_PIPELINE", "v5")

import numpy as np  # noqa: E402

from .. import bench as B  # noqa: E402
from ..page_bench import OUT  # noqa: E402


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    import config
    from ocr_text_region import clean_text
    from ocr_v5 import OcrV5
    from ..ocr_v3_eval import score as score_real
    from ..real_words_tune import fetch
    from ..train_real_words import split as mlt_split

    eng = OcrV5()
    eng.read(np.full((96, 320, 3), 255, np.uint8))

    def read(frame):
        r = eng.read(frame)
        return clean_text(" ".join(r["lines"]))

    t0 = time.time()
    rend = B.score(read, "val" if split == "val" else "test")
    rows = rend.pop("rows")
    _, va, te = mlt_split()
    words = fetch(va if split == "val" else te)
    real, real_rows = score_real(read, words, keep_rows=True)
    res = {"split": split, "label": label, "minutes": round((time.time() - t0) / 60, 1),
           "settings": {k: getattr(config, k) for k in dir(config) if k.startswith("OCR5_")},
           "rec_model": json.loads(open(os.path.join(config.OCR5_REC_DIR, "model.json"), encoding="utf-8").read())
           if os.path.exists(os.path.join(config.OCR5_REC_DIR, "model.json")) else None,
           "rendered": rend, "real_words": real}
    v4 = OUT / "ocr_v4_eval.json"
    if v4.exists():
        d = json.loads(v4.read_text(encoding="utf-8"))
        res["ocr_v4_same_sets"] = d.get("validation" if split == "val" else "test", {}).get("SV")
    name = f"ocr_v5_{split}" + (f"_{label}" if label else "")
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    (OUT / f"{name}_rows.json").write_text(json.dumps({"rendered": rows, "real": real_rows}, ensure_ascii=False,
                                                      indent=0, default=float), encoding="utf-8")
    print(json.dumps({"rendered_cer": rend["overall"]["cer"], "rendered_bn": rend["bn"]["cer"],
                      "rendered_en": rend["en"]["cer"], "by_cond": {c: v["all"]["cer"] for c, v in rend["by_cond"].items()},
                      "latency_ms_median": rend["latency_ms_median"], "real": real,
                      "v4": res.get("ocr_v4_same_sets")}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
