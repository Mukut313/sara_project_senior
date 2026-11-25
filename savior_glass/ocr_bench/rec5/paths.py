"""Where the recogniser's training data lives. The datasets are not in the repository; they sit in the project's
"data set" folder. SARA_DATA_DIR overrides the search."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
GLASS = HERE.parents[1]                        # savior_glass/
REPO = GLASS.parent


def data_root() -> Path:
    cands = [os.environ.get("SARA_DATA_DIR", ""), str(REPO / "data set"), r"D:\SARA senior project\data set"]
    for c in cands:
        if c and (Path(c) / "mlt19_words").is_dir():
            return Path(c)
    return Path(cands[1])


DATA = data_root()
CORPUS = DATA / "ocr_corpus"
MLT = DATA / "mlt19_words"
COCO = DATA / "coco2017" / "val2017"
RUNS = GLASS / "ocr_bench" / "rec5_runs"       # checkpoints and logs (git-ignored)
