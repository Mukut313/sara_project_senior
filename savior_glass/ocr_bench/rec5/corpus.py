"""Build the text corpus the line renderer draws from.

  bn_sentences.txt  sentences from Bangla Wikipedia (wikimedia/wikipedia 20231101.bn, CC BY-SA), cut at । ? !,
                    kept when every character is in the recogniser's character set
Run once:  python -m ocr_bench.rec5.corpus
"""
from __future__ import annotations

import re
import sys

from .charset import encodable, normalize
from .paths import CORPUS

_SPLIT = re.compile(r"(?<=[।?!])\s+|\n+")


def build_bn(limit: int = 1_500_000) -> None:
    import pyarrow.parquet as pq
    out = CORPUS / "bn_sentences.txt"
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for p in sorted(CORPUS.glob("bnwiki_*.parquet")):
            pf = pq.ParquetFile(p)
            for g in range(pf.num_row_groups):
                for doc in pf.read_row_group(g, columns=["text"]).column(0).to_pylist():
                    for s in _SPLIT.split(doc):
                        s = normalize(s)
                        if not 8 <= len(s) <= 220 or not encodable(s):
                            continue
                        bn = sum(1 for c in s if "ঀ" <= c <= "৿")
                        if bn < 0.5 * len(s.replace(" ", "")):
                            continue
                        f.write(s + "\n")
                        n += 1
                        if n >= limit:
                            print("sentences", n)
                            return
    print("sentences", n)


if __name__ == "__main__":
    build_bn(int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000)
