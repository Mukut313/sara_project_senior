"""Numbers for the OCR v5 report: confusion analysis, training curves, speed, architecture.

    python -m ocr_bench.rec5.report_data
Reads the saved result files (results/ocr_bench/*.json, rec5_runs/r1|r2/log.json) and writes
results/ocr_bench/ocr_v5_report_data.json. Nothing is re-run.

Character confusion: each reading is aligned to its reference with the edit-distance alignment (Unicode NFC, code
points). Aligned pairs give the confusion counts; a reference character with no partner is a deletion (missed, shown
as the column "∅"), a read character with no partner an insertion (extra).
"""
from __future__ import annotations

import collections
import json
import unicodedata

import torch

from ..page_bench import OUT
from .charset import CHARS
from .model import SaraRec
from .paths import GLASS, RUNS


def align(gt: str, hyp: str):
    """[(g, h)] with g or h = None for an insertion / deletion (Levenshtein backtrace)."""
    a, b = list(unicodedata.normalize("NFC", gt)), list(unicodedata.normalize("NFC", hyp))
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            out.append((a[i - 1], b[j - 1])); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((a[i - 1], None)); i -= 1
        else:
            out.append((None, b[j - 1])); j -= 1
    return out[::-1]


def confusion(rows, script):
    """rows [(gt, hyp)] -> counts per (g, h); script 'bn' or 'en' picks which reference characters are tallied."""
    keep = (lambda c: "ঀ" <= c <= "৿") if script == "bn" else (lambda c: c.isascii() and c.isalnum())
    pairs, freq = collections.Counter(), collections.Counter()
    sub = dele = ins = ok = 0
    for gt, hyp in rows:
        for g, h in align(gt, hyp):
            if g is None:
                if h is not None and keep(h):
                    ins += 1
                continue
            if not keep(g):
                continue
            freq[g] += 1
            pairs[(g, h)] += 1
            if h is None:
                dele += 1
            elif h == g:
                ok += 1
            else:
                sub += 1
    return pairs, freq, {"correct": ok, "substituted": sub, "missed": dele, "extra": ins}


def matrix(pairs, freq, n=14):
    top = [c for c, _ in freq.most_common(n)]
    cols = top + ["other", "∅"]
    m = []
    for g in top:
        row = []
        for h in top:
            row.append(pairs.get((g, h), 0))
        row.append(sum(v for (gg, hh), v in pairs.items() if gg == g and hh is not None and hh not in top))
        row.append(pairs.get((g, None), 0))
        m.append(row)
    return {"rows": top, "cols": cols, "counts": m, "support": [freq[c] for c in top]}


def top_errors(pairs, k=15):
    errs = [((g, h), v) for (g, h), v in pairs.items() if h != g]
    errs.sort(key=lambda x: -x[1])
    return [{"ref": g, "read": h if h is not None else "∅", "count": v} for (g, h), v in errs[:k]]


def arch():
    m = SaraRec().eval()
    rows, x = [], torch.zeros(1, 1, 48, 320)
    with torch.no_grad():
        x = m.stem(x); rows.append(("stem conv 3x3 /2", tuple(x.shape[1:]), sum(p.numel() for p in m.stem.parameters())))
        for i, blk in enumerate(m.stages):
            x = blk(x)
            rows.append((f"inverted residual {i + 1}", tuple(x.shape[1:]), sum(p.numel() for p in blk.parameters())))
        x = m.squash(x); rows.append(("height squash conv 3x1", tuple(x.shape[1:]), sum(p.numel() for p in m.squash.parameters())))
        for i, blk in enumerate(m.ctx):
            x = blk(x)
            rows.append((f"context block {i + 1} (dw 1x7 + FFN)", tuple(x.shape[1:]), sum(p.numel() for p in blk.parameters())))
        x = m.head(x); rows.append(("CTC head conv 1x1", tuple(x.shape[1:]), sum(p.numel() for p in m.head.parameters())))
    return [{"layer": a, "out": "x".join(map(str, s)), "params": p} for a, s, p in rows], sum(p.numel() for p in m.parameters())


def main():
    res = {}
    load = lambda p: json.loads((OUT / p).read_text(encoding="utf-8"))   # noqa: E731
    pages = {k: load(f"page_eval_{k}.json") for k in ("v4", "v5_easyocr", "v5_sara")}
    pages["v5_sara_before_scan"] = None
    pages.pop("v5_sara_before_scan")
    res["pages"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "settings")} for k, v in pages.items()}
    res["page_ms"] = {k: [round(r["ms"]) for r in v["rows"]] for k, v in pages.items()}
    for split in ("val", "test"):
        try:
            d = load(f"ocr_v5_{split}_scan.json" if split == "test" else f"ocr_v5_{split}_final.json")
            res[split] = {"rendered": {k: d["rendered"][k] for k in ("overall", "bn", "en", "by_cond", "latency_ms_median")},
                          "real_words": d["real_words"], "v4": d.get("ocr_v4_same_sets")}
        except FileNotFoundError:
            res[split] = None
    # confusion: v5 SARA reader on the camera pages and the real test words; v4 on the same pages for contrast
    rows_v5 = [(r["gt"], r["hyp"]) for r in pages["v5_sara"]["rows"]]
    rows_v4 = [(r["gt"], r["hyp"]) for r in pages["v4"]["rows"]]
    real_rows = []
    try:
        rr = load("ocr_v5_test_scan_rows.json")["real"]
        real_rows = [(r["gt"], r["hyp"]) for r in rr]
    except FileNotFoundError:
        pass
    res["confusion"] = {}
    for name, rows in (("pages_v5", rows_v5), ("pages_v4", rows_v4), ("real_test_words_v5", real_rows)):
        for sc in ("bn", "en"):
            if not rows:
                continue
            pairs, freq, kinds = confusion(rows, sc)
            res["confusion"][f"{name}_{sc}"] = {"kinds": kinds, "matrix": matrix(pairs, freq), "top_errors": top_errors(pairs)}
    # training curves
    res["training"] = {}
    for run in ("r1", "r2"):
        p = RUNS / run / "log.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            res["training"][run] = {"args": d["args"], "best": d["best"],
                                    "log": [{k: r[k] for k in ("step", "real_bn", "real_en", "rendered", "minutes")} for r in d["log"]]}
    res["architecture"], res["params"] = arch()
    res["charset_size"] = len(CHARS)
    res["model_json"] = json.loads((GLASS / "models" / "ocr5_rec" / "model.json").read_text(encoding="utf-8"))
    meta = json.loads((RUNS / "mlt_train" / "index.json").read_text(encoding="utf-8"))
    bn = sum(1 for m in meta if any("ঀ" <= c <= "৿" for c in m[3]))
    res["data"] = {"mlt_train_words": len(meta), "mlt_train_bn": bn, "mlt_train_latin": len(meta) - bn,
                   "bn_sentences": sum(1 for _ in open(RUNS.parents[2] / "data set" / "ocr_corpus" / "bn_sentences.txt",
                                                       encoding="utf-8")) if (RUNS.parents[2] / "data set" / "ocr_corpus" / "bn_sentences.txt").exists() else None}
    (OUT / "ocr_v5_report_data.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                                 encoding="utf-8")
    print("written", OUT / "ocr_v5_report_data.json")
    for k, v in res["confusion"].items():
        print(k, v["kinds"], v["top_errors"][:5])


if __name__ == "__main__":
    main()
