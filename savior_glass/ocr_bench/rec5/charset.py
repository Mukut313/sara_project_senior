"""Character set of the SARA line recogniser, and the label normalisation shared by training, evaluation and the glass.

Index 0 is the CTC blank. The list is written next to every exported model (charset.txt) and the glass reads it from
there, so changing this file only affects models trained afterwards.
"""
from __future__ import annotations

import unicodedata

_BN = (
    "ঁংঃ"                                   # ঁ ং ঃ
    "অআইঈউঊঋএঐওঔ"
    "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ"
    "়ঽ"                                          # nukta ় , avagraha ঽ
    "ািীুূৃৄ"            # া ি ী ু ূ ৃ ৄ
    "েৈোৌ্ৎৗ"            # ে ৈ ো ৌ ্ ৎ ৗ
    "০১২৩৪৫৬৭৮৯"
    "৳।॥"                                         # ৳ । ॥
)
_ASCII = "".join(chr(c) for c in range(0x21, 0x7F))        # printable ASCII without space (space is added below)
_LATIN1 = "".join(c for c in map(chr, range(0xC0, 0x100)) if c.isalpha())   # À..ÿ letters (French/German/... in MLT)
_EXTRA = "°•€£"

CHARS = [" "] + list(dict.fromkeys(_BN + _ASCII + _LATIN1 + _EXTRA))
BLANK = 0
INDEX = {c: i + 1 for i, c in enumerate(CHARS)}            # model class = list index + 1

_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
    "…": "...", " ": " ", "‌": "", "‍": "", "­": "", "﻿": "",
    "৷": "।",                                     # Bengali currency numerator four, sometimes used as danda
    "|": "|",
}


def normalize(text: str) -> str:
    """NFC, typographic quotes and dashes to ASCII, joiners removed, runs of spaces collapsed."""
    t = unicodedata.normalize("NFC", text)
    t = "".join(_MAP.get(c, c) for c in t)
    t = unicodedata.normalize("NFC", t)
    return " ".join(t.split())


def encodable(text: str) -> bool:
    return all(c in INDEX for c in text)


def encode(text: str) -> list[int]:
    return [INDEX[c] for c in text if c in INDEX]


def decode_greedy(ids, probs=None):
    """CTC greedy decode of one sequence of class ids -> (text, mean confidence of the kept characters)."""
    out, conf, prev = [], [], BLANK
    for t, k in enumerate(ids):
        k = int(k)
        if k != BLANK and k != prev:
            out.append(CHARS[k - 1])
            if probs is not None:
                conf.append(float(probs[t]))
        prev = k
    text = "".join(out)
    return " ".join(text.split()), (sum(conf) / len(conf) if conf else 0.0)
