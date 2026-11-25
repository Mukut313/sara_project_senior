"""Rendered training lines that look like what the glass camera sends the recogniser.

A line of text (Bangla Wikipedia, frequency word lists, numbers / prices / phone numbers / medicine labels, mixed
Bangla + English, random strings) is shaped with HarfBuzz (shaped_text.render_mask), put on paper, a sign, a coloured
label or a COCO photo patch, tilted and warped a little, then passed through a model of the camera: shrunk to a small
text height (the C270 at reading distance), defocus / motion blur, noise, JPEG, uneven light and shadow. The crop is
cut like a text detector would cut it (loose or tight, sometimes with bits of the neighbouring lines) and scaled to
the recogniser's height.

Fonts: only TRAIN fonts. The benchmark's validation and test fonts (bench.BN_FONT / EN_FONT "val" and "test",
including every Arial, Times, Calibri, Georgia, Verdana and Nirmala file) are never rendered here, so the rendered
benchmark stays a fair test.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .fastshape import render_mask
from .charset import encodable, normalize
from .paths import COCO, CORPUS, GLASS

H = 48                     # recogniser input height
MAX_W = 800                # widest training line

_FONTS_DIR = GLASS / "ocr_bench" / "fonts"
_BN_TRAIN = ["HindSiliguri-Regular.ttf", "AnekBangla.ttf", "BalooDa2.ttf", "Galada-Regular.ttf", "Mina-Regular.ttf",
             "Atma-Regular.ttf"]
_EXCLUDE_EN = ("arial", "times", "calibri", "georgia", "verdana", "nirmala")     # benchmark val/test families
_SKIP_EN = ("symbol", "wingding", "webding", "marlett", "mt extra", "holomdl", "segmdl", "segoe mdl", "segoeicons",
            "bssym", "refspcl", "outlook", "msreferencespecialty", "sylfaen",
            # decorative / handwriting-script display faces: rare on signs and labels, they only confuse the reader
            "alger", "bauhs", "bkant", "blackadd", "brushsci", "chiller", "curlz", "edwardsc", "frscript", "freescpt",
            "gigi", "harlowsi", "harngton", "itckrist", "jokerman", "kunstler", "magnetob", "oldengl", "parchm",
            "ravie", "vivaldi", "viner", "vladimir", "mistral", "palscri", "script", "matura", "inkfree", "pristina",
            "bradhitc", "forte", "informal", "jucs", "ltype", "playbill", "snap", "stencil", "tempsitc", "chiller",
            "showcard", "goudysto", "ocraext", "pertili", "rage", "holomdl", "segoeprb", "segoepr", "segoesc",
            "gabriola", "mvboli", "msuighur", "webdings", "wingding", "marlett", "gothic", "papyrus", "elephnt",
            "castelar", "colonna", "engr", "felixti", "gilsanub", "goudyst", "hatten", "impact", "niageng",
            "niagsol", "onyx", "poster", "rockeb", "tcceb", "tcm", "tcb")


def _has_glyphs(path, text):
    import freetype
    ft = _ft(path)
    return all(ft.get_char_index(ord(c)) for c in text if not c.isspace())


@lru_cache(maxsize=256)
def _ft(path):
    import freetype
    return freetype.Face(path)


def bn_fonts():
    fs = [str(_FONTS_DIR / f) for f in _BN_TRAIN]
    fs += [str(p) for p in sorted((CORPUS / "fonts").glob("*.ttf"))]
    return [f for f in fs if Path(f).exists()]


def en_fonts():
    win = Path(r"C:\Windows\Fonts")
    out = []
    for p in sorted(win.glob("*.tt[fc]")):
        n = p.name.lower()
        if n.startswith(_EXCLUDE_EN) or any(s in n for s in _SKIP_EN) or p.suffix.lower() == ".ttc":
            continue
        try:
            if _has_glyphs(str(p), "AaBbZz09%&@"):
                out.append(str(p))
        except Exception:  # noqa: BLE001
            continue
    return out


# ------------------------------------------------------------------ text
class Sentences:
    """The sentence file memory-mapped, with a line-offset index built once (bn_sentences.idx.npy): every data
    worker shares the same pages instead of holding 1.5 M Python strings (~0.5 GB each)."""

    def __init__(self, path: Path):
        self.ok = path.exists()
        if not self.ok:
            return
        idx = path.with_suffix(".idx.npy")
        if not idx.exists() or idx.stat().st_mtime < path.stat().st_mtime:
            raw = np.fromfile(path, np.uint8)
            ends = np.flatnonzero(raw == 10)
            starts = np.concatenate([[0], ends[:-1] + 1])
            np.save(idx, np.stack([starts, ends]).T.astype(np.int64))
        self.idx = np.load(idx, mmap_mode="r")
        self.raw = np.memmap(path, np.uint8, mode="r")

    def __bool__(self):
        return self.ok and len(self.idx) > 0

    def choice(self, rng) -> str:
        a, b = self.idx[rng.randrange(len(self.idx))]
        return bytes(self.raw[a:b]).decode("utf-8", "ignore").strip()


class TextSampler:
    def __init__(self, rng: random.Random):
        self.rng = rng
        lex = GLASS / "ocr_bench" / "lexicon"
        self.bn_words, self.bn_w = self._freq(lex / "bn_50k.txt")
        self.en_words, self.en_w = self._freq(lex / "en_50k.txt")
        self.bn_sent = Sentences(CORPUS / "bn_sentences.txt")

    @staticmethod
    def _freq(path):
        words, w = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.rsplit(" ", 1)
            if len(parts) == 2 and encodable(normalize(parts[0])) and parts[0].strip():
                words.append(normalize(parts[0]))
                w.append(float(parts[1]) ** 0.4)          # flatten Zipf: rare words still appear
        cw = np.cumsum(w)
        return words, cw / cw[-1]

    def _pick(self, words, cw, k):
        idx = np.searchsorted(cw, [self.rng.random() for _ in range(k)])
        return [words[min(i, len(words) - 1)] for i in idx]

    def _k(self, hi=8):
        return min(hi, max(1, int(self.rng.expovariate(1 / 3.0)) + 1))

    def _case(self, s):
        r = self.rng.random()
        return s.upper() if r < 0.22 else (s.title() if r < 0.45 else (s.capitalize() if r < 0.6 else s))

    def _punct(self, toks):
        r = self.rng
        if r.random() < 0.3:
            i = r.randrange(len(toks))
            toks[i] = toks[i] + r.choice([",", ".", ":", ";", "!", "?", "-", ")", "'s", "%"])
        if r.random() < 0.08:
            i = r.randrange(len(toks))
            toks[i] = r.choice(["(", '"', "'", "[", "#", "*"]) + toks[i]
        return toks

    def number(self):
        r = self.rng
        bd = r.random() < 0.5
        digits = "০১২৩৪৫৬৭৮৯" if bd else "0123456789"
        d = lambda n: "".join(r.choice(digits) for _ in range(n))          # noqa: E731
        kind = r.randrange(11)
        if kind == 0:
            return r.choice(["৳", "Tk ", "Tk. ", "BDT ", "টাকা "]) + d(r.randint(1, 5)) + r.choice(["", ".00", "/-", " টাকা"])
        if kind == 1:
            return r.choice(["+880 ", "0", "০"]) + d(4) + r.choice(["-", " ", ""]) + d(6)
        if kind == 2:
            return f"{d(2)}{r.choice('/-.')}{d(2)}{r.choice('/-.')}{d(4)}"
        if kind == 3:
            return f"{d(2)}:{d(2)}" + r.choice(["", " AM", " PM", " am", " pm"])
        if kind == 4:
            return d(r.randint(1, 4)) + r.choice([" mg", "mg", " ml", " gm", " kg", "%", " km", " MG", " Tablet", " টি"])
        if kind == 5:
            return r.choice(["Exp: ", "EXP ", "Mfg: ", "MFD: ", "Batch No: ", "B.No. ", "Reg. No. "]) + d(r.randint(2, 6))
        if kind == 6:
            return d(r.randint(1, 3)) + "." + d(r.randint(1, 2))
        if kind == 7:
            return r.choice(["Room ", "Road ", "House ", "রোড ", "বাড়ি ", "কক্ষ ", "No. ", "নং "]) + d(r.randint(1, 3))
        if kind == 8:
            return d(r.randint(1, 3)) + r.choice([",", ""]) + d(3)
        if kind == 9:
            return d(r.randint(4, 12))
        return " ".join(d(r.randint(1, 4)) for _ in range(r.randint(2, 4)))

    def random_string(self):
        r = self.rng
        if r.random() < 0.5:
            pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/.:#&@()+"
            return " ".join("".join(r.choice(pool) for _ in range(r.randint(2, 8))) for _ in range(r.randint(1, 4)))
        cons = "কখগঘচছজঝটঠডঢণতথদধনপফবভমযরলশষসহ"
        vs = ["", "া", "ি", "ী", "ু", "ূ", "ে", "ো", "ৈ", "ৌ", "ৃ", "ং", "্র", "্য", "ঁ"]
        return " ".join("".join(r.choice(cons) + r.choice(vs) for _ in range(r.randint(1, 4))) for _ in range(r.randint(1, 4)))

    def sample(self) -> str:
        r = self.rng.random()
        if r < 0.36 and self.bn_sent:
            toks = self.bn_sent.choice(self.rng).split() or ["।"]
            k = min(len(toks), self._k(10))
            i = self.rng.randrange(len(toks) - k + 1)
            s = " ".join(toks[i:i + k])
        elif r < 0.48:
            s = " ".join(self._pick(self.bn_words, self.bn_w, self._k(6)))
        elif r < 0.74:
            s = " ".join(self._punct([self._case(w) for w in self._pick(self.en_words, self.en_w, self._k(7))]))
        elif r < 0.85:
            s = self.number()
            if self.rng.random() < 0.4:
                pre = self._pick(self.bn_words, self.bn_w, 1) if self.rng.random() < 0.5 else \
                    [self._case(w) for w in self._pick(self.en_words, self.en_w, 1)]
                s = pre[0] + " " + s
        elif r < 0.93:
            a = self._pick(self.bn_words, self.bn_w, self._k(3))
            b = [self._case(w) for w in self._pick(self.en_words, self.en_w, self._k(3))]
            toks = a + b
            self.rng.shuffle(toks)
            s = " ".join(toks)
        else:
            s = self.random_string()
        s = normalize(s)
        if len(s) > 64:
            s = s[:64].rsplit(" ", 1)[0] or s[:64]
        return s


# ------------------------------------------------------------------ picture
_COCO_FILES: list = []


def _coco_patch(rng, h, w):
    global _COCO_FILES
    if not _COCO_FILES and COCO.is_dir():
        _COCO_FILES = sorted(COCO.glob("*.jpg"))[:3000]
    if not _COCO_FILES:
        return None
    im = cv2.imread(str(rng.choice(_COCO_FILES)))
    if im is None:
        return None
    ih, iw = im.shape[:2]
    s = max(h / ih, w / iw) * rng.uniform(1.0, 2.5)
    im = cv2.resize(im, (max(w, int(iw * s) + 1), max(h, int(ih * s) + 1)))
    y, x = rng.randrange(im.shape[0] - h + 1), rng.randrange(im.shape[1] - w + 1)
    return im[y:y + h, x:x + w].astype(np.float32)


def _background(rng, h, w):
    r = rng.random()
    if r < 0.12:
        p = _coco_patch(rng, h, w)
        if p is not None:
            return cv2.GaussianBlur(p, (0, 0), rng.uniform(1.5, 6)), "photo"
    if r < 0.62:
        c = np.array([rng.uniform(175, 255)] * 3) + np.array([rng.uniform(-18, 18) for _ in range(3)])   # paper
        kind = "light"
    elif r < 0.80:
        c = np.array([rng.uniform(0, 80) for _ in range(3)])                                                # dark sign
        kind = "dark"
    else:
        c = np.array([rng.uniform(0, 255) for _ in range(3)])                                               # coloured
        kind = "color"
    bg = np.ones((h, w, 3), np.float32) * np.clip(c, 0, 255)
    if rng.random() < 0.5:                                     # paper texture / print noise
        n = np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 12), (h, w)).astype(np.float32)
        bg += cv2.GaussianBlur(n, (0, 0), rng.uniform(0.5, 3))[..., None]
    if rng.random() < 0.3:                                     # gradient
        g = np.linspace(rng.uniform(-30, 0), rng.uniform(0, 30), w if rng.random() < 0.5 else h, dtype=np.float32)
        bg += (g[None, :, None] if g.shape[0] == w else g[:, None, None])
    return np.clip(bg, 0, 255), kind


def _ink(rng, kind, bg_mean):
    if kind == "dark":
        base = rng.uniform(170, 255)
    elif kind == "light":
        base = rng.uniform(0, 90)
    else:
        base = 255 - bg_mean + rng.uniform(-60, 60) if rng.random() < 0.8 else rng.uniform(0, 255)
    c = np.clip(np.array([base + rng.uniform(-40, 40) for _ in range(3)]), 0, 255)
    if rng.random() < 0.12:                                    # low contrast print
        c = bg_mean + (c - bg_mean) * rng.uniform(0.3, 0.55)
    gap = float(np.mean(c)) - bg_mean
    if abs(gap) < 45:                                          # keep at least some contrast between ink and paper
        c = np.clip(c + np.sign(gap or 1) * (45 - abs(gap)), 0, 255)
    return c.astype(np.float32)


def _warp(img, rng, fill):
    h, w = img.shape[:2]
    if rng.random() < 0.45:                                    # small tilt (detector boxes are not perfectly level)
        a = rng.uniform(-2.5, 2.5)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    if rng.random() < 0.3:                                     # perspective (page not flat to the camera)
        j = 0.06 * h
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[rng.uniform(-j, j), rng.uniform(-j, j)] for _ in range(4)])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    if rng.random() < 0.1:                                     # curved page
        amp, per = rng.uniform(1, 0.06 * h + 1), rng.uniform(1.0, 3.0) * w
        mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        my = my + amp * np.sin(2 * math.pi * mx / per + rng.uniform(0, 6.28)).astype(np.float32)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img


def _camera(img, rng):
    """img: float32 BGR at render size. Returns uint8 grey at camera size."""
    h, w = img.shape[:2]
    # text line height on the camera sensor: 9..40 px, mostly small (log-uniform)
    target = math.exp(rng.uniform(math.log(9), math.log(42)))
    s = min(1.0, target / h)
    small = cv2.resize(img, (max(4, int(w * s)), max(4, int(h * s))), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.55:
        # defocus (fixed-focus webcam); small text gets less, so the line stays readable to a person
        small = cv2.GaussianBlur(small, (0, 0), rng.uniform(0.3, min(1.4, 0.3 + 0.045 * target)))
    if rng.random() < 0.15:                                                       # motion blur (head movement)
        k = rng.randint(2, 5)
        ker = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            ker[k // 2, :] = 1
        else:
            ker[:, k // 2] = 1
        small = cv2.filter2D(small, -1, ker / k)
    if rng.random() < 0.35:                                                       # uneven light / shadow
        sh, sw = small.shape[:2]
        gx = np.linspace(rng.uniform(0.55, 1.0), rng.uniform(0.55, 1.0), sw, dtype=np.float32)
        gy = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), sh, dtype=np.float32)
        small = small * (gy[:, None] * gx[None, :])[..., None]
    if rng.random() < 0.2:
        g = rng.uniform(0.6, 1.6)
        small = 255 * (np.clip(small, 0, 255) / 255) ** g
    noise = rng.uniform(0, 9)
    small = small + np.random.default_rng(rng.randrange(1 << 30)).normal(0, noise, small.shape).astype(np.float32)
    small = np.clip(small, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 92)])
        small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if rng.random() < 0.3:
        grey = cv2.createCLAHE(clipLimit=rng.uniform(1.5, 3.0), tileGridSize=(2, 8)).apply(grey)
    return grey


def to_height(grey, rng=None):
    h, w = grey.shape[:2]
    nw = max(8, int(round(w * H / h)))
    interp = cv2.INTER_CUBIC if (rng is None or rng.random() < 0.6) else cv2.INTER_LINEAR
    if h > H:
        interp = cv2.INTER_AREA
    return cv2.resize(grey, (nw, H), interpolation=interp)


class LineRenderer:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.text = TextSampler(self.rng)
        self.bn = bn_fonts()
        self.en = en_fonts()
        self.mixed_ok = {}

    def _font_for(self, s):
        has_bn = any("\u0980" <= c <= "\u09ff" for c in s)
        has_lat = any(c.isascii() and c.isalpha() or "\u00c0" <= c <= "\u00ff" for c in s)
        pool = self.bn if has_bn else (self.en + self.bn if self.rng.random() < 0.15 else self.en)
        for _ in range(6):
            f = self.rng.choice(pool)
            if not has_lat and not has_bn:
                if _has_glyphs(f, s):
                    return f
            elif _has_glyphs(f, s):
                return f
        return None

    def sample(self):
        """(text, uint8 grey image at height H) or None when this draw failed."""
        try:
            return self._sample()
        except Exception:  # noqa: BLE001  (a font that can't draw a glyph, a degenerate crop): skip this draw
            return None

    def _sample(self):
        rng = self.rng
        s = self.text.sample()
        if not s.strip():
            return None
        font = self._font_for(s)
        if font is None:
            return None
        size = rng.choice((28, 34, 40, 48, 56))          # few sizes: the glyph cache hits; the camera step rescales
        m = render_mask(s, font, size).astype(np.float32) / 255
        if m.shape[0] < 6 or m.shape[1] < 6:
            return None
        if rng.random() < 0.12:
            m = cv2.dilate(m, np.ones((2, 2), np.uint8))                        # bold
        if rng.random() < 0.1:                                                  # italic
            sh = rng.uniform(-0.25, 0.25)
            mh, mw = m.shape
            M = np.float32([[1, sh, -sh * mh / 2 if sh > 0 else -sh * mh / 2], [0, 1, 0]])
            m = cv2.warpAffine(m, M, (mw + int(abs(sh) * mh) + 2, mh))
        mh, mw = m.shape
        # detector-like crop margins (in line heights)
        lh = max(mh, size)
        top, bot = int(lh * rng.uniform(0.05, 0.45)), int(lh * rng.uniform(0.05, 0.45))
        left, right = int(lh * rng.uniform(0.0, 0.6)), int(lh * rng.uniform(0.0, 0.6))
        Hc, Wc = mh + top + bot, mw + left + right
        bg, kind = _background(rng, Hc, Wc)
        ink = _ink(rng, kind, float(bg.mean()))
        mask = np.zeros((Hc, Wc), np.float32)
        mask[top:top + mh, left:left + mw] = m
        if rng.random() < 0.15:                                                 # bits of the neighbouring lines
            other = render_mask(self.text.sample() or "a", font, size).astype(np.float32) / 255
            oh, ow = other.shape
            x = rng.randint(0, max(0, Wc - 1))
            wcut = min(ow, Wc - x)
            if rng.random() < 0.5:
                cut = min(oh, Hc, rng.randint(1, max(2, top + 2)))
                mask[:cut, x:x + wcut] = np.maximum(mask[:cut, x:x + wcut], other[oh - cut:, :wcut][:cut])
            else:
                cut = min(oh, Hc, rng.randint(1, max(2, bot + 2)))
                mask[Hc - cut:, x:x + wcut] = np.maximum(mask[Hc - cut:, x:x + wcut], other[:cut, :wcut])
        if rng.random() < 0.05:                                                 # underline / box edge
            y = rng.choice([top + mh + 2, Hc - 2, 1])
            mask[max(0, y - 1):y + 1, :] = 1
        a = rng.uniform(0.75, 1.0)
        img = bg * (1 - a * mask[..., None]) + ink[None, None, :] * (a * mask[..., None])
        img = _warp(img, rng, tuple(float(v) for v in bg.reshape(-1, 3).mean(0)))
        grey = _camera(img, rng)
        if rng.random() < 0.04:
            cut = rng.uniform(0.0, 0.12)
            gh = grey.shape[0]
            grey = grey[int(gh * cut):] if rng.random() < 0.5 else grey[:gh - int(gh * cut)]
        out = to_height(grey, rng)
        if out.shape[1] > MAX_W:
            if out.shape[1] > MAX_W * 1.25:
                return None
            out = cv2.resize(out, (MAX_W, H), interpolation=cv2.INTER_AREA)
        return s, out
