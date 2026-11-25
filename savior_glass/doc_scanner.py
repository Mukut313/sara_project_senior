"""Scanner front end for OCR v5: what a phone scanner app does before reading, on the glass.

  find_page(img)      the four corners of a page / label / sign that fills a good part of the picture, if any
  flatten(img, quad)  that page warped flat and upright (perspective removed)
  even_light(grey)    shadows and uneven light removed (divide by the blurred background), contrast stretched
  crop_line(img, q)   one detected text line cut out along its own tilt, ready for the recogniser
  deskew_angle(quads) the common tilt of the text lines, to straighten a picture without a page outline
  AutoCapture         decides when to read by itself: text in view, picture steady and sharp; keeps the sharpest
                      of the last frames; gives one short spoken hint when the text is too small, cut off or blurry

All plain OpenCV on the CPU; a few milliseconds per frame at the sizes used.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


LAST_FLATTEN_MARGIN = 0     # margin (px) the last flatten() call added around the page

# ------------------------------------------------------------------ page outline
def _order(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_page(img: np.ndarray, min_area: float = 0.15, max_area: float = 0.97) -> Optional[np.ndarray]:
    """Corners (tl, tr, br, bl) of the largest convex four-sided outline covering min_area..max_area of the picture,
    in img pixels; None when there is no clear page edge (then the picture is read as it is)."""
    h, w = img.shape[:2]
    s = 480.0 / max(h, w)
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    g = cv2.GaussianBlur(g, (5, 5), 0)
    best, best_a = None, 0.0
    total = g.shape[0] * g.shape[1]
    v = float(np.median(g))
    for lo, hi in ((int(max(0, 0.66 * v)), int(min(255, 1.33 * v))), (30, 90), (60, 160)):
        e = cv2.dilate(cv2.Canny(g, lo, hi), np.ones((3, 3), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.02, 0.035, 0.05):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    break
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            a = cv2.contourArea(ap) / total
            if min_area <= a <= max_area and a > best_a:
                q = _order(ap)
                angs = []                                        # reject very skewed shapes (not a page)
                for i in range(4):
                    p0, p1, p2 = q[i - 1], q[i], q[(i + 1) % 4]
                    v1, v2 = p0 - p1, p2 - p1
                    cosv = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
                    angs.append(math.degrees(math.acos(max(-1, min(1, cosv)))))
                if min(angs) > 55 and max(angs) < 125:
                    best, best_a = q, a
        if best is not None:
            break
    return None if best is None else best / s


def flatten(img: np.ndarray, quad: np.ndarray, max_side: int = 1600) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    k = min(1.0, max_side / max(w, h, 1))
    w, h = max(16, int(w * k)), max(16, int(h * k))
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    flat = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # a margin in the paper's own colour: text running to the edge of the sheet is otherwise cut into pieces or
    # missed by the text finder (seen on validation words, 2026-11-25)
    p = max(8, int(0.08 * max(w, h)))
    global LAST_FLATTEN_MARGIN
    LAST_FLATTEN_MARGIN = p
    edge = np.concatenate([flat[:2].reshape(-1, flat.shape[2] if flat.ndim == 3 else 1),
                           flat[-2:].reshape(-1, flat.shape[2] if flat.ndim == 3 else 1)])
    colour = [float(v) for v in np.median(edge, axis=0)]
    return cv2.copyMakeBorder(flat, p, p, p, p, cv2.BORDER_CONSTANT, value=colour)


# ------------------------------------------------------------------ four-point warp from the text itself
def _line(p, q):
    """Homogeneous line through two points."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def _fit(points):
    """Homogeneous least-squares line through points (cv2.fitLine)."""
    vx, vy, x0, y0 = cv2.fitLine(np.float32(points).reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return _line((x0, y0), (x0 + vx, y0 + vy))


def _meet(a, b):
    p = np.cross(a, b)
    return None if abs(p[2]) < 1e-9 else np.array([p[0] / p[2], p[1] / p[2]], np.float32)


def text_quad(quads: list, shape, margin: float = 0.6) -> Optional[np.ndarray]:
    """The four corners (tl, tr, br, bl) of the text block, estimated from its lines when no page edge is visible
    (the page fills the view or runs off it), as a scanner app does:
      top    = the top edge of the first line, bottom = the bottom edge of the last line
      left   = a line fitted through the left ends of all lines (left margins line up)
      right  = a line fitted through the right ends of the full-length lines (ragged short lines left out)
    Corners are pushed outwards by `margin` line heights. None when the lines do not give a sound quadrilateral.
    quads: line boxes as (tl, tr, br, bl), all from one text block, at least 3 lines."""
    qs = [np.asarray(q, np.float32) for q in quads]
    qs = [q for q in qs if np.linalg.norm(q[1] - q[0]) > 2.5 * np.linalg.norm(q[3] - q[0])]
    if len(qs) < 3:
        return None
    qs.sort(key=lambda q: float(q[:, 1].mean()))
    top, bot = qs[0], qs[-1]
    lh = float(np.median([np.linalg.norm(q[3] - q[0]) for q in qs]))
    lens = np.array([np.linalg.norm(q[1] - q[0]) for q in qs])
    full = [q for q, L in zip(qs, lens) if L >= 0.85 * lens.max()]
    T = _line(top[0], top[1])
    B = _line(bot[3], bot[2])
    L = _fit([p for q in qs for p in (q[0], q[3])])
    R = _fit([p for q in full for p in (q[1], q[2])]) if len(full) >= 2 else None
    if R is None:                                 # one full-length line: right edge parallel to the left one
        d = np.float32([L[1], -L[0]])
        p = full[0][1]
        R = _line(p, p + d)
    # move each side edge out so that every line end lies inside it (the longest line must not be cut off)
    def cover(edge, pts, inside_pt):
        """Shift the edge line outwards (parallel) until every point is on the inner side of it or on it."""
        a, b, c0 = edge
        n = np.hypot(a, b) + 1e-9
        s = np.sign(a * inside_pt[0] + b * inside_pt[1] + c0) or 1.0
        worst = min(s * (a * p[0] + b * p[1] + c0) / n for p in pts)   # signed distance, > 0 = inside
        if worst < 0:
            c0 = c0 - s * worst * n
        return np.array([a, b, c0])
    mid = np.mean([q.mean(0) for q in qs], axis=0)
    L = cover(L, [p for q in qs for p in (q[0], q[3])], mid)
    R = cover(R, [p for q in qs for p in (q[1], q[2])], mid)
    T = cover(T, [p for q in qs for p in (q[0], q[1])], mid)
    B = cover(B, [p for q in qs for p in (q[2], q[3])], mid)
    c = [_meet(T, L), _meet(T, R), _meet(B, R), _meet(B, L)]
    if any(x is None for x in c):
        return None
    c = np.array(c, np.float32)
    # push out: along each edge's outward normal by `margin` line heights
    ctr = c.mean(0)
    c = ctr + (c - ctr) * (1 + margin * lh / max(1.0, float(np.linalg.norm(c - ctr, axis=1).mean())))
    if not cv2.isContourConvex(c.reshape(-1, 1, 2).astype(np.int32)):
        return None
    for i in range(4):                            # corners must look like a page seen at an angle, not a sliver
        v1, v2 = c[i - 1] - c[i], c[(i + 1) % 4] - c[i]
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6), -1, 1)))
        if not 55 <= ang <= 125:
            return None
    H, W = shape[:2]
    if cv2.contourArea(c) < 0.02 * H * W:
        return None
    return c


def perspective_amount(quad: np.ndarray) -> float:
    """How far the quadrilateral is from an upright rectangle, in degrees: the larger of the tilt of its top edge and
    the angle between opposite edges (keystone). 0 = already flat and level."""
    tl, tr, br, bl = quad
    ang = lambda a, b: np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))   # noqa: E731
    top, bot = ang(tl, tr), ang(bl, br)
    left, right = ang(tl, bl), ang(tr, br)
    return float(max(abs(top), abs(top - bot), abs(left - right)))


def warp_quad(img: np.ndarray, quad: np.ndarray, max_side: int = 1800) -> tuple:
    """Homography from the quadrilateral to an upright rectangle (no extra margin: text_quad already has one).
    Returns (warped picture, 3x3 matrix)."""
    tl, tr, br, bl = quad.astype(np.float32)
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    k = min(1.0, max_side / max(w, h, 1))
    w, h = max(16, int(w * k)), max(16, int(h * k))
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
    # outside the camera picture: plain paper colour (the median of the picture), not stretched edge pixels, which
    # drew streaks that look like strokes in front of lines that run off the view
    paper = [float(v) for v in np.median(img.reshape(-1, img.shape[2] if img.ndim == 3 else 1), axis=0)]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=paper), M


# ------------------------------------------------------------------ colour scan (for people, not for the reader)
def color_scan(bgr: np.ndarray, margin: int = 0) -> np.ndarray:
    """A scanner-app "magic colour" picture for display and saving, in this order:
      1. camera grain removed (light non-local-means colour denoising, which keeps letter edges; four strengths
         were compared on real webcam captures of 25 Nov 2026, stronger ones smudged the letters)
      2. each colour channel divided by its own smooth estimate of the paper: shadows, uneven light AND the camera's
         colour cast go (paper becomes neutral white); ink keeps its colour
      3. ink darkened, colour lifted only slightly (a strong lift turned leftover grain into pink speckles)
      4. a small scan is enlarged 1.5-2x and gently sharpened, so letters look crisp on a screen
    The recogniser does not use this (it reads the grey picture); it is made only when a scan is shown or saved."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if margin and min(bgr.shape[:2]) > 4 * margin:                  # the plain margin flatten() added: not page
        bgr = bgr[margin:-margin, margin:-margin]
    bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 4, 10, 7, 21)    # light: stronger smudged the letters
    h, w = bgr.shape[:2]
    # paper brightness per channel: on a quarter-size copy a max filter wipes out the (darker) letters and a wide
    # median smooths what is left; broad shadows, light gradients and the colour cast stay in it and are divided out
    s = 0.25
    out = np.empty((h, w, 3), np.float32)
    for ch in range(3):
        small = cv2.resize(bgr[..., ch], (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        k = max(3, (min(small.shape) // 25) | 1)
        bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        bg = cv2.medianBlur(bg, max(3, (min(small.shape) // 6) | 1))
        bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 2), (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        out[..., ch] = bgr[..., ch].astype(np.float32) / np.maximum(bg, 1) * 250
    out = 255.0 * np.clip(out / 255.0, 0, 1) ** 1.5                     # darker ink, paper stays near white
    out = np.clip(out, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if max(h, w) < 1100:                                                # small scan: enlarge for the screen
        f = min(2.0, 1100.0 / max(h, w))
        out = cv2.resize(out, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(out, (0, 0), 1.0)
    return cv2.addWeighted(out, 2.0, blur, -1.0, 0)


# ------------------------------------------------------------------ enhancement ("scan" filters)
def enhance(grey: np.ndarray, mode: str) -> np.ndarray:
    """Scanner-app style clean-up of a grey picture before its lines are read. Modes (combine with '+'):
      even     shadows / uneven light removed (paper made uniformly white), contrast stretched
      clahe    local contrast (CLAHE, clip 2, 8x8 tiles)
      sharp    unsharp mask (sigma 1.0, amount 0.7): crisper letter edges after camera blur
      denoise  edge-preserving smoothing (bilateral 5 px) against sensor noise and JPEG blocks
      none     unchanged
    Applied in the order written, e.g. "denoise+even+sharp"."""
    out = grey
    for m in [x for x in (mode or "none").split("+") if x and x != "none"]:
        if m == "even":
            out = even_light(out, 1.0)
        elif m == "evensoft":
            out = even_light(out, 0.6)
        elif m == "clahe":
            out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)
        elif m == "sharp":
            blur = cv2.GaussianBlur(out, (0, 0), 1.0)
            out = cv2.addWeighted(out, 1.7, blur, -0.7, 0)
        elif m == "denoise":
            out = cv2.bilateralFilter(out, 5, 30, 5)
    return out


# ------------------------------------------------------------------ light
def even_light(grey: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Remove shadows / uneven light: divide by a smooth estimate of the paper, then stretch the contrast.
    Works for dark text on light paper; light-on-dark parts are left nearly as they are."""
    g = grey.astype(np.float32)
    k = max(15, (min(grey.shape[:2]) // 12) | 1)
    bg = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3).astype(np.float32)
    norm = np.clip(g / np.maximum(bg, 1) * 235, 0, 255)
    out = g * (1 - strength) + norm * strength
    lo, hi = np.percentile(out, (1, 99.5))
    if hi - lo > 20:
        out = (out - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ lines
def crop_line(img: np.ndarray, quad: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Cut one text line (quad tl, tr, br, bl) out along its own tilt. Lines standing on their side (taller than wide
    with more than one letter's height) are turned to read left to right."""
    q = quad.astype(np.float32)
    w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    w, h = max(w, 4), max(h, 4)
    if pad:
        c = q.mean(0)
        q = c + (q - c) * (1 + pad)
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    crop = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if h > 1.6 * w:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def line_angle(q: np.ndarray) -> float:
    d = (q[1] - q[0] + q[2] - q[3]) / 2
    return math.degrees(math.atan2(float(d[1]), float(d[0])))


def deskew_angle(quads: list) -> float:
    """Median tilt (degrees) of the wider-than-tall lines, weighted by length; 0 when there are none."""
    a, wts = [], []
    for q in quads:
        w = np.linalg.norm(q[1] - q[0])
        h = np.linalg.norm(q[3] - q[0])
        if w > 2 * h:
            a.append(line_angle(q))
            wts.append(w)
    if not a:
        return 0.0
    o = np.argsort(a)
    cw = np.cumsum(np.array(wts)[o])
    return float(np.array(a)[o][np.searchsorted(cw, cw[-1] / 2)])


def reading_order(items: list) -> list:
    """items: [(quad, ...)] -> same items in reading order: lines top to bottom, left to right within a line.
    Two boxes are on one line when their vertical centres are within half a line height."""
    if not items:
        return []
    def geo(it):
        q = it[0]
        ys = q[:, 1]
        return float(ys.mean()), float(max(4.0, ys.max() - ys.min())), float(q[:, 0].min())
    rows = []
    for it in sorted(items, key=lambda it: geo(it)[0]):
        yc, hh, x = geo(it)
        if rows and abs(yc - rows[-1]["yc"]) < 0.5 * max(hh, rows[-1]["h"]):
            r = rows[-1]
            r["items"].append(it)
            n = len(r["items"])
            r["yc"] = (r["yc"] * (n - 1) + yc) / n
            r["h"] = max(r["h"], hh)
        else:
            rows.append({"yc": yc, "h": hh, "items": [it]})
    out = []
    for r in rows:
        out.append(sorted(r["items"], key=lambda it: geo(it)[2]))
    return out                                                   # list of rows, each a list of items


# ------------------------------------------------------------------ auto capture
def sharpness(grey_small: np.ndarray) -> float:
    return float(cv2.Laplacian(grey_small, cv2.CV_32F).var())


class AutoCapture:
    """Feed every preview frame; ask a text check now and then; fire when the view is steady, sharp and has text.

        ac = AutoCapture(text_check)          # text_check(frame) -> list of line quads (fast, low resolution)
        for frame in camera:
            ev = ac.feed(frame)               # None, or {"fire": best_frame} or {"hint": "closer" | "steady" | ...}

    Rules (each tunable in config):
      steady   mean absolute change of a 160-px grey thumbnail below steady_diff for steady_s seconds
      sharp    sharpness at least sharp_ratio of the best seen in the last 3 s, and above min_sharp
      text     the text check finds at least one line whose height is >= min_line_frac of the picture height
      re-arm   after firing, the view must change (diff > move_diff) before it can fire again, so one page is
               read once; a button press can always force a read
    """

    def __init__(self, text_check, steady_s=0.5, steady_diff=4.0, move_diff=10.0, sharp_ratio=0.7, min_sharp=25.0,
                 check_every_s=0.35, min_line_frac=0.022, hint_every_s=4.0, buffer=6):
        self.text_check = text_check
        self.steady_s, self.steady_diff, self.move_diff = steady_s, steady_diff, move_diff
        self.sharp_ratio, self.min_sharp = sharp_ratio, min_sharp
        self.check_every_s, self.min_line_frac, self.hint_every_s = check_every_s, min_line_frac, hint_every_s
        self.buf = deque(maxlen=buffer)
        self.sharp_hist = deque(maxlen=60)
        self.prev = None
        self.steady_since = None
        self.armed = True
        self.last_check, self.last_hint, self.text_seen = 0.0, 0.0, None
        self.state = "looking"

    def reset(self, armed: bool = True):
        self.buf.clear()
        self.steady_since, self.armed, self.text_seen = None, armed, None

    def feed(self, frame: np.ndarray) -> Optional[dict]:
        now = time.time()
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        thumb = cv2.resize(g, (160, max(1, int(g.shape[0] * 160 / g.shape[1]))), interpolation=cv2.INTER_AREA)
        mid = cv2.resize(g, (480, max(1, int(g.shape[0] * 480 / g.shape[1]))), interpolation=cv2.INTER_AREA)
        sh = sharpness(mid)
        diff = 255.0 if self.prev is None else float(np.mean(cv2.absdiff(thumb, self.prev)))
        self.prev = thumb
        self.sharp_hist.append((now, sh))
        if diff > self.move_diff:
            self.armed = True                                    # the view changed: a new page may be read
        if diff < self.steady_diff:
            self.steady_since = self.steady_since or now
            self.buf.append((sh, frame))
        else:
            self.steady_since, self.text_seen = None, None
            self.buf.clear()
        if not self.armed:
            self.state = "done"
            return None
        if now - self.last_check >= self.check_every_s:
            self.last_check = now
            quads = self.text_check(frame)
            H = g.shape[0]
            tall = [q for q in quads if (q[:, 1].max() - q[:, 1].min()) >= self.min_line_frac * H]
            self.text_seen = {"n": len(quads), "readable": len(tall), "quads": quads}
        if not self.text_seen or not self.text_seen["n"]:
            self.state = "looking"
            return None
        if not self.text_seen["readable"]:
            self.state = "small"
            return self._hint(now, "closer")
        if self.steady_since is None or now - self.steady_since < self.steady_s:
            self.state = "steadying"
            return None
        best_recent = max(s for t, s in self.sharp_hist if now - t < 3.0)
        cand = max(self.buf, key=lambda b: b[0]) if self.buf else (sh, frame)
        if cand[0] < max(self.min_sharp, self.sharp_ratio * best_recent):
            self.state = "blurry"
            return self._hint(now, "steady")
        edge = self._cut_edge(self.text_seen["quads"], g.shape)
        self.armed = False
        self.state = "fired"
        return {"fire": cand[1], "sharp": cand[0], "edge": edge}

    def _hint(self, now, kind):
        if now - self.last_hint < self.hint_every_s:
            return None
        self.last_hint = now
        return {"hint": kind}

    @staticmethod
    def _cut_edge(quads, shape) -> str:
        """Which picture edge the text runs into ("left", "right", "top", "bottom" or ""), for a spoken hint."""
        H, W = shape[:2]
        m = 0.01
        hits = {"left": 0, "right": 0, "top": 0, "bottom": 0}
        for q in quads:
            if q[:, 0].min() <= m * W:
                hits["left"] += 1
            if q[:, 0].max() >= (1 - m) * W:
                hits["right"] += 1
            if q[:, 1].min() <= m * H:
                hits["top"] += 1
            if q[:, 1].max() >= (1 - m) * H:
                hits["bottom"] += 1
        k = max(hits, key=hits.get)
        return k if hits[k] >= max(2, 0.3 * len(quads)) else ""
