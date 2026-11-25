"""Tests for doc_scanner.py (OCR v5 scanner front end). No camera, no models: synthetic pictures only.

    python scripts/test_doc_scanner.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import doc_scanner as ds  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def page_scene(tilt=True):
    """A white page with dark lines of 'text' on a darker table, seen at an angle; returns picture and true corners."""
    page = np.full((400, 300, 3), 235, np.uint8)
    for i, y in enumerate(range(40, 380, 30)):
        cv2.rectangle(page, (25, y), (25 + 230 - (i % 3) * 40, y + 12), (30, 30, 30), -1)
    img = np.full((720, 1280, 3), (60, 80, 100), np.uint8)
    img = cv2.add(img, np.random.default_rng(0).integers(0, 25, img.shape, dtype=np.uint8))
    src = np.float32([[0, 0], [300, 0], [300, 400], [0, 400]])
    dst = np.float32([[420, 120], [830, 160], [870, 640], [380, 610]]) if tilt else np.float32([[440, 100], [840, 100], [840, 633], [440, 633]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(page, M, (1280, 720))
    mask = cv2.warpPerspective(np.full((400, 300), 255, np.uint8), M, (1280, 720))
    img[mask > 0] = warped[mask > 0]
    return img, dst


def test_find_and_flatten():
    img, true = page_scene()
    q = ds.find_page(img)
    check("find_page finds a tilted page", q is not None)
    if q is None:
        return
    err = float(np.abs(q - true).max())
    check("page corners within 12 px", err < 12, f"max error {err:.1f} px")
    flat = ds.flatten(img, q)
    h, w = flat.shape[:2]
    check("flattened page is upright (taller than wide, like the page)", h > w, f"{w}x{h}")
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    rows = (g < 100).mean(1)
    lines = int(((rows[1:] > 0.3) & (rows[:-1] <= 0.3)).sum())
    check("the 12 text lines run level after flattening", 10 <= lines <= 13, f"{lines} line starts")


def test_no_page():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8), (0, 0), 6)
    check("no page in a texture picture", ds.find_page(img) is None)


def test_even_light():
    g = np.full((300, 600), 230, np.float32)
    g *= np.linspace(0.35, 1.0, 600)[None, :]                     # strong shadow on the left
    for y in range(40, 280, 40):
        g[y:y + 10, 40:560] *= 0.25
    g = g.astype(np.uint8)
    e = ds.even_light(g)
    left_paper, right_paper = e[20:30, 50:100].mean(), e[20:30, 500:550].mean()
    check("shadow evened out (paper left vs right within 25 levels)", abs(left_paper - right_paper) < 25,
          f"{left_paper:.0f} vs {right_paper:.0f}; before {g[20:30, 50:100].mean():.0f} vs {g[20:30, 500:550].mean():.0f}")
    check("ink still dark after evening", e[45, 300] < 90, f"{e[45, 300]}")


def test_crop_and_order():
    img = np.full((200, 400), 255, np.uint8)
    q = np.float32([[50, 60], [350, 80], [348, 110], [48, 90]])  # a line tilted ~4 degrees
    crop = ds.crop_line(img, q)
    check("crop_line output size follows the box", abs(crop.shape[1] - 301) < 4 and abs(crop.shape[0] - 30) < 4,
          f"{crop.shape[1]}x{crop.shape[0]}")
    a = (np.float32([[10, 10], [100, 10], [100, 30], [10, 30]]), "A")
    b = (np.float32([[120, 12], [200, 12], [200, 32], [120, 32]]), "B")
    c = (np.float32([[10, 50], [200, 50], [200, 70], [10, 70]]), "C")
    rows = ds.reading_order([c, b, a])
    order = [[it[1] for it in r] for r in rows]
    check("reading order: rows top to bottom, left to right", order == [["A", "B"], ["C"]], str(order))
    ang = ds.deskew_angle([np.float32([[0, 0], [200, 17.5], [200, 37.5], [0, 20]])])
    check("deskew angle of a 5-degree line", abs(ang - 5) < 0.5, f"{ang:.2f}")


def test_text_quad():
    """Lines of a page seen at an angle with no page edge in view: corners from the lines, warp makes them level."""
    src = np.float32([[0, 0], [1000, 0], [1000, 700], [0, 700]])
    dst = np.float32([[80, 60], [1180, 130], [1150, 690], [110, 620]])          # keystone + tilt
    M = cv2.getPerspectiveTransform(src, dst)
    lines = []
    for i, y in enumerate(range(100, 650, 90)):
        x1 = 900 if i % 2 == 0 else 700                                      # ragged right margin
        r = np.float32([[60, y], [x1, y], [x1, y + 40], [60, y + 40]])
        lines.append(cv2.perspectiveTransform(r[None], M)[0])
    q = ds.text_quad(lines, (720, 1280))
    check("text_quad finds four corners from the lines alone", q is not None)
    if q is None:
        return
    check("perspective amount is clearly above zero", ds.perspective_amount(q) > 3, f"{ds.perspective_amount(q):.1f} deg")
    img = np.full((720, 1280), 255, np.uint8)
    for l in lines:
        cv2.fillPoly(img, [l.astype(np.int32)], 0)
    flat, H = ds.warp_quad(img, q)
    back = [cv2.perspectiveTransform(l[None], H)[0] for l in lines]
    tilts = [abs(np.degrees(np.arctan2(b[1][1] - b[0][1], b[1][0] - b[0][0]))) for b in back]
    check("after the warp every line is level (< 1 degree)", max(tilts) < 1.0, f"max {max(tilts):.2f} deg")
    inside = all(0 <= p[0] < flat.shape[1] and 0 <= p[1] < flat.shape[0] for b in back for p in b)
    check("no line end is cut off by the warp", inside)


def test_auto_capture():
    big_line = [np.float32([[300, 300], [900, 300], [900, 340], [300, 340]])]
    tiny_line = [np.float32([[600, 350], [660, 350], [660, 356], [600, 356]])]
    state = {"lines": big_line}
    ac = ds.AutoCapture(lambda f: state["lines"], steady_s=0.3, check_every_s=0.0, hint_every_s=0.0)
    rng = np.random.default_rng(2)
    base = np.full((720, 1280, 3), 128, np.uint8)               # a scene with large shapes and fine detail
    for _ in range(60):
        x, y = int(rng.integers(0, 1200)), int(rng.integers(0, 650))
        cv2.rectangle(base, (x, y), (x + int(rng.integers(30, 200)), y + int(rng.integers(20, 120))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    base = cv2.add(base, rng.integers(0, 40, base.shape, dtype=np.uint8))
    # moving camera: never fires
    fired = False
    for i in range(10):
        ev = ac.feed(np.roll(base, i * 40, axis=1))
        fired |= bool(ev and "fire" in ev)
    check("does not fire while the view moves", not fired)
    # steady and sharp: fires within ~0.5 s
    t0, ev = time.time(), None
    while time.time() - t0 < 2.0:
        ev = ac.feed(base)
        if ev and "fire" in ev:
            break
        time.sleep(0.03)
    check("fires once the view is steady with text", bool(ev and "fire" in ev), f"after {time.time() - t0:.2f} s")
    # same page again: does not fire twice
    again = any((ac.feed(base) or {}).get("fire") is not None for _ in range(20))
    check("does not read the same page twice", not again)
    # new view (move, then steady) with tiny text: asks to come closer
    state["lines"] = tiny_line
    for i in range(5):
        ac.feed(np.roll(base, 200 + i * 40, axis=0))
    hint = None
    for _ in range(20):
        ev = ac.feed(np.roll(base, 360, axis=0))
        if ev and "hint" in ev:
            hint = ev["hint"]
            break
        time.sleep(0.03)
    check("tiny text -> 'closer' hint", hint == "closer", str(hint))
    # blurry steady view: 'steady' hint, no fire
    ac2 = ds.AutoCapture(lambda f: big_line, steady_s=0.1, check_every_s=0.0, hint_every_s=0.0)
    for _ in range(5):
        ac2.feed(base)                                         # sharp frames set the reference sharpness
    for i in range(5):
        ac2.feed(np.roll(base, 300 + 40 * i, axis=1))          # move away
    blur = cv2.GaussianBlur(np.roll(base, 520, axis=1), (0, 0), 6)
    out = []
    t0 = time.time()
    while time.time() - t0 < 1.0:
        out.append(ac2.feed(blur))
        time.sleep(0.03)
    check("blurry view -> 'steady' hint and no read",
          any(e and e.get("hint") == "steady" for e in out) and not any(e and "fire" in e for e in out))


if __name__ == "__main__":
    test_find_and_flatten()
    test_no_page()
    test_even_light()
    test_crop_and_order()
    test_text_quad()
    test_auto_capture()
    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
