"""
OCR Mode — offline Bangla + English reading (EasyOCR, no internet).

Triggered only on ACTION button press (not continuous) to avoid
blocking the CPU. EasyOCR is lazy-loaded on first activation so the
application starts quickly and ~600 MB of model weight doesn't sit
in RAM while another mode is active.

After recognition, text is NFC-normalized and matched against
assets/ocr_lexicon.txt so typical Bangla sign errors get repaired
before TTS. Optional Tesseract (ben+eng) is used only if EasyOCR
returns nothing.
"""
import logging
import os
import re
import time
from typing import Optional

import cv2
import numpy as np

import config
from ocr_repair import repair_ocr_text, tesseract_available, tesseract_read
from ocr_text_region import _boxes as _text_boxes
from ocr_text_region import clean_text, dominant_block, order_lines
from utils import detect_language
from .base_mode import BaseMode

logger = logging.getLogger("smart_glass.ocr_mode")

# A detection only "looks like text" if at least half of its non-space
# characters are letters or digits (Bangla, Latin, or Bangla numerals —
# prices, phone numbers, room numbers etc. are meaningful to read aloud).
# EasyOCR frequently emits short symbol/noise fragments (".. | --", "I I I",
# stray punctuation from edges and textures) that pass the confidence +
# length filters but are gibberish when read aloud — this catches those
# before they reach the TTS queue.
_CONTENT_RE = re.compile(r"[^\W_]", re.UNICODE)
_MIN_CONTENT_RATIO = 0.5


def _looks_like_text(text: str) -> bool:
    stripped = text.replace(" ", "")
    if not stripped:
        return False
    content_chars = len(_CONTENT_RE.findall(stripped))
    return (content_chars / len(stripped)) >= _MIN_CONTENT_RATIO


def _reading_order_key(item):
    """Sort EasyOCR detections top-to-bottom, then left-to-right.

    EasyOCR returns detections in whatever order its detector finds them,
    which often does NOT match the natural reading order of the page —
    joining them as-is interleaves unrelated lines into nonsense sentences.
    Using the bounding box's top-left corner restores natural reading order.
    """
    if item and len(item) >= 1 and item[0]:
        bbox = item[0]
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        return (min(ys), min(xs))
    return (0, 0)


class OCRMode(BaseMode):

    def __init__(self) -> None:
        self._reader = None   # lazy-loaded EasyOCR (printed)
        self._ekush = None    # lazy-loaded Ekush CNN (handwritten letters)

        # Capture and playback are two separate steps: the CAPTURE button
        # snaps a frame, runs OCR, and stores the result here; the READ
        # button speaks whatever is currently stored. This lets the user
        # hold the camera steady only for the (quick) capture, then move
        # it away before listening to a (possibly long) read-out.
        self._stored_text: Optional[str] = None
        self._stored_lang: str = "en"
        # set by main.py to the speech function: pages are spoken line by line while the rest is still being read
        self.speak_line = None
        self.streamed = False   # the last capture was already spoken line by line
        self._v5 = None         # OCR v5 engine (ocr_v5.OcrV5) when config.OCR_PIPELINE == "v5"
        self._auto = None       # auto capture (doc_scanner.AutoCapture), v5 only
        self.last_v5 = None     # the last v5 reading (text, lines, boxes, timings) for logs and the preview

    def _ensure_ekush(self):
        if self._ekush is None:
            try:
                from ekush_letters import EkushRecognizer
                self._ekush = EkushRecognizer()
            except Exception as exc:
                logger.warning("Ekush handwritten CNN not loaded: %s", exc)
                self._ekush = False
        return self._ekush if self._ekush not in (None, False) else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_v5(self) -> bool:
        if getattr(config, "OCR_PIPELINE", "v2") != "v5":
            return False
        if self._v5 is None:
            try:
                from ocr_v5 import OcrV5
                self._v5 = OcrV5()
                self._v5.read(np.full((96, 320, 3), 255, np.uint8))           # warm up both networks
                logger.info("OCR v5 ready (text finder: %s)", self._v5.finder.backend)
            except Exception as exc:  # noqa: BLE001
                logger.error("OCR v5 not available (%s); using the EasyOCR reader", exc)
                self._v5 = False
        return bool(self._v5)

    def activate(self) -> None:
        logger.info("OCR mode activated")
        # Start each OCR session with a clean slate
        self._stored_text = None
        self._stored_lang = "en"
        if self._auto is not None:
            self._auto.reset()
        if self._ensure_v5():
            return
        if self._reader is None:
            logger.info("Loading EasyOCR model (Bangla + English) — first use…")
            try:
                import easyocr
                # gpu=False is correct for RPi 5 (no CUDA GPU)
                use_gpu = False
                try:
                    import torch
                    use_gpu = bool(torch.cuda.is_available()) and os.environ.get("OCR_GPU", "1") == "1"
                except Exception:  # noqa: BLE001
                    pass
                # GPU when there is one (a laptop): about ten times faster. The Pi has none and runs on CPU as before.
                self._reader = easyocr.Reader(["bn", "en"], gpu=use_gpu, verbose=False)
                logger.info("EasyOCR on %s", "GPU" if use_gpu else "CPU")
                if not use_gpu:
                    from ocr_detector_onnx import install
                    install(self._reader)      # ONNX Runtime text detector when its file is present (config.OCR_DETECTOR)
                    if config.OCR_DETECTOR == "hailo":
                        import hailo_text_detector
                        hailo_text_detector.install(self._reader)   # text found on the AI HAT; CPU detector kept behind it
                import ocr_recognizer_ft
                ocr_recognizer_ft.install(self._reader)   # fine-tuned letter recogniser when config names one
                logger.info("EasyOCR ready")
            except Exception as exc:
                logger.error("Failed to load EasyOCR: %s", exc)
                self._reader = None
        self._ensure_ekush()

    def deactivate(self) -> None:
        logger.info("OCR mode deactivated")

    def cleanup(self) -> None:
        if self._v5:
            self._v5.close()
        self._v5 = None
        det = getattr(self._reader, "_hailo_det", None) if self._reader is not None else None
        if det is not None:
            det.close()
        self._reader = None
        self._ekush = None

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        CAPTURE step (triggered by the capture/action button): snap the
        frame, run OCR, store the cleaned-up result for later playback,
        and return only a short confirmation to speak — NOT the full text.
        Call read_stored() (triggered by the READ button) to hear it.
        """
        if frame is None:
            return "ক্যামেরা প্রস্তুত নয়"  # camera not ready

        if self._v5:
            return self._process_v5(frame)

        if self._reader is None and self._ensure_ekush() is None:
            return "ইঞ্জিন লোড হয়নি, অনুগ্রহ করে অপেক্ষা করুন"  # engine not loaded

        texts = []
        self.streamed = False
        self.last_boxes, self.last_boxes_t = [], time.time()
        v2 = getattr(config, "OCR_PIPELINE", "legacy") == "v2"
        if self._reader is not None and v2:
            texts = self._read_v2(frame)
        if self._reader is not None and not v2:
            preprocessed = self._preprocess(frame)
            decoder = getattr(config, "OCR_DECODER", "beamsearch")
            decode_kwargs = {
                "decoder": decoder,
                "canvas_size": getattr(config, "OCR_CANVAS_SIZE", 2560),
                "slope_ths": float(getattr(config, "OCR_SLOPE_THS", 0.1)),
            }
            if decoder == "beamsearch":
                decode_kwargs["beamWidth"] = getattr(config, "OCR_BEAM_WIDTH", 5)
            try:
                results = self._reader.readtext(
                    preprocessed,
                    detail=1,
                    paragraph=False,   # paragraph=True changes tuple format; keep False for stability
                    width_ths=0.7,
                    height_ths=0.7,
                    **decode_kwargs,
                )
            except TypeError:
                results = self._reader.readtext(
                    preprocessed,
                    detail=1,
                    paragraph=False,
                    width_ths=0.7,
                    height_ths=0.7,
                )
            except Exception as exc:
                logger.warning("EasyOCR inference error: %s", exc)
                results = []

            results = sorted(results, key=_reading_order_key)
            s = frame.shape[1] / float(preprocessed.shape[1])
            for b in _text_boxes(results):   # every box, for the preview; the kept ones are marked below
                self.last_boxes.append([int(b["x0"] * s), int(b["y0"] * s), int(b["x1"] * s), int(b["y1"] * s), b["c"], False])
            for item in results:
                if len(item) == 3:
                    _bbox, text, conf = item
                elif len(item) == 2:
                    text, conf = item
                else:
                    continue
                text = text.strip()
                if conf < config.OCR_CONFIDENCE or len(text) < config.OCR_MIN_CHARS:
                    continue
                if not _looks_like_text(text):
                    logger.debug("Discarding non-text OCR fragment (conf=%.2f): %r", conf, text)
                    continue
                texts.append(text)
                if len(item) == 3:
                    ys = [p[1] * s for p in item[0]]
                    for box in self.last_boxes:
                        if abs(box[1] - min(ys)) < 1 and abs(box[3] - max(ys)) < 1:
                            box[5] = True

            if not texts and tesseract_available():
                tess = tesseract_read(preprocessed)
                if tess:
                    texts = [tess]
                    logger.info("EasyOCR empty; using offline Tesseract")

        # The handwritten-letter reader is tried only when the text detector saw something it could not read. On a
        # picture with no text at all it used to "read" letters that are not there (on the Pi, a blank frame gave 48
        # made-up characters after 27 s).
        if not texts and not (v2 and getattr(self, "no_text_boxes", False)):
            ekush = self._ensure_ekush()
            if ekush is not None and ekush.ok:
                hw = ekush.read_frame(frame)
                if hw:
                    texts = [hw]
                    logger.info("Using Ekush handwritten letters: %s", hw[:80])

        if not texts:
            self._stored_text = None
            self._stored_lang = "en"
            return "কোনো লেখা পাওয়া যায়নি"  # no text found

        combined = clean_text(" ".join(texts)) if v2 else repair_ocr_text(" ".join(texts))
        lang = detect_language(combined)
        logger.info("OCR result (lang=%s, %d chars): %s", lang, len(combined), combined[:80])

        # Store for the READ button — capture and playback are deliberately
        # separate steps (see __init__ note).
        self._stored_text = combined
        self._stored_lang = lang

        if self.streamed:
            # every line has already been spoken while it was read; READ repeats the whole page
            return "পড়া শেষ। আবার শুনতে রিড বাটন চাপুন" if lang == "bn" else "Done. Press READ to hear it again."
        if getattr(config, "OCR_SPEAK_ON_CAPTURE", True):
            # say what was read straight away, so the user hears the detection itself; READ repeats it
            return ("লেখা আছে: " if lang == "bn" else "The text says: ") + combined
        if lang == "bn":
            return "লেখা সংরক্ষণ করা হয়েছে। শুনতে রিড বাটন চাপুন"   # "Text saved. Press READ to listen"
        return "Text captured. Press the READ button to listen."

    # ------------------------------------------------------------------
    # OCR v5: scanner-style reading and auto capture
    # ------------------------------------------------------------------

    def _process_v5(self, frame: np.ndarray) -> Optional[str]:
        self.streamed = False
        stream = self.speak_line is not None and getattr(config, "OCR_STREAM_PAGES", False)
        spoken = []

        def on_line(t):
            spoken.append(t)
            if len(spoken) == 1:
                t = ("লেখা আছে: " if detect_language(t) == "bn" else "The text says: ") + t
            self.speak_line(t)

        # pages are spoken line by line while the rest is read; short signs are said in one go below
        res = self._v5.read(frame, on_line=on_line if stream else None,
                            stream_min_rows=int(getattr(config, "OCR_STREAM_MIN_LINES", 4)))
        self.last_v5 = res
        self.last_boxes, self.last_boxes_t = [], time.time()
        if res.get("page") is None and not res.get("rotated"):     # boxes are in camera pixels: show them
            for q, txt, c in res["boxes"]:
                self.last_boxes.append([int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()),
                                        int(q[:, 1].max()), float(c), c >= config.OCR5_MIN_CONF])
        text = res["text"]
        if os.environ.get("OCR_DEBUG_DIR"):
            self._save_debug_v5(frame, res)
        if not text:
            self._stored_text, self._stored_lang = None, "en"
            if res["timings"]["lines"] and getattr(config, "OCR_AUTO_HINTS", True):
                return "লেখা পরিষ্কার নয়, একটু কাছে এনে স্থির রাখুন"   # text not clear: bring it closer, hold still
            return "কোনো লেখা পাওয়া যায়নি"  # no text found
        lang = res["lang"]
        # lines of a page are joined with a space: a sentence wrapped over two lines must not get a full stop in the
        # middle; the speech engine pauses at the page's own punctuation
        self._stored_text, self._stored_lang = " ".join(res["lines"]), lang
        logger.info("OCR v5 result (lang=%s, %d chars): %s", lang, len(text), " | ".join(res["lines"])[:160])
        if spoken:
            self.streamed = True
            return "পড়া শেষ। আবার শুনতে রিড বাটন চাপুন" if lang == "bn" else "Done. Press READ to hear it again."
        if getattr(config, "OCR_SPEAK_ON_CAPTURE", True):
            return ("লেখা আছে: " if lang == "bn" else "The text says: ") + self._stored_text
        return "লেখা সংরক্ষণ করা হয়েছে। শুনতে রিড বাটন চাপুন" if lang == "bn" else "Text captured. Press READ to listen."

    _HINTS = {
        "closer": "লেখাটি ক্যামেরার আরও কাছে আনুন",     # bring the text closer to the camera
        "steady": "স্থির রাখুন",                         # hold still
        "left": "লেখা বাঁ দিকে কেটে যাচ্ছে",             # the text is cut off on the left
        "right": "লেখা ডান দিকে কেটে যাচ্ছে",
        "top": "লেখা উপরে কেটে যাচ্ছে",
        "bottom": "লেখা নিচে কেটে যাচ্ছে",
    }

    def auto_feed(self, frame: np.ndarray) -> Optional[dict]:
        """Auto capture (config.OCR_AUTO_CAPTURE): call with live camera frames while text mode is on.
        Returns None, {"hint": text to speak} or {"fire": frame to read, "say": optional hint to speak first}."""
        if not self._v5 or not getattr(config, "OCR_AUTO_CAPTURE", False) or frame is None:
            return None
        if self._auto is None:
            import doc_scanner
            self._auto = doc_scanner.AutoCapture(self._v5.quick_lines,
                                                 steady_s=float(getattr(config, "OCR_AUTO_STEADY_S", 0.5)),
                                                 min_line_frac=float(getattr(config, "OCR_AUTO_MIN_LINE_FRAC", 0.028)))
        ev = self._auto.feed(frame)
        if not ev:
            return None
        if "hint" in ev:
            return {"hint": self._HINTS[ev["hint"]]} if getattr(config, "OCR_AUTO_HINTS", True) else None
        out = {"fire": ev["fire"]}
        if ev.get("edge") and getattr(config, "OCR_AUTO_HINTS", True):
            out["say"] = self._HINTS[ev["edge"]]
        return out

    def auto_state(self) -> str:
        return self._auto.state if self._auto is not None else ""

    def auto_hold(self) -> None:
        """Do not auto-read the page that is in view now (it was just read by a button press)."""
        if self._auto is not None:
            self._auto.reset(armed=False)

    def _save_debug_v5(self, frame, res) -> None:
        try:
            import json
            d = os.path.join(os.environ["OCR_DEBUG_DIR"], time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}")
            os.makedirs(d, exist_ok=True)
            cv2.imwrite(os.path.join(d, "original.jpg"), frame)
            vis = res["image"].copy()
            for q, txt, c in res["boxes"]:
                cv2.polylines(vis, [q.astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 200, 0) if c >= config.OCR5_MIN_CONF else (0, 0, 255), 2)
            cv2.imwrite(os.path.join(d, "boxes.jpg"), vis)
            if res.get("scan") is not None:
                cv2.imwrite(os.path.join(d, "scan.jpg"), res["scan"])          # flattened, grey: as the reader saw it
            import doc_scanner
            cv2.imwrite(os.path.join(d, "scan_color.jpg"), doc_scanner.color_scan(res["image"], res.get("page_margin", 0)))   # for people
            with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
                json.dump({"text": res["text"], "timings": res["timings"], "page": res["page"],
                           "lines": [{"box": q.round(1).tolist(), "text": t, "conf": round(float(c), 3)}
                                     for q, t, c in res["boxes"]]}, f, ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR debug save failed: %s", exc)

    def read_stored(self) -> Optional[str]:
        """
        READ step (triggered by the dedicated read/playback button):
        speak the most recently captured text, or a short notice if
        nothing has been captured yet in this session.
        """
        if not self._stored_text:
            return "কোনো লেখা সংরক্ষিত নেই। প্রথমে ক্যাপচার করুন"  # "Nothing saved yet. Capture first"

        if self._stored_lang == "bn":
            return "পড়া হচ্ছে: " + self._stored_text   # "Reading: ..."
        return "Reading: " + self._stored_text

    def _read_v2(self, frame: np.ndarray) -> list:
        """OCR pipeline v2: CLAHE, tuned EasyOCR, dominant text block, reading order (see config.OCR_PIPELINE)."""
        h, w = frame.shape[:2]
        w0 = w
        read_w = int(getattr(config, "OCR_READ_WIDTH", 1200))     # a camera picture is enlarged to this width before reading
        if w < read_w:
            frame = cv2.resize(frame, (read_w, int(h * read_w / w)), interpolation=cv2.INTER_CUBIC)
        elif w > 1600:
            frame = cv2.resize(frame, (1600, int(h * 1600 / w)), interpolation=cv2.INTER_AREA)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        decoder = getattr(config, "OCR_V2_DECODER", getattr(config, "OCR_DECODER", "beamsearch"))
        kw = dict(getattr(config, "OCR_V2_PARAMS", {}))
        kw.update(decoder=decoder, canvas_size=getattr(config, "OCR_CANVAS_SIZE", 2560))
        if decoder == "beamsearch":
            kw["beamWidth"] = getattr(config, "OCR_BEAM_WIDTH", 5)
        # The text detector runs ONCE per capture and its boxes are used by both paths below. Before 14 October the
        # line-by-line path ran it to count the lines and the normal path ran it again: on the Raspberry Pi 5 a
        # three-line sign took 34 s instead of 18 s (benchmark_report/REPORT.md, 13 October 2026).
        det_kw = {k: kw[k] for k in ("text_threshold", "low_text", "link_threshold", "canvas_size", "mag_ratio", "slope_ths")
                  if k in kw}
        # margin EasyOCR adds around each detected box before cutting the word out (its default is 0.1 of the
        # box height); a larger margin keeps the first and last letter when the detector box is tight
        det_kw["add_margin"] = float(getattr(config, "OCR_ADD_MARGIN", 0.1))
        kw["add_margin"] = det_kw["add_margin"]
        rec_kw = {k: kw[k] for k in ("decoder", "beamWidth", "contrast_ths", "adjust_contrast") if k in kw}
        self.no_text_boxes = False
        try:
            hl, fl = self._reader.detect(gray, width_ths=0.7, height_ths=0.7, **det_kw)
        except Exception as exc:
            logger.warning("EasyOCR detection error: %s", exc)
            return []
        boxes, free = (list(hl[0]) if hl else []), (list(fl[0]) if fl else [])
        if not boxes and not free:
            second = self._second_pass(gray, kw)
            if second:
                return [second]
            self.no_text_boxes = True          # nothing that looks like text: say so at once (process_frame)
            return []
        if self.speak_line is not None and getattr(config, "OCR_STREAM_PAGES", False):
            page = self._read_page_streaming(gray, kw, w0, boxes)
            if page is not None:
                return page
        try:
            det = self._reader.recognize(gray, horizontal_list=boxes, free_list=free, detail=1, paragraph=False, **rec_kw)
        except Exception as exc:
            logger.warning("EasyOCR inference error: %s", exc)
            return []
        block = dominant_block(det, getattr(config, "OCR_V2_MIN_CONF", 0.2))
        # boxes for the preview, in the camera picture's own pixels; the main text block is marked
        s, kept = w0 / float(gray.shape[1]), {(b["x0"], b["y0"]) for b in block}
        self.last_boxes = [[int(b["x0"] * s), int(b["y0"] * s), int(b["x1"] * s), int(b["y1"] * s), b["c"],
                            (b["x0"], b["y0"]) in kept] for b in _text_boxes(det)]
        text = order_lines(block)
        conf = float(np.mean([b["c"] for b in block])) if block else 0.0
        low = float(getattr(config, "OCR_ROTATE_LOWCONF", 0.0))
        if (not text and (getattr(config, "OCR_ROTATE_FALLBACK", False) or low > 0)) or (text and conf < low):
            # nothing read, or read with low confidence: the text may stand on its side (a vertical sign, a packet
            # held sideways). Try the picture turned by 90 degrees each way and keep the most confident reading,
            # the upright one included. Costs time only in those cases.
            best = (conf, text) if text else (0.0, "")
            for turn in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                try:
                    det2 = self._reader.readtext(cv2.rotate(gray, turn), detail=1, paragraph=False, width_ths=0.7,
                                                 height_ths=0.7, **kw)
                except Exception as exc:
                    logger.warning("EasyOCR inference error (rotated): %s", exc)
                    continue
                block2 = dominant_block(det2, getattr(config, "OCR_V2_MIN_CONF", 0.2))
                t2 = order_lines(block2)
                if t2:
                    c2 = float(np.mean([b["c"] for b in block2]))
                    if c2 > best[0]:
                        best = (c2, t2)
            text = best[1]
        if not text:
            text = self._second_pass(gray, kw)
        if os.environ.get("OCR_DEBUG_DIR"):
            self._save_debug(frame, gray, det, block, text)
        return [text] if text else []

    def _second_pass(self, gray, kw) -> str:
        """config.OCR_SECOND_PASS (None = off): when the first pass keeps no text, look again with other detector
        settings (for example the picture at half size, which finds large blurred letters) and keep the reading only
        if its mean confidence reaches OCR_SECOND_PASS_MIN_CONF, so a text-free scene stays "no text found".
        Costs time only when nothing was read. Chosen by ocr_bench/probe_second_pass.py and ocr_v4_eval.py."""
        p = getattr(config, "OCR_SECOND_PASS", None)
        if not p:
            return ""
        kw2 = dict(kw)
        kw2.update(p)
        try:
            det2 = self._reader.readtext(gray, detail=1, paragraph=False, width_ths=0.7, height_ths=0.7, **kw2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EasyOCR inference error (second pass): %s", exc)
            return ""
        block2 = dominant_block(det2, getattr(config, "OCR_V2_MIN_CONF", 0.2))
        if not block2:
            return ""
        c2 = float(np.mean([b["c"] for b in block2]))
        return order_lines(block2) if c2 >= float(getattr(config, "OCR_SECOND_PASS_MIN_CONF", 0.5)) else ""

    def _save_debug(self, frame, gray, det, block, text) -> None:
        """OCR_DEBUG_DIR=<folder>: keep what the reader saw for one capture (for testing, never on by default):
        original.jpg, boxes.jpg (green = main text block that is spoken, red = other boxes), crops/NN.png and
        result.json (every box with its text and confidence, and the final text). Never raises."""
        try:
            import json
            d = os.path.join(os.environ["OCR_DEBUG_DIR"], time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}")
            os.makedirs(os.path.join(d, "crops"), exist_ok=True)
            cv2.imwrite(os.path.join(d, "original.jpg"), frame)
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            kept = {(b["x0"], b["y0"]) for b in block}
            rows = []
            for i, b in enumerate(_text_boxes(det)):
                x0, y0, x1, y1 = (int(v) for v in (b["x0"], b["y0"], b["x1"], b["y1"]))
                main = (b["x0"], b["y0"]) in kept
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0) if main else (0, 0, 255), 2)
                crop = gray[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
                if crop.size:
                    cv2.imwrite(os.path.join(d, "crops", f"{i:02d}.png"), crop)
                rows.append({"i": i, "box": [x0, y0, x1, y1], "text": b["t"], "conf": round(float(b["c"]), 4), "main_block": main})
            cv2.imwrite(os.path.join(d, "boxes.jpg"), vis)
            with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
                json.dump({"text": text, "boxes": rows, "settings": {"add_margin": getattr(config, "OCR_ADD_MARGIN", 0.1),
                           "rotate_lowconf": getattr(config, "OCR_ROTATE_LOWCONF", 0.0)}}, f, ensure_ascii=False, indent=1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OCR debug save failed: %s", exc)

    def _read_page_streaming(self, gray: np.ndarray, kw: dict, w0: int, boxes: list):
        """Whole pages: read and speak the text line by line in reading order. boxes are the text detector's
        upright boxes [x_min, x_max, y_min, y_max] for this picture (found once, in _read_v2).
        Returns None when the picture has fewer than OCR_STREAM_MIN_LINES lines (the normal path then reads them)."""
        rec_kw = {k: kw[k] for k in ("decoder", "beamWidth", "contrast_ths", "adjust_contrast") if k in kw}
        if not boxes:
            return None
        # the same main-text-block rule as the normal path, on geometry only (no confidence before recognition):
        # without it, background text in a photographed scene counted as "lines" and was read (validation CER
        # 2.14 % -> 4.95 %, results/ocr_bench/page_streaming_control.json). Tilted boxes are not read on pages.
        quads = [([[b[0], b[2]], [b[1], b[2]], [b[1], b[3]], [b[0], b[3]]], i, 1.0) for i, b in enumerate(boxes)]
        boxes = [boxes[d["t"]] for d in dominant_block(quads, 0.0)]
        free = []
        # group the boxes into lines: a box joins the line whose centre is within half a text height
        lines = []
        for b in sorted(boxes, key=lambda b: (b[2] + b[3]) / 2):
            yc, h = (b[2] + b[3]) / 2, max(1, b[3] - b[2])
            if lines and abs(yc - lines[-1]["yc"]) < 0.5 * max(h, lines[-1]["h"]):
                lines[-1]["boxes"].append(b)
            else:
                lines.append({"yc": yc, "h": h, "boxes": [b]})
        if len(lines) < getattr(config, "OCR_STREAM_MIN_LINES", 4):
            return None
        self.streamed = True
        min_conf = getattr(config, "OCR_V2_MIN_CONF", 0.2)
        s = w0 / float(gray.shape[1])
        out, first = [], True
        for ln in lines + ([{"boxes": [], "free": free}] if free else []):
            try:
                res = self._reader.recognize(gray, horizontal_list=sorted(ln["boxes"], key=lambda b: b[0]),
                                             free_list=ln.get("free", []), detail=1, paragraph=False, **rec_kw)
            except Exception as exc:
                logger.warning("EasyOCR recognition error: %s", exc)
                continue
            words = []
            for item in res:
                if len(item) < 3 or float(item[2]) < min_conf:
                    continue
                words.append(item[1])
                xs, ys = [p[0] for p in item[0]], [p[1] for p in item[0]]
                self.last_boxes.append([int(min(xs) * s), int(min(ys) * s), int(max(xs) * s), int(max(ys) * s),
                                        float(item[2]), True])
            text = clean_text(" ".join(words))
            if not text:
                continue
            out.append(text)
            if first:
                lead = "লেখা আছে: " if detect_language(text) == "bn" else "The text says: "
                self.speak_line(lead + text)
                first = False
            else:
                self.speak_line(text)
        return out

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale → denoise → CLAHE.
        2x upscaling is the single biggest accuracy boost for Bangla script.

        NOTE: deliberately stops at grayscale and does NOT binarize
        (no adaptiveThreshold). EasyOCR's recognizer is trained on
        natural grayscale/color images with anti-aliased glyph edges;
        measured side-by-side on the same frames, binarizing *lowered*
        average confidence (~0.69 vs ~0.84 on Latin text) and pushed
        some valid detections below OCR_CONFIDENCE, causing text to be
        dropped entirely — the opposite of the intended effect.
        """
        h, w = frame.shape[:2]

        # 1. Upscale small/medium frames to ~1200px wide — critical for Bangla.
        # (Tested 1800/2400 too — larger upscales reduced confidence further,
        # likely amplifying blur; 1200 is the measured sweet spot.)
        target_w = 1200
        if w < target_w:
            scale = target_w / w
            frame = cv2.resize(
                frame, (target_w, int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        elif w > 1600:
            # Downscale only if very large
            scale = 1600 / w
            frame = cv2.resize(
                frame, (1600, int(h * scale)), interpolation=cv2.INTER_AREA
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Bilateral filter — smooths noise while keeping text edges sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # 3. CLAHE for contrast normalisation (helps with uneven lighting)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced
