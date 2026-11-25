"""SARA line recogniser at run time (OCR v5): reads cropped text lines, Bangla + English, on ONNX Runtime.

The model folder (config.OCR5_REC_DIR) holds
  rec.onnx       the network: (N, 1, 48, W) grey, (x / 127.5 - 1) -> (N, W / 4, C) class scores
  rec_int8.onnx  optional 8-bit copy for the Pi's CPU (used when config.OCR5_REC_INT8 and the file exists)
  charset.txt    one character per line, in class order (class 0 = CTC blank is not listed); a space is "<space>"
  model.json     how it was trained and scored
Trained by ocr_bench/rec5/train.py, exported by ocr_bench/rec5/export.py.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger("smart_glass.ocr")

H = 48


def to_height(grey: np.ndarray) -> np.ndarray:
    h, w = grey.shape[:2]
    nw = max(8, int(round(w * H / max(h, 1))))
    return cv2.resize(grey, (nw, H), interpolation=cv2.INTER_AREA if h > H else cv2.INTER_CUBIC)


class EasyOcrLineReader:
    """The glass's earlier letter reader (EasyOCR, Bangla + English, with the fine-tuned weights of
    config.OCR_RECOGNIZER_PATH) behind the same read(crops) interface, for OCR v5 with OCR5_RECOGNIZER=easyocr.
    More accurate on some text, much slower on the Pi's CPU, and it needs PyTorch."""

    def __init__(self):
        import config
        import easyocr
        import ocr_recognizer_ft
        try:
            import torch
            gpu = bool(torch.cuda.is_available()) and os.environ.get("OCR_GPU", "1") == "1"
        except Exception:  # noqa: BLE001
            gpu = False
        self.reader = easyocr.Reader(["bn", "en"], gpu=gpu, verbose=False, detector=False)
        ocr_recognizer_ft.install(self.reader)
        self.path = "easyocr:" + os.path.basename(getattr(config, "OCR_RECOGNIZER_PATH", "") or "released")
        self.min_conf = float(getattr(config, "OCR_V2_MIN_CONF", 0.1))   # EasyOCR's own scale, as in v2/v4
        self.flip_below = 0.2
        logger.info("OCR v5 recogniser: EasyOCR (%s)", self.path)

    def read(self, crops: list, batch: int = 16) -> list:
        out = []
        for c in crops:
            g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if c.ndim == 3 else c
            if g.shape[0] > 64:                                     # EasyOCR reads at height 64
                g = cv2.resize(g, (max(8, int(g.shape[1] * 64 / g.shape[0])), 64), interpolation=cv2.INTER_AREA)
            h, w = g.shape[:2]
            try:
                res = self.reader.recognize(g, horizontal_list=[[0, w, 0, h]], free_list=[], detail=1,
                                            decoder="greedy", paragraph=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("EasyOCR line read failed: %s", exc)
                res = []
            if res:
                out.append((" ".join(r[1] for r in res).strip(), float(np.mean([r[2] for r in res]))))
            else:
                out.append(("", 0.0))
        return out


def make_recognizer():
    """The line reader named by config.OCR5_RECOGNIZER ("sara" = SaraRecognizer, "easyocr" = EasyOcrLineReader)."""
    import config
    if getattr(config, "OCR5_RECOGNIZER", "sara") == "easyocr":
        return EasyOcrLineReader()
    return SaraRecognizer(config.OCR5_REC_DIR, bool(getattr(config, "OCR5_REC_INT8", True)),
                          int(getattr(config, "OCR5_THREADS", 0)))


class SaraRecognizer:
    def __init__(self, model_dir: str, int8: bool = True, threads: int = 0, max_w: int = 1600):
        import onnxruntime as ort
        path = os.path.join(model_dir, "rec_int8.onnx")
        if not (int8 and os.path.exists(path)):
            path = os.path.join(model_dir, "rec.onnx")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        with open(os.path.join(model_dir, "charset.txt"), encoding="utf-8") as f:
            self.chars = [(" " if c == "<space>" else c) for c in f.read().split("\n") if c != ""]
        self.max_w = max_w
        self.path = path
        logger.info("OCR v5 recogniser: %s (%d characters)", os.path.basename(path), len(self.chars))

    def _decode(self, ids, probs):
        out, conf, prev = [], [], 0
        for k, p in zip(ids, probs):
            if k and k != prev:
                out.append(self.chars[k - 1])
                conf.append(p)
            prev = k
        return " ".join("".join(out).split()), (float(np.mean(conf)) if conf else 0.0)

    def read(self, crops: list, batch: int = 16) -> list:
        """crops: grey (or BGR) line images of any size -> [(text, confidence 0..1)] in the same order."""
        lines = []
        for c in crops:
            if c.ndim == 3:
                c = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            im = to_height(c)
            if im.shape[1] > self.max_w:
                im = cv2.resize(im, (self.max_w, H), interpolation=cv2.INTER_AREA)
            lines.append(im)
        order = sorted(range(len(lines)), key=lambda i: lines[i].shape[1])
        out = [("", 0.0)] * len(lines)
        for k in range(0, len(order), batch):
            idx = order[k:k + batch]
            ws = [lines[i].shape[1] for i in idx]
            W = max(32, -(-max(ws) // 8) * 8)
            x = np.empty((len(idx), 1, H, W), np.float32)
            for j, i in enumerate(idx):
                w = ws[j]
                x[j, 0, :, :w] = lines[i]
                x[j, 0, :, w:] = lines[i][:, -1:]
            x = x / 127.5 - 1.0
            y = self.sess.run(None, {self.inp: x})[0]                       # N, T, C
            y = y - y.max(-1, keepdims=True)
            p = np.exp(y)
            p /= p.sum(-1, keepdims=True)
            ids, conf = p.argmax(-1), p.max(-1)
            for j, i in enumerate(idx):
                n = max(1, ws[j] // 4)
                out[i] = self._decode(ids[j, :n].tolist(), conf[j, :n].tolist())
        return out
