"""Text finder for OCR v5: PaddleOCR PP-OCRv5 mobile text detector (DBNet, Apache-2.0, 4.8 MB).

The same network runs in two places, so the laptop behaves like the glass:
  * on the Raspberry Pi AI HAT (Hailo-8): Hailo's stock compile, models/hailo/paddle_ocr_v5_mobile_detection.hef,
    544 x 960 uint8 in, normalisation inside (hailo_text_detector.HailoTextDetector);
  * on any CPU (laptop, or the Pi without the HAT): models/ppocr/ppocrv5_mobile_det.onnx with ONNX Runtime.
Both give a text-probability map; boxes come out of the same PaddleOCR DBPostProcess (hailo_text_detector.db_boxes).

TextFinder.find(img) -> list of (quad float32 4x2 in img pixels, score), one per text line.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger("smart_glass.ocr")

_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


class OnnxDet:
    """PP-OCRv5 mobile detector on ONNX Runtime. limit = longer side the picture is scaled to (multiple of 32)."""

    def __init__(self, path: str, limit: int = 960, threads: int = 0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.limit = limit

    def prob(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = min(1.0 * self.limit / max(h, w), 4.0)
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR).astype(np.float32)
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        p = self.sess.run(None, {self.inp: x})[0][0, 0]
        return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)

    def close(self):
        self.sess = None


class TextFinder:
    """Chip first when asked for (and working), CPU otherwise; one interface for the OCR v5 pipeline."""

    def __init__(self, backend: str = "auto"):
        self.backend, self.det = "", None
        if backend in ("hailo", "auto") and getattr(config, "AI_HAT", False) or backend == "hailo":
            try:
                from hailo_text_detector import HailoTextDetector
                self.det = HailoTextDetector(config.HAILO_OCR_DET_HEF, getattr(config, "OCR_HAILO_MODE", "tile"))
                self.det.prob(np.zeros((64, 64, 3), np.uint8))
                self.backend = "hailo"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hailo text finder not usable (%s); using the CPU", exc)
                self.det = None
        if self.det is None:
            self.det = OnnxDet(config.OCR5_DET_ONNX, int(getattr(config, "OCR5_DET_LIMIT", 960)),
                               int(getattr(config, "OCR5_THREADS", 0)))
            self.backend = "cpu"
        logger.info("OCR v5 text finder: %s", self.backend)

    def prob(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            return self.det.prob(img)
        except Exception as exc:  # noqa: BLE001
            if self.backend != "hailo":
                raise
            logger.warning("Hailo text finder failed (%s); switching to the CPU for good", exc)
            try:
                self.det.close()
            except Exception:  # noqa: BLE001
                pass
            self.det = OnnxDet(config.OCR5_DET_ONNX, int(getattr(config, "OCR5_DET_LIMIT", 960)))
            self.backend = "cpu"
            return self.det.prob(img)

    def find(self, img: np.ndarray, thresh=0.3, box_thresh=0.5, unclip=1.6):
        from hailo_text_detector import db_boxes
        p = self.prob(img)
        return db_boxes(p, thresh, box_thresh, unclip), p

    def close(self):
        try:
            self.det.close()
        except Exception:  # noqa: BLE001
            pass
