"""
Windows Development / Testing Script
=====================================
Tests all three Smart Glass modes on Windows using:
  - Webcam via OpenCV
  - tkinter window for live camera display
  - TTS: Piper if installed, else gTTS for Bangla, else Windows SAPI (English only)
  - Keyboard keys instead of GPIO buttons

Keyboard controls (click the window first):
  M  ->  Cycle mode    (OCR -> Object -> Currency -> Online Claude)
  A  ->  CAPTURE       (OCR: snap + EasyOCR + store text — offline
                        | Object/Currency: detect and announce immediately
                        | Claude: send the frame to Claude online)
  C  ->  CLAUDE        (always online: send current camera frame to Claude API)
  R  ->  READ          (OCR: speak last EasyOCR text | Claude: speak last online result)
  +  ->  Volume up
  -  ->  Volume down
  Q  ->  Quit

Install:
  pip install easyocr ultralytics torch torchvision numpy Pillow gtts pygame
"""

import os
import re
import sys
import queue
import logging
import subprocess
import tempfile
import threading
import time
import tkinter as tk
import warnings
from tkinter import ttk

# EasyOCR sets pin_memory=True internally; suppress the GPU-not-found noise
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)

import cv2
import numpy as np
from PIL import Image, ImageTk

_BANGLA_RE = re.compile(r"[ঀ-৿]")

# ---------------------------------------------------------------------------
# Logging with UTF-8 so Bangla prints in terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        )
    ],
)
logger = logging.getLogger("smart_glass_win")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODE_OCR, MODE_OBJECT, MODE_CURRENCY, MODE_CLAUDE = 0, 1, 2, 3
MODE_NAMES_EN = [
    "Text Reading Mode",
    "Object Detection Mode",
    "Currency Detection Mode",
    "Online Claude Mode",
]
MODE_NAMES_BN = [
    "টেক্সট রিডিং মোড",
    "অবজেক্ট ডিটেকশন মোড",
    "কারেন্সি ডিটেকশন মোড",
    "অনলাইন ক্লড মোড",
]

YOLO_MODEL_PATH     = os.path.join(BASE_DIR, "models", "yolov8n.pt")
CURRENCY_MODEL_PATH = os.path.join(BASE_DIR, "models", "currency_mobilenet.pt")
LABELS_BN_PATH      = os.path.join(BASE_DIR, "assets", "labels_bn.json")

# Piper TTS — drop piper.exe + the .onnx models into models/piper/ to enable
# natural neural TTS. Falls back to Windows SAPI automatically if not present.
# Download: https://github.com/rhasspy/piper/releases  (piper_windows_amd64.zip)
PIPER_EXE      = os.path.join(BASE_DIR, "models", "piper", "piper.exe")
PIPER_EN_MODEL = os.path.join(BASE_DIR, "models", "piper", "en_US-hfc_female-medium.onnx")
PIPER_BN_MODEL = os.path.join(BASE_DIR, "models", "piper", "bn_BD-medium.onnx")

OCR_CONFIDENCE           = 0.4
OBJECT_CONFIDENCE        = 0.50
CURRENCY_CONFIDENCE      = 0.65
OCR_MIN_CHARS            = 3
OBJECT_SCAN_INTERVAL     = 2.0
OBJECT_ANNOUNCE_COOLDOWN = 4.0
VOLUME_STEP              = 10
CAMERA_INDEX             = 0
DISPLAY_W, DISPLAY_H    = 640, 480


# Known Bangla phrases → English, used only if gTTS/Piper are unavailable.
_BN_EN_FALLBACK = (
    ("সামনে আছে: ", "In front: "),
    ("সামনে আছে:", "In front:"),
    ("দুই টাকার নোট", "2 taka"),
    ("পাঁচ টাকার নোট", "5 taka"),
    ("দশ টাকার নোট", "10 taka"),
    ("বিশ টাকার নোট", "20 taka"),
    ("পঞ্চাশ টাকার নোট", "50 taka"),
    ("একশত টাকার নোট", "100 taka"),
    ("দুইশত টাকার নোট", "200 taka"),
    ("পাঁচশত টাকার নোট", "500 taka"),
    ("এক হাজার টাকার নোট", "1000 taka"),
    # authenticity verdict appended by CurrencyMode.detect_live
    ("জাল যাচাই করা হয়নি", "Authenticity not checked"),
    ("জাল টাকা", "Counterfeit"),
    ("আসল", "Genuine"),
    # emotion-adaptive prefixes (roboeye.fer_emotion.tts_style_for_emotion)
    ("ঠিক আছে। ধীরে শুনুন। ", "It's okay. Take your time. "),
    ("রাগ করবেন না। ", "It's alright. "),
    ("ভালো। ", "Good. "),
    ("লেখা সংরক্ষণ করা হয়েছে। শুনতে রিড বাটন চাপুন", "Text saved. Press R to listen."),
    ("পড়া হচ্ছে: ", "Reading: "),
    ("কোনো লেখা পাওয়া যায়নি", "No text found."),
    ("কোনো লেখা সংরক্ষিত নেই। প্রথমে ক্যাপচার করুন", "Nothing saved yet. Capture first."),
    ("নোট সনাক্ত করা যায়নি। ক্যামেরার সামনে ধরুন।", "No note found. Hold it in front of the camera."),
    ("নোট নিশ্চিত করা যায়নি। আরও কাছে ধরুন।", "Could not confirm the note. Hold it closer."),
    ("ক্যামেরা প্রস্তুত নয়", "Camera not ready."),
    ("অনলাইন ক্লড মোড", "Online Claude Mode"),
    ("কোনো অনলাইন বিবরণ নেই। আগে C চাপুন।", "No online description yet. Press C first."),
)


def _object_names_bn_to_en() -> list[tuple[str, str]]:
    """Bangla COCO names (assets/labels_bn.json) back to English, longest first."""
    try:
        import json
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "labels_bn.json"),
                  encoding="utf-8") as fh:
            en_to_bn = json.load(fh)
    except Exception:
        return []
    return sorted(((bn, en) for en, bn in en_to_bn.items()), key=lambda p: -len(p[0]))


_OBJECT_BN_EN = _object_names_bn_to_en()


def _english_fallback(text: str) -> str:
    out = text
    for bn, en in _BN_EN_FALLBACK:
        out = out.replace(bn, en)
    for bn, en in _OBJECT_BN_EN:
        out = out.replace(bn, en)
    out = out.replace("।", ".")
    if _BANGLA_RE.search(out):
        return "Detection complete. Bangla is on the screen."
    return out


# ---------------------------------------------------------------------------
# TTS Engine — Piper (offline) → gTTS (Bangla) → Windows SAPI (English)
# ---------------------------------------------------------------------------

class WindowsTTS:
    """
    Background TTS worker.

    Windows SAPI has no Bangla voice, so Bangla detections used to be replaced
    with a short English cue and were not spoken. Order now:

      1. Piper, if piper.exe + voice models are present
      2. gTTS (needs internet) for Bangla — speaks the real detected text
      3. Windows SAPI for English, or an English gloss of known Bangla phrases
    """

    def __init__(self, volume: int = 85):
        self._volume     = max(0, min(100, volume))
        self._q: queue.Queue = queue.Queue(maxsize=5)
        self._stop_event = threading.Event()
        self._proc       = None
        self._proc_lock  = threading.Lock()
        self._mixer      = None

        self._worker = threading.Thread(
            target=self._run, daemon=True, name="tts-worker"
        )
        self._worker.start()
        logger.info("[TTS] Worker started (Piper / gTTS Bangla / SAPI English)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, emotion: str = "neutral", adaptive: bool = True) -> None:
        if not text or not text.strip():
            return

        try:
            from assistive import adapt_feedback
            bangla = len(_BANGLA_RE.findall(text.replace(" ", ""))) / max(len(text.replace(" ", "")), 1) > 0.25
            adapted = adapt_feedback(text, emotion, enabled=adaptive, bangla=bangla)
            text = adapted["text"]
            rate_delta = int(adapted.get("rate") or 0)
        except Exception:
            rate_delta = 0

        stripped     = text.replace(" ", "")
        bangla_ratio = len(_BANGLA_RE.findall(stripped)) / max(len(stripped), 1)
        lang         = "bn" if bangla_ratio > 0.25 else "en"

        logger.info("[TTS] Queuing (%s, emo=%s, adapt=%s): %s", lang, emotion, adaptive, text[:70])

        if self._q.full():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                pass
        self._q.put_nowait((text.strip(), lang, rate_delta))

    def speaking(self) -> bool:
        """True while something is queued or being spoken (text auto-capture stays quiet meanwhile)."""
        return self._q.unfinished_tasks > 0

    def stop(self) -> None:
        self._stop_event.set()
        if self._mixer is not None:
            try:
                self._mixer.music.stop()
            except Exception:
                pass
        with self._proc_lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
        while not self._q.empty():
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break
        self._stop_event.clear()

    def set_volume(self, pct: int) -> None:
        self._volume = max(0, min(100, pct))

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.mixer.init()
            self._mixer = pygame.mixer
        except Exception as exc:
            logger.warning("[TTS] pygame mixer unavailable (%s) — gTTS playback limited", exc)

        while True:
            item = self._q.get()
            if isinstance(item, tuple) and len(item) == 3:
                text, lang, rate_delta = item
            else:
                text, lang = item
                rate_delta = 0
            try:
                # offline first: Piper voices in-process (as on the glass); then the Piper program; online Google
                # speech only if neither is available
                if self._piper_py_speak(text, lang):
                    pass
                elif os.path.isfile(PIPER_EXE):
                    self._piper_speak(text, lang)
                elif lang == "bn":
                    if not self._gtts_speak(text, "bn"):
                        self._sapi_speak(_english_fallback(text), "en", rate_delta)
                else:
                    self._sapi_speak(text, lang, rate_delta)
            except Exception as exc:
                logger.error("[TTS] Error: %s", exc)
            finally:
                self._q.task_done()

    # ------------------------------------------------------------------
    # Piper TTS backend (natural neural voice, fully offline)
    # ------------------------------------------------------------------

    def _piper_speak(self, text: str, lang: str) -> None:
        model = PIPER_BN_MODEL if lang == "bn" else PIPER_EN_MODEL
        if not os.path.isfile(model):
            logger.warning("[TTS] Piper model not found for lang=%s, falling back", lang)
            if lang == "bn" and self._gtts_speak(text, "bn"):
                return
            self._sapi_speak(text, lang, 0)
            return

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name

            # Render text → WAV
            piper_proc = subprocess.Popen(
                [PIPER_EXE, "--model", model, "--output_file", tmp_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            piper_proc.stdin.write(text.encode("utf-8"))
            piper_proc.stdin.close()
            piper_proc.wait()

            if self._stop_event.is_set():
                return

            # Play WAV via PowerShell SoundPlayer (blocking, killable)
            escaped = tmp_path.replace("\\", "\\\\")
            play_script = (
                f"$p = New-Object System.Media.SoundPlayer '{escaped}'; $p.PlaySync()"
            )
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-WindowStyle", "Hidden", "-Command", play_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._proc_lock:
                self._proc = proc

            while proc.poll() is None:
                if self._stop_event.is_set():
                    proc.terminate()
                    break
                time.sleep(0.05)

            logger.info("[TTS] Piper done")

        except Exception as exc:
            logger.warning("[TTS] Piper failed (%s), falling back to SAPI", exc)
            self._sapi_speak(text, lang, 0)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _piper_py_speak(self, text: str, lang: str) -> bool:
        """Offline speech with the glass's own Piper voices through the piper-tts Python package (no internet, the
        same voices the Pi uses: models/piper/bn_BD-google-medium.onnx and en_US-amy-low.onnx; fetch them once with
        scripts/get_speech_models.py). Mixed Bangla + English text is split by script, each part spoken with its own
        voice. Plays through pygame when its mixer works, else through Windows' own player (winsound). Returns False
        when the package or a voice is missing (the caller then falls back)."""
        try:
            from piper import PiperVoice
        except ImportError:
            return False
        import config as _cfg
        import wave
        models = {"bn": _cfg.PIPER_BN_MODEL, "en": _cfg.PIPER_EN_MODEL}
        if not hasattr(self, "_piper_voices"):
            self._piper_voices = {}
        parts, cur, cl = [], "", None
        for tok in text.split():
            bn = len(_BANGLA_RE.findall(tok))
            lat = sum(c.isascii() and c.isalpha() for c in tok)
            t = "bn" if bn and bn >= lat else ("en" if lat else (cl or lang))
            if cl and t != cl:
                parts.append((cl, cur))
                cur = ""
            cl, cur = t, (cur + " " + tok).strip()
        if cur:
            parts.append((cl, cur))
        for pl, part in parts:
            if pl not in self._piper_voices:
                p = models[pl]
                self._piper_voices[pl] = PiperVoice.load(p) if os.path.isfile(p) and os.path.isfile(p + ".json") else None
            if self._piper_voices[pl] is None:
                logger.warning("[TTS] Piper voice missing for %s (run scripts/get_speech_models.py)", pl)
                return False
        for pl, part in parts:
            if self._stop_event.is_set():
                return True
            v = self._piper_voices[pl]
            if hasattr(v, "synthesize_stream_raw"):
                pcm = b"".join(v.synthesize_stream_raw(part))
            else:
                pcm = b"".join(c.audio_int16_bytes for c in v.synthesize(part))
            rate = int(v.config.sample_rate)
            if self._mixer is None:                        # Windows' own player: volume applied to the samples
                pcm = (np.frombuffer(pcm, np.int16).astype(np.float32) * (self._volume / 100.0)).clip(-32768, 32767).astype(np.int16).tobytes()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                with wave.open(tmp_path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(rate)
                    w.writeframes(pcm)
                logger.info("[TTS] Piper offline (%s): %s", pl, part[:60])
                if self._mixer is not None:
                    self._mixer.music.load(tmp_path)
                    self._mixer.music.set_volume(self._volume / 100.0)
                    self._mixer.music.play()
                    while self._mixer.music.get_busy():
                        if self._stop_event.is_set():
                            self._mixer.music.stop()
                            break
                        time.sleep(0.03)
                    try:
                        self._mixer.music.unload()
                    except Exception:
                        pass
                else:
                    import winsound
                    winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                    end = time.time() + len(pcm) / 2 / rate + 0.1
                    while time.time() < end:
                        if self._stop_event.is_set():
                            winsound.PlaySound(None, 0)
                            break
                        time.sleep(0.03)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        return True

    def _gtts_speak(self, text: str, lang: str) -> bool:
        """Speak via Google TTS. Returns False if download/playback fails."""
        if self._mixer is None:
            return False
        tmp_path = None
        try:
            from gtts import gTTS
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            gTTS(text=text, lang=lang).save(tmp_path)
            if self._stop_event.is_set():
                return True
            logger.info("[TTS] gTTS (%s): %s", lang, text[:60])
            self._mixer.music.load(tmp_path)
            self._mixer.music.set_volume(self._volume / 100.0)
            self._mixer.music.play()
            while self._mixer.music.get_busy():
                if self._stop_event.is_set():
                    self._mixer.music.stop()
                    break
                time.sleep(0.05)
            try:
                self._mixer.music.unload()
            except Exception:
                pass
            logger.info("[TTS] Done (gTTS)")
            return True
        except Exception as exc:
            logger.warning("[TTS] gTTS failed (%s)", exc)
            return False
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Windows SAPI backend (offline English fallback)
    # ------------------------------------------------------------------

    def _sapi_speak(self, text: str, lang: str, rate_delta: int = 0) -> None:
        speak_text = text if lang == "en" else _english_fallback(text)
        safe = speak_text.replace("'", "''")
        sapi_rate = max(-10, min(10, 1 + int(rate_delta)))
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Volume = {self._volume}; $s.Rate = {sapi_rate}; $s.Speak('{safe}');"
        )
        logger.info("[TTS] SAPI: %s", speak_text[:60])
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-WindowStyle", "Hidden", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._proc_lock:
            self._proc = proc

        while proc.poll() is None:
            if self._stop_event.is_set():
                proc.terminate()
                break
            time.sleep(0.05)

        logger.info("[TTS] Done (SAPI)")


# ---------------------------------------------------------------------------
# Load modes
# ---------------------------------------------------------------------------

def load_modes():
    sys.path.insert(0, BASE_DIR)
    from modes.ocr_mode      import OCRMode
    from modes.object_mode   import ObjectMode
    from modes.currency_mode import CurrencyMode
    from modes.claude_mode   import ClaudeMode
    return [OCRMode(), ObjectMode(), CurrencyMode(), ClaudeMode()]


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class SmartGlassWin:

    def __init__(self, root: tk.Tk):
        self._root       = root
        self._tts        = WindowsTTS(volume=90)
        self._modes      = load_modes()
        self._mode       = MODE_OCR
        self._volume     = 90
        self._lock       = threading.Lock()
        self._running    = True
        self._detecting  = False
        self._frame      = None
        self._frame_lock = threading.Lock()
        self._emotion_label = "neutral"
        self._emotion_face = None
        self._adaptive_on = True
        self._pose = None
        self._currency_hits = []
        self._currency_spoken = None
        self._currency_votes = __import__("collections").deque(maxlen=4)   # last scans: (value, verdict, text)
        self._currency_last_seen = 0.0
        self._straighten_spoken = False
        self._emo_lock = threading.Lock()
        try:
            from assistive import EmotionDetector
            self._emotion = EmotionDetector(download=True)
        except Exception as exc:
            self._emotion = None
            logger.warning("Emotion detector unavailable: %s", exc)

        self._build_ui()
        self._start_camera()
        self._start_scan_thread()
        self._start_emotion_thread()

        self._modes[self._mode].activate()
        if self._emotion is not None:
            self._log("Emotion detector ready (FER+ / HGB). E toggles adaptive speech.")
        else:
            self._log("Emotion detector unavailable.")
        self._log("Features: 1) OCR EasyOCR bn+en  2) Object YOLOv8  3) Currency YOLO+pose")
        self._log("          4) Claude online  5) Emotion HUD  6) Adaptive TTS  7) Volume")
        self._tts.speak("Smart Glass ready. Text Reading Mode.")
        self._log("Ready | M=mode A=capture C=Claude E=emotion adaptive R=read +/-=vol Q=quit")

        root.bind("<Key-m>",     lambda e: self._switch_mode())
        root.bind("<Key-M>",     lambda e: self._switch_mode())
        root.bind("<Key-a>",     lambda e: self._action())
        root.bind("<Key-A>",     lambda e: self._action())
        root.bind("<Key-c>",     lambda e: self._claude_action())
        root.bind("<Key-C>",     lambda e: self._claude_action())
        root.bind("<Key-e>",     lambda e: self._toggle_adaptive())
        root.bind("<Key-E>",     lambda e: self._toggle_adaptive())
        root.bind("<Key-r>",     lambda e: self._read_stored())
        root.bind("<Key-R>",     lambda e: self._read_stored())
        root.bind("<Key-plus>",  lambda e: self._vol_up())
        root.bind("<Key-equal>", lambda e: self._vol_up())
        root.bind("<Key-minus>", lambda e: self._vol_down())
        root.bind("<Key-q>",     lambda e: self._quit())
        root.bind("<Key-Q>",     lambda e: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        self._refresh_frame()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._root.title("Smart Glass — Windows Test")
        self._root.configure(bg="#1a1a2e")
        self._root.resizable(False, False)

        # Camera feed
        self._canvas = tk.Canvas(
            self._root, width=DISPLAY_W, height=DISPLAY_H,
            bg="black", highlightthickness=0,
        )
        self._canvas.pack()

        # Top info row
        info_frame = tk.Frame(self._root, bg="#1a1a2e")
        info_frame.pack(fill=tk.X, padx=10, pady=(6, 0))

        self._mode_var = tk.StringVar(value=f"● {MODE_NAMES_EN[self._mode]}")
        tk.Label(info_frame, textvariable=self._mode_var,
                 font=("Segoe UI", 12, "bold"), fg="#00ff88", bg="#1a1a2e"
                 ).pack(side=tk.LEFT)

        self._hud_var = tk.StringVar(value="emotion: —  adapt: ON  pose: —")
        tk.Label(info_frame, textvariable=self._hud_var,
                 font=("Segoe UI", 9), fg="#7ec8e3", bg="#1a1a2e"
                 ).pack(side=tk.LEFT, padx=14)

        self._tts_var = tk.StringVar(value="ready")
        tk.Label(info_frame, textvariable=self._tts_var,
                 font=("Segoe UI", 10), fg="#aaaaaa", bg="#1a1a2e"
                 ).pack(side=tk.RIGHT)

        # Result area — Nirmala UI renders Bangla
        self._result_var = tk.StringVar(value="Press A for offline OCR, or C for online Claude")
        tk.Label(
            self._root, textvariable=self._result_var,
            font=("Nirmala UI", 13), fg="#ffffff", bg="#1a1a2e",
            wraplength=630, justify="left",
        ).pack(padx=10, pady=(4, 2))

        # Clickable demo buttons — every feature on screen
        btn_row = tk.Frame(self._root, bg="#1a1a2e")
        btn_row.pack(fill=tk.X, padx=8, pady=(4, 2))
        buttons = [
            ("Mode (M)", self._switch_mode),
            ("Capture (A)", self._action),
            ("Claude (C)", self._claude_action),
            ("Emotion (E)", self._toggle_adaptive),
            ("Read (R)", self._read_stored),
            ("Vol +", self._vol_up),
            ("Vol -", self._vol_down),
            ("Quit (Q)", self._quit),
        ]
        for label, cmd in buttons:
            def _click(fn=cmd):
                fn()
                self._root.focus_set()
            tk.Button(
                btn_row, text=label, command=_click,
                font=("Segoe UI", 8, "bold"), fg="#111", bg="#00ff88",
                activebackground="#66ffaa", relief=tk.FLAT, padx=7, pady=3,
                takefocus=0,
            ).pack(side=tk.LEFT, padx=2)

        # Volume bar
        vol_row = tk.Frame(self._root, bg="#1a1a2e")
        vol_row.pack(pady=(0, 2))
        tk.Label(vol_row, text="Vol:", fg="#777", bg="#1a1a2e",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._vol_bar = ttk.Progressbar(vol_row, length=130, maximum=100, mode="determinate")
        self._vol_bar["value"] = self._volume
        self._vol_bar.pack(side=tk.LEFT, padx=4)
        self._vol_lbl = tk.Label(vol_row, text=f"{self._volume}%",
                                  fg="#777", bg="#1a1a2e", font=("Segoe UI", 9))
        self._vol_lbl.pack(side=tk.LEFT)

        # Hint
        tk.Label(
            self._root,
            text="  Features: OCR | Object YOLO | Currency YOLO+pose | Claude | Emotion HUD | Adaptive TTS  ",
            font=("Segoe UI", 9), fg="#888", bg="#0d0d1e",
        ).pack(fill=tk.X)

        # Log box
        self._log_box = tk.Text(
            self._root, height=6,
            font=("Nirmala UI", 9),          # Bangla-capable font
            bg="#0a0a18", fg="#88ff88",
            state=tk.DISABLED, relief=tk.FLAT, wrap=tk.WORD,
        )
        self._log_box.pack(fill=tk.X, padx=8, pady=(2, 8))

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_box.config(state=tk.NORMAL)
        self._log_box.insert(tk.END, f"[{ts}] {msg}\n")
        self._log_box.see(tk.END)
        self._log_box.config(state=tk.DISABLED)
        logger.info(msg)

    def _set_tts_status(self, msg: str):
        self._root.after(0, lambda: self._tts_var.set(f"🔊 {msg}"))

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _start_camera(self):
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self._log("ERROR: Cannot open webcam")
            return
        # Ask the webcam for 1280 x 720 (small print needs the pixels; CAMERA_WIDTH / CAMERA_HEIGHT override); the
        # window shows it scaled to DISPLAY_W x DISPLAY_H. A camera that can't do it gives its own size.
        cw = int(os.environ.get("CAMERA_WIDTH", "1280"))
        ch = int(os.environ.get("CAMERA_HEIGHT", "720"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cw)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        def _grab():
            while self._running:
                ret, frame = self._cap.read()
                if ret:
                    with self._frame_lock:
                        self._frame = frame
                else:
                    time.sleep(0.01)

        threading.Thread(target=_grab, daemon=True, name="camera").start()
        self._log(f"Camera started ({int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")

    def _refresh_frame(self):
        with self._frame_lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is not None:
            vis = frame.copy()
            with self._emo_lock:
                face = self._emotion_face
                label = self._emotion_label
                adaptive = self._adaptive_on
                pose = self._pose
            if face:
                x, y, w, h = face
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.putText(
                    vis, f"{label}  adapt={'ON' if adaptive else 'OFF'}",
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2,
                )
            else:
                cv2.putText(
                    vis, f"emotion:{label} adapt={'ON' if adaptive else 'OFF'}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2,
                )
            with self._emo_lock:
                hits = list(self._currency_hits)
            for hit in hits:
                x, y, w, h = hit["bbox"]
                auth = hit.get("auth")
                if auth == "counterfeit":
                    color = (0, 0, 255)
                elif auth == "genuine":
                    color = (0, 220, 0)
                else:
                    color = (0, 220, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                tag = hit.get("auth_en") or ""
                cv2.putText(
                    vis, f"{hit['name']} {tag} {hit['conf']*100:.0f}%",
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )
            if pose and pose.get("ok") and pose.get("quad") is not None:
                pts = pose["quad"].astype(np.int32).reshape((-1, 1, 2))
                color = (0, 0, 255) if pose.get("needs_straighten") else (0, 255, 255)
                cv2.polylines(vis, [pts], True, color, 2)
                msg = f"R{pose['roll']:+.0f} P{pose['pitch']:+.0f} Y{pose['yaw']:+.0f}"
                if pose.get("needs_straighten"):
                    msg += "  STRAIGHTEN"
                cv2.putText(vis, msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                pose_txt = msg
            else:
                pose_txt = "—"
            with self._lock:
                in_ocr = self._mode == MODE_OCR
            if in_ocr:
                ocr = self._modes[MODE_OCR]
                # text boxes of the last reading (green = read and spoken, red = not sure, not spoken)
                if time.time() - getattr(ocr, "last_boxes_t", 0) < 8:
                    for b in getattr(ocr, "last_boxes", []) or []:
                        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0) if b[5] else (0, 0, 255), 2)
                st = ocr.auto_state() if hasattr(ocr, "auto_state") else ""
                if st:
                    names = {"looking": "looking for text", "small": "text too small: closer",
                             "steadying": "hold still...", "blurry": "blurry: hold still", "fired": "reading",
                             "done": "read - move to the next text (or press A)"}
                    cv2.putText(vis, "AUTO: " + names.get(st, st), (10, vis.shape[0] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9 * vis.shape[1] / 1280 + 0.3, (0, 255, 255), 2)
            self._hud_var.set(
                f"emotion: {label}  adapt: {'ON' if adaptive else 'OFF'}  pose: {pose_txt}"
            )
            if vis.shape[1] != DISPLAY_W or vis.shape[0] != DISPLAY_H:
                vis = cv2.resize(vis, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            self._tk_img = ImageTk.PhotoImage(image=img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        if self._running:
            self._root.after(33, self._refresh_frame)

    def _start_emotion_thread(self):
        def _loop():
            while self._running:
                if self._emotion is None:
                    time.sleep(0.5)
                    continue
                with self._frame_lock:
                    frame = self._frame.copy() if self._frame is not None else None
                if frame is not None:
                    try:
                        result = self._emotion.predict(frame)
                        with self._emo_lock:
                            self._emotion_label = result.get("label") or "neutral"
                            self._emotion_face = result.get("face")
                    except Exception as exc:
                        logger.warning("Emotion error: %s", exc)
                time.sleep(0.45)
        threading.Thread(target=_loop, daemon=True, name="emotion").start()

    def _toggle_adaptive(self):
        with self._emo_lock:
            self._adaptive_on = not self._adaptive_on
            on = self._adaptive_on
        msg = "Emotion adaptive feedback on." if on else "Emotion adaptive feedback off."
        self._log(msg)
        self._tts.speak(msg, emotion="neutral", adaptive=False)

    def _speak(self, text: str) -> None:
        with self._emo_lock:
            emo = self._emotion_label
            on = self._adaptive_on
        self._tts.speak(text, emotion=emo, adaptive=on)

    def _latest_frame(self):
        """A copy of the newest camera frame (more frames for the UV check)."""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def _update_pose_from_currency(self, frame):
        mode = self._modes[MODE_CURRENCY]
        bbox = getattr(mode, "last_bbox", None)
        name = getattr(mode, "last_class", None)
        if not bbox or not name:
            with self._emo_lock:
                self._pose = None
            return
        x, y, w, h = bbox
        try:
            from assistive import estimate_pose
            pose = estimate_pose(frame, (x, y, x + w, y + h), name)
        except Exception as exc:
            logger.warning("Pose error: %s", exc)
            pose = None
        with self._emo_lock:
            self._pose = pose
        # remembered for the spoken hint (_currency_hint): "hold it straight" only when the note looked crooked
        if not hasattr(self, "_straighten_votes"):
            self._straighten_votes = __import__("collections").deque(maxlen=4)
        self._straighten_votes.append(bool(pose and pose.get("needs_straighten")))

    # ------------------------------------------------------------------
    # Object mode auto-scan
    # ------------------------------------------------------------------

    def _start_scan_thread(self):
        def _loop():
            while self._running:
                with self._lock:
                    mode = self._mode
                if mode == MODE_OBJECT:
                    with self._frame_lock:
                        frame = self._frame.copy() if self._frame is not None else None
                    if frame is not None and not self._detecting:
                        try:
                            result = self._modes[MODE_OBJECT].process_frame(frame)
                            if result:
                                self._root.after(0, self._show_result, result)
                                self._set_tts_status("speaking...")
                                self._speak(result)
                                self._set_tts_status("ready")
                        except Exception as exc:
                            logger.warning("Object scan error: %s", exc)
                    time.sleep(OBJECT_SCAN_INTERVAL)
                elif mode == MODE_CURRENCY:
                    with self._frame_lock:
                        frame = self._frame.copy() if self._frame is not None else None
                    if frame is not None and not self._detecting:
                        try:
                            import config
                            from modes.currency_mode import is_uv_frame
                            cur = self._modes[MODE_CURRENCY]
                            # the note is under the UV lamp: speak the UV fibre check once, as a press would (main.py)
                            if getattr(config, "UV_FRAME_CHECK_ENABLED", False) and is_uv_frame(frame):
                                if self._currency_spoken != "uv":
                                    self._currency_spoken = "uv"
                                    cur.frame_source = self._latest_frame
                                    result = cur.process_frame(frame)
                                    self._root.after(0, self._show_result, result)
                                    self._speak(result)
                                time.sleep(0.35)
                                continue
                            hits = cur.detect_live(frame)
                            with self._emo_lock:
                                self._currency_hits = hits
                            now = time.time()
                            # every physical note in view, followed over the scans and named once it is sure
                            # (_currency_track_notes): a brief dip in one note's confidence no longer changes the set
                            notes = self._currency_track_notes(hits, now)
                            if hits:   # every boxed note and its confidence, for tuning from real tests (log only)
                                logger.info("Currency boxes: %s | named: %s",
                                            ", ".join(f"{h['name']} {h['conf']:.2f} {h['bbox'][2]}x{h['bbox'][3]}" for h in hits),
                                            ", ".join(n["name"] for n in notes) or "-")
                                self._update_pose_from_currency(frame)   # keeps the STRAIGHTEN overlay live
                            self._currency_votes.append((tuple(sorted(n["name"] for n in notes)), notes))
                            say = self._currency_stable_announcement()
                            if not say:
                                say = self._currency_hint(now)
                            if say:
                                self._root.after(0, self._show_result, say)
                                self._speak(say)
                            if hits:
                                self._currency_last_any = now
                            if now - getattr(self, "_currency_last_any", 0.0) > 1.5:
                                # no note in view at all for 1.5 s: it was taken away, so the next note (even the
                                # same value) is announced again as soon as it is shown. A dip in confidence while the
                                # note is still boxed is NOT a removal (it re-announced the same 10 Taka note three
                                # times in 7 s in the 25 Nov laptop test).
                                self._currency_spoken = None
                                with self._emo_lock:
                                    self._pose = None
                        except Exception as exc:
                            logger.warning("Currency scan error: %s", exc)
                    time.sleep(0.35)
                elif mode == MODE_OCR and self._ocr_auto_on():
                    try:
                        self._ocr_auto_step()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Auto text capture error: %s", exc)
                    time.sleep(0.12)
                else:
                    with self._emo_lock:
                        self._currency_hits = []
                    time.sleep(0.2)

        threading.Thread(target=_loop, daemon=True, name="scan").start()

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        iw = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        ih = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = iw * ih
        return inter / float(aw * ah + bw * bh - inter + 1e-9)

    def _currency_track_notes(self, hits, now):
        """The notes in view, each followed over the scans and named once it is sure. Returns the named notes.

        Laptop test of 25 Nov 2026 (183 scans logged): with two notes side by side their confidences rose and fell
        from frame to frame, so the set of named notes kept changing ({10, 2} -> {10} -> {10, 2}) and the voice
        switched between "2 notes ..." and "10 Taka"; and 5, 50, 100 and 1,000 Taka never reached the naming bar
        in a single frame although they were boxed.
          * One note = one track. Every box of at least 0.25 is matched to the track it overlaps most (>= 0.3);
            several boxes on one note in a scan (overlap > CURRENCY_COUNT_IOU) are competing guesses for that note.
          * A track gets its name (and keeps it while it stays in view) when
              - its best guess reaches the validated naming bar (config.CURRENCY_ANNOUNCE_CONF) in one scan, or
              - its value is consistent: the same top guess in at least 4 of its last 5 scans, at a mean confidence
                of at least 0.5, and no other value scoring within 0.15 of it in those scans
                (CURRENCY_CONSISTENT_N / _MIN_CONF / _MARGIN).
            A named track changes its value only on the same evidence for another value.
          * A note counts while it was boxed in the last second, so a brief dip does not drop it from the count.
            Tracks not boxed for 1 s are forgotten (the note was taken away)."""
        import config
        bar = float(config.CURRENCY_ANNOUNCE_CONF)
        need = int(os.environ.get("CURRENCY_CONSISTENT_N", "4"))
        min_conf = float(os.environ.get("CURRENCY_CONSISTENT_MIN_CONF", "0.5"))
        margin = float(os.environ.get("CURRENCY_CONSISTENT_MARGIN", "0.15"))
        same = float(getattr(config, "CURRENCY_COUNT_IOU", 0.5))
        if not hasattr(self, "_cur_tracks"):
            self._cur_tracks = []
        # group the boxes of this scan by physical note: a box overlapping a more confident one is another guess
        groups = []
        for h in sorted(hits, key=lambda h: -h["conf"]):
            for g in groups:
                if self._iou(h["bbox"], g[0]["bbox"]) > same:
                    g.append(h)
                    break
            else:
                groups.append([h])
        used = set()
        for g in groups:
            best, bi = None, 0.3
            for k, t in enumerate(self._cur_tracks):
                v = self._iou(g[0]["bbox"], t["bbox"])
                if k not in used and v >= bi:
                    best, bi = k, v
            if best is None:
                self._cur_tracks.append({"bbox": g[0]["bbox"], "scans": [], "last": now, "first": now, "name": None})
                best = len(self._cur_tracks) - 1
            used.add(best)
            t = self._cur_tracks[best]
            t["bbox"], t["last"] = g[0]["bbox"], now
            scores = {}
            for h in g:
                scores[h["name"]] = max(scores.get(h["name"], 0.0), h["conf"])
            t["scans"] = (t["scans"] + [{"scores": scores, "top": g[0]}])[-5:]
            # naming: a sure scan, or a consistent value over the last scans
            cand = None
            if g[0]["conf"] >= bar:
                cand = g[0]["name"]
            elif len(t["scans"]) >= need:
                tops = [max(s["scores"], key=s["scores"].get) for s in t["scans"]]
                v = max(set(tops), key=tops.count)
                if tops.count(v) >= need:
                    mine = [s["scores"][v] for s in t["scans"] if v in s["scores"]]
                    rival = max([c for s in t["scans"] for n, c in s["scores"].items() if n != v] or [0.0])
                    if sum(mine) / len(mine) >= min_conf and rival <= sum(mine) / len(mine) - margin:
                        cand = v
            # a named note whose top guess has been another value in the last 2 scans (at >= 0.4) is not that value
            # any more: a note swapped for another in the same place kept the old name (25 Nov test: 50 Taka boxed,
            # "2 Taka" remembered). Its name is dropped and earned again from fresh evidence.
            if t["name"] and not cand and len(t["scans"]) >= 2:
                last2 = t["scans"][-2:]
                if all(max(s["scores"], key=s["scores"].get) != t["name"] and max(s["scores"].values()) >= 0.4
                       for s in last2):
                    t["name"] = None
                    t["scans"] = t["scans"][-2:]
            if cand and cand != t["name"]:
                t["name"] = cand
                t["named_scan"] = dict(g[0], name=cand, conf=t["scans"][-1]["scores"].get(cand, g[0]["conf"]))
            elif cand == t["name"] and cand:
                t["named_scan"] = dict(g[0], name=cand, conf=t["scans"][-1]["scores"].get(cand, g[0]["conf"]))
        self._cur_tracks = [t for t in self._cur_tracks if now - t["last"] <= 1.0]
        # a note counts while it was boxed in the last second; overlapping tracks are one note
        notes, boxes = [], []
        for t in sorted(self._cur_tracks, key=lambda t: t["first"]):
            if t["name"] and all(self._iou(t["bbox"], b) <= same for b in boxes):
                notes.append(dict(t["named_scan"], bbox=t["bbox"]))
                boxes.append(t["bbox"])
        return notes

    def _currency_hint(self, now):
        """A spoken hint when a note has been in view for 3 s without being named: "নোট সোজা করে ধরুন" when it looked
        crooked in 3 of the last 4 scans, else "নোটটি আরও কাছে, আলোতে ধরুন" (low confidence usually comes from a
        small, dark or blurred note). At most once every 30 s."""
        unnamed = [t for t in getattr(self, "_cur_tracks", []) if not t["name"] and now - t["last"] <= 0.5
                   and now - t["first"] >= 3.0]
        if not unnamed or now - getattr(self, "_currency_hint_t", 0.0) < 30.0:
            return None
        self._currency_hint_t = now
        if sum(getattr(self, "_straighten_votes", [])) >= 3:
            return "নোট সোজা করে ধরুন"
        return "নোটটি আরও কাছে, আলোতে ধরুন"

    def _currency_stable_announcement(self):
        """Live currency speech for everything in view: one note or several.

        Each scan gives the set of named notes (_currency_track_notes). A set is spoken when it is stable: the first
        time after nothing was named, when it was seen in 2 of the last 3 scans (~0.7 s); a CHANGE to a different
        value or a different set of notes needs 3 of the last 4 scans, so a detector wavering between two values on
        one note (5 / 10 Taka in the 25 Nov laptop test) stays silent.
        The same set is said again every CURRENCY_REPEAT_S seconds (6) while it stays in view, and at once when it
        is shown again after ~1 s with nothing named.
          one note    the value, then one short verdict decided over those scans: "সম্ভবত আসল" when most were likely
                      genuine, "আসল কিনা হাতে যাচাই করুন" for values with a counterfeit check (200 / 500 / 1,000),
                      nothing more for values without one
          several     how many, each value, and the total (CurrencyMode.total_sentence, as a press on the glass);
                      counterfeit checks stay single-note, as on the glass
        Returns the sentence to speak, or None."""
        import config
        votes = list(self._currency_votes)
        sigs = [v[0] for v in votes if v[0]]
        if not sigs:
            return None
        last3, last4 = [v[0] for v in votes[-3:]], [v[0] for v in votes[-4:]]
        sig = max(set(sigs), key=sigs.count)
        first = self._currency_spoken is None
        if (first and last3.count(sig) < 2) or (not first and sig != self._currency_spoken and last4.count(sig) < 3):
            return None
        repeat_s = float(os.environ.get("CURRENCY_REPEAT_S", "6"))
        if sig == self._currency_spoken and time.time() - getattr(self, "_currency_spoken_t", 0.0) < repeat_s:
            return None
        self._currency_spoken = sig
        self._currency_spoken_t = time.time()
        cur = self._modes[MODE_CURRENCY]
        notes = next(v[1] for v in reversed(votes) if v[0] == sig)
        if len(sig) >= 2:
            return cur.total_sentence(notes) + "।"
        name = sig[0]
        value = cur._BN.get(name, name.replace("_", " "))
        mine = [v[1][0] for v in votes if v[0] == sig]
        genuine = sum(1 for n in mine if n.get("auth") in ("likely_genuine", "genuine"))
        checked = name in getattr(config, "BD_AUTH_DENOMINATIONS", ("200_taka", "500_taka", "1000_taka"))
        if genuine * 2 > len(mine):
            return f"{value}। সম্ভবত আসল।"
        if checked:
            return f"{value}। আসল কিনা হাতে যাচাই করুন।"
        return f"{value}।"

    def _ocr_auto_on(self) -> bool:
        import config
        return bool(getattr(config, "OCR_AUTO_CAPTURE", False)) and bool(getattr(self._modes[MODE_OCR], "_v5", None))

    def _ocr_auto_step(self):
        """Text mode auto capture (OCR v5), as main.py does on the glass: read by itself when text is in view and
        the picture is steady and sharp; quiet while speaking or reading."""
        if self._detecting or self._tts.speaking():
            return
        ocr = self._modes[MODE_OCR]
        frame = self._latest_frame()
        if frame is None:
            return
        ev = ocr.auto_feed(frame)
        if not ev:
            return
        if "hint" in ev:
            self._log("Auto: " + ev["hint"])
            self._speak(ev["hint"])
            return
        self._detecting = True
        try:
            self._root.after(0, self._result_var.set, "⏳ Reading (auto)…")
            if ev.get("say"):
                self._speak(ev["say"])
            t0 = time.perf_counter()
            result = ocr.process_frame(ev["fire"])
            ms = (time.perf_counter() - t0) * 1000
            tm = (getattr(ocr, "last_v5", None) or {}).get("timings", {})
            self._root.after(0, self._log, f"Auto read in {ms:.0f} ms ({tm.get('lines', '?')} lines, "
                                           f"finder {tm.get('backend', '?')})")
        finally:
            self._detecting = False
        msg = result or "Nothing detected."
        self._root.after(0, self._show_result, msg)
        self._speak(msg)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _switch_mode(self):
        with self._lock:
            old = self._mode
            new = (self._mode + 1) % len(self._modes)
            self._mode = new
        self._tts.stop()
        self._modes[old].deactivate()
        self._modes[new].activate()
        self._mode_var.set(f"● {MODE_NAMES_EN[new]}")
        if new == MODE_CLAUDE:
            self._result_var.set("Press C or A — online Claude (needs internet + API key)")
        elif new == MODE_OCR:
            self._result_var.set("Press A for offline EasyOCR, or C for online Claude")
        elif new == MODE_CURRENCY:
            self._result_var.set("Hold the note — it says the value and whether it is real or jaal")
        else:
            self._result_var.set("Press A to detect")
        self._speak(MODE_NAMES_EN[new])
        self._log(f"Mode: {MODE_NAMES_EN[new]}")

    def _action(self):
        with self._lock:
            mode = self._mode
        with self._frame_lock:
            frame = self._frame.copy() if self._frame is not None else None
        if frame is None:
            self._speak("Camera not ready.")
            return
        if mode == MODE_OBJECT:
            self._modes[MODE_OBJECT].force_announce_now()
            return
        if self._detecting:
            return

        self._detecting = True
        self._result_var.set("⏳ Detecting…")
        self._set_tts_status("processing…")
        self._log(f"Running {MODE_NAMES_EN[mode]}…")

        def _infer():
            try:
                src = frame
                # text: read the sharpest frame of the next OCR_BEST_FRAME_S seconds, as main.py does
                import config
                secs = float(getattr(config, "OCR_BEST_FRAME_S", 0))
                if mode == MODE_OCR and secs > 0:
                    from frame_pick import best_frame
                    picked, pick = best_frame(frame, self._latest_frame, secs)
                    src = frame if picked is None else picked
                    logger.info("OCR best frame: %s", pick)
                result = self._modes[mode].process_frame(src)
                if mode == MODE_OCR and hasattr(self._modes[mode], "auto_hold"):
                    self._modes[mode].auto_hold()     # this page was read by the key: auto waits for a new view
                msg    = result if result else "Nothing detected."
                if mode == MODE_CURRENCY:
                    self._update_pose_from_currency(frame)
            except Exception as exc:
                msg = f"Error: {exc}"
                logger.error("Inference error: %s", exc)
            finally:
                self._detecting = False
            self._root.after(0, self._show_result, msg)
            self._set_tts_status("speaking…")
            self._speak(msg)
            self._set_tts_status("ready")

        threading.Thread(target=_infer, daemon=True, name="action").start()

    def _claude_action(self):
        """C key — always send the current frame to Claude (online).

        Works from any mode so EasyOCR (A in Text Reading) stays offline.
        """
        with self._frame_lock:
            frame = self._frame.copy() if self._frame is not None else None
        if frame is None:
            self._speak("Camera not ready.")
            return
        if self._detecting:
            return

        with self._lock:
            old = self._mode
            self._mode = MODE_CLAUDE
        if old != MODE_CLAUDE:
            self._tts.stop()
            self._modes[old].deactivate()
            self._modes[MODE_CLAUDE].activate()
            self._mode_var.set(f"● {MODE_NAMES_EN[MODE_CLAUDE]}")
            self._log(f"Mode: {MODE_NAMES_EN[MODE_CLAUDE]}")

        self._detecting = True
        self._result_var.set("⏳ Claude (online)…")
        self._set_tts_status("calling Claude…")
        self._log("Running Online Claude Mode…")

        def _infer():
            try:
                result = self._modes[MODE_CLAUDE].process_frame(frame)
                msg = result if result else "Nothing detected."
            except Exception as exc:
                msg = f"Error: {exc}"
                logger.error("Claude inference error: %s", exc)
            finally:
                self._detecting = False
            self._root.after(0, self._show_result, msg)
            self._set_tts_status("speaking…")
            self._speak(msg)
            self._set_tts_status("ready")

        threading.Thread(target=_infer, daemon=True, name="claude").start()

    def _read_stored(self):
        """R key — speak back the last EasyOCR or Claude result."""
        with self._lock:
            mode = self._mode
        if mode == MODE_OCR:
            result = self._modes[MODE_OCR].read_stored()
        elif mode == MODE_CLAUDE:
            result = self._modes[MODE_CLAUDE].read_stored()
        else:
            return

        if not result:
            return

        self._show_result(result)
        self._log("Reading back captured text…")
        self._set_tts_status("speaking…")
        self._speak(result)
        self._set_tts_status("ready")

    def _show_result(self, text: str):
        self._result_var.set(text)
        self._log(f"↳ {text[:120]}")

    def _vol_up(self):
        self._volume = min(100, self._volume + VOLUME_STEP)
        self._tts.set_volume(self._volume)
        self._vol_bar["value"] = self._volume
        self._vol_lbl["text"]  = f"{self._volume}%"

    def _vol_down(self):
        self._volume = max(0, self._volume - VOLUME_STEP)
        self._tts.set_volume(self._volume)
        self._vol_bar["value"] = self._volume
        self._vol_lbl["text"]  = f"{self._volume}%"

    def _quit(self):
        self._running = False
        self._tts.stop()
        self._modes[self._mode].deactivate()
        for m in self._modes:
            m.cleanup()
        if hasattr(self, "_cap"):
            self._cap.release()
        self._root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 55)
    print("  Smart Glass — Windows Test (Piper / SAPI TTS)")
    print("=" * 55)
    print("  No internet required — Piper if installed, else Windows SAPI")
    print()

    root = tk.Tk()
    SmartGlassWin(root)
    root.mainloop()


if __name__ == "__main__":
    main()
