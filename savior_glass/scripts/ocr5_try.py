"""Try OCR v5 on the laptop: on photos, or live on the webcam with auto capture.

    python scripts/ocr5_try.py photo1.jpg photo2.png ...     read each photo, print the text and timings
    python scripts/ocr5_try.py --cam [index]                  live: auto capture reads by itself; SPACE = read now,
                                                              R = say the last text again,
                                                              S = save the frame, Q = quit
Every reading is saved under logs/ocr5_try/<time>/ (original, boxes, result.json) so a wrong reading can be looked at
and, with the typed reference text, added to a test set.
Speech: the glass's offline Piper voices (Bangla + English); --mute = text only.
Fully offline. Uses the same code as the glass (ocr_v5.OcrV5, doc_scanner.AutoCapture).
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)
os.environ.setdefault("OCR_PIPELINE", "v5")
os.environ.setdefault("OCR_DEBUG_DIR", os.path.join(HERE, "logs", "ocr5_try"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


class Speaker:
    """Offline speech with the glass's own Piper voices (models/piper: Bangla bn_BD-google-medium, English
    en_US-amy-low; get them once with scripts/get_speech_models.py). Mixed lines are split by script, each part
    spoken with its own voice. Plays through winsound on Windows, aplay elsewhere. --mute turns it off."""

    def __init__(self, enabled=True):
        import queue
        import threading
        import config
        self.enabled = enabled
        self.models = {"bn": config.PIPER_BN_MODEL, "en": config.PIPER_EN_MODEL}
        self.voices = {}
        self.q = queue.Queue()
        self.busy = False
        if enabled:
            threading.Thread(target=self._run, daemon=True).start()

    def speak(self, text):
        print("  [speak]", text)
        if self.enabled and text and text.strip():
            self.q.put(text)

    def speaking(self):
        return self.busy or not self.q.empty()

    @staticmethod
    def _parts(text):
        import re
        parts, cur, lang = [], "", None
        for tok in text.split():
            bn = sum("ঀ" <= c <= "৿" for c in tok)
            lat = sum(c.isascii() and c.isalpha() for c in tok)
            t = "bn" if bn >= lat and bn else ("en" if lat else (lang or "bn"))
            if lang and t != lang:
                parts.append((lang, cur))
                cur = ""
            lang, cur = t, (cur + " " + tok).strip()
        if cur:
            parts.append((lang, cur))
        return parts

    def _voice(self, lang):
        if lang not in self.voices:
            from piper import PiperVoice
            self.voices[lang] = PiperVoice.load(self.models[lang]) if os.path.exists(self.models[lang]) else None
            if self.voices[lang] is None:
                print(f"  (no {lang} voice: run scripts/get_speech_models.py)")
        return self.voices[lang]

    def _run(self):
        import io
        import wave
        while True:
            text = self.q.get()
            self.busy = True
            try:
                for lang, part in self._parts(text):
                    v = self._voice(lang)
                    if v is None:
                        continue
                    if hasattr(v, "synthesize_stream_raw"):
                        pcm = b"".join(v.synthesize_stream_raw(part))
                    else:
                        pcm = b"".join(c.audio_int16_bytes for c in v.synthesize(part))
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(int(v.config.sample_rate))
                        w.writeframes(pcm)
                    if sys.platform == "win32":
                        import winsound
                        winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)
                    else:
                        import subprocess
                        subprocess.run(["aplay", "-q", "-"], input=buf.getvalue())
            except Exception as exc:  # noqa: BLE001
                print("  (speech failed:", exc, ")")
            finally:
                self.busy = False


def _print(res):
    tm = res["timings"]
    print(f"--- {tm['lines']} lines | total {tm['total_ms']:.0f} ms (page {tm['page_ms']:.0f}, find {tm['find_ms']:.0f}, "
          f"read {tm['read_ms']:.0f}) | finder {tm['backend']} | page flattened: {res['page'] is not None}")
    for q, t, c in res["boxes"]:
        print(f"  {c:.2f}  {t}")
    print("TEXT:\n" + (res["text"] or "(nothing read)"))


def photos(paths, speaker):
    from modes.ocr_mode import OCRMode
    m = OCRMode()
    m.speak_line = speaker.speak
    m.activate()
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("cannot read", p)
            continue
        print("\n==>", p, img.shape[1], "x", img.shape[0])
        spoken = m.process_frame(img)
        _print(m.last_v5)
        speaker.speak(spoken)
        while speaker.speaking():
            time.sleep(0.1)


def live(speaker, index=0):
    from modes.ocr_mode import OCRMode
    m = OCRMode()
    m.speak_line = speaker.speak
    m.activate()
    speaker.speak("লেখা পড়ার জন্য প্রস্তুত")                       # ready to read text
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.environ.get("CAMERA_WIDTH", "1280")))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.environ.get("CAMERA_HEIGHT", "720")))
    print("camera", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        ev = None
        if time.time() - last_t > 0.12 and not speaker.speaking():   # quiet while speaking, as on the glass
            last_t = time.time()
            ev = m.auto_feed(frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            p = os.path.join(HERE, "logs", f"frame_{time.strftime('%H%M%S')}.jpg")
            cv2.imwrite(p, frame)
            print("saved", p)
        if key == ord(" "):
            ev = {"fire": frame}
        if key == ord("r"):
            speaker.speak(m.read_stored())                              # READ button: say the last text again
        if ev and "hint" in ev:
            speaker.speak(ev["hint"])
        if ev and "fire" in ev:
            if ev.get("say"):
                speaker.speak(ev["say"])
            speaker.speak("পড়ছি")                                       # "reading"
            spoken = m.process_frame(ev["fire"])
            _print(m.last_v5)
            if m.last_v5.get("image") is not None:                      # the scanned page in colour, as a scanner app shows it
                import doc_scanner
                sc = doc_scanner.color_scan(m.last_v5["image"], m.last_v5.get("page_margin", 0))
                cv2.imshow("Scan", cv2.resize(sc, (720, max(1, int(720 * sc.shape[0] / sc.shape[1])))))
            speaker.speak(spoken)
            m.auto_hold()
        vis = frame.copy()
        for b in getattr(m, "last_boxes", []) or []:
            if time.time() - m.last_boxes_t < 6:
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 220, 0) if b[5] else (0, 0, 255), 2)
        cv2.putText(vis, "AUTO: " + m.auto_state(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("OCR v5 (SPACE read, S save, Q quit)", cv2.resize(vis, (960, int(960 * vis.shape[0] / vis.shape[1]))))
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    a = sys.argv[1:]
    mute = "--mute" in a
    a = [x for x in a if x != "--mute"]
    sp = Speaker(not mute)
    if not a:
        print(__doc__)
    elif a[0] == "--cam":
        live(sp, int(a[1]) if len(a) > 1 else 0)
    else:
        photos(a, sp)
