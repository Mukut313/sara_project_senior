"""SARA line recogniser (OCR v5): a small conv + CTC text-line reader for Bangla + English, trained on rendered lines
that imitate the glass camera and on real ICDAR 2019 MLT words. Runs as ONNX on the Pi's CPU; conv-only so it can also
be compiled for the Hailo-8 later. See README.md in this folder."""
