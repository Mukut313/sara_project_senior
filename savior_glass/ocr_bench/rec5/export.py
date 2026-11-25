"""Export a trained recogniser for the glass: ONNX (any CPU), an 8-bit copy for the Pi, the character list.

    python -m ocr_bench.rec5.export <run name> [out dir]       (default out: models/ocr5_rec)

The 8-bit copy (static, per-channel, QDQ; calibrated on 512 rendered + real TRAINING lines) is written only when it
reads the validation sets within 0.5 CER points of the full-precision model; otherwise the glass uses rec.onnx.
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from .. import bench as B
from .charset import CHARS
from .model import SaraRec, count_params
from .paths import GLASS, RUNS
from .synth import H, LineRenderer, to_height


def _calib(n=256):
    from .train import augment_real, real_cache
    rng = random.Random(7)
    ren = LineRenderer(7)
    real = real_cache()
    out = []
    while len(out) < n:
        if rng.random() < 0.3:
            t, im = real.choice(rng)
            out.append(to_height(augment_real(im, rng), rng))
        else:
            s = ren.sample()
            if s:
                out.append(s[1])
    return out


class _Reader:
    def __init__(self, imgs):
        self.it = iter(imgs)

    def get_next(self):
        im = next(self.it, None)
        if im is None:
            return None
        w = 320                                    # one width: the calibrator stacks every activation it sees
        im = im[:, :w]
        x = np.empty((1, 1, H, w), np.float32)
        x[0, 0, :, :im.shape[1]] = im
        x[0, 0, :, im.shape[1]:] = im[:, -1:]
        return {"x": x / 127.5 - 1.0}


def _onnx_cer(path, rv, rend):
    import sys as _s
    _s.path.insert(0, str(GLASS))
    from sara_rec import SaraRecognizer
    r = SaraRecognizer(str(Path(path).parent), int8=Path(path).name.endswith("int8.onnx"))
    hyp = r.read([im for _, _, im in rv])
    res = {lang: float(np.mean([B.cer(t, h[0]) for (l, t, _), h in zip(rv, hyp) if l == lang])) for lang in ("bn", "en")}
    hyp2 = r.read([im for _, im in rend])
    res["rendered"] = float(np.mean([B.cer(t, h[0]) for (t, _), h in zip(rend, hyp2)]))
    res["score"] = (res["bn"] + res["en"] + res["rendered"]) / 3
    return res


def main():
    name = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else GLASS / "models" / "ocr5_rec"
    ck = torch.load(RUNS / name / "best.pt", map_location="cpu")
    m = SaraRec(ck.get("width", 1.0)).eval()
    m.load_state_dict(ck["model"])
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "rec.onnx"
    torch.onnx.export(m, torch.zeros(1, 1, H, 320), str(onnx_path), input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "n", 3: "w"}, "y": {0: "n", 1: "t"}}, opset_version=17, dynamo=False)
    (out / "charset.txt").write_text("\n".join("<space>" if c == " " else c for c in CHARS) + "\n", encoding="utf-8")

    from .train import real_val
    from ..train_task7 import val_crops
    rv = real_val()
    rend = val_crops({"bn": B.BN_FONT["val"], "en": B.EN_FONT["val"]})
    fp = _onnx_cer(onnx_path, rv, rend)
    print("fp32", fp)

    info = {"run": name, "step": ck.get("step"), "params_M": round(count_params(m) / 1e6, 3), "input": f"1x{H}xW grey",
            "validation_fp32": fp}
    q_path = out / "rec_int8.onnx"
    try:
        if __import__("os").environ.get("SKIP_INT8") == "1":
            raise RuntimeError("skipped (SKIP_INT8=1)")
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = out / "rec_pre.onnx"
        quant_pre_process(str(onnx_path), str(pre))
        quantize_static(str(pre), str(q_path), _Reader(_calib()), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.Percentile, extra_options={"CalibPercentile": 99.99})
        pre.unlink(missing_ok=True)
        q = _onnx_cer(q_path, rv, rend)
        print("int8", q)
        info["validation_int8"] = q
        if q["score"] - fp["score"] > 0.005:
            print("8-bit copy reads worse than the rule allows; not kept")
            q_path.unlink(missing_ok=True)
            info["int8_kept"] = False
        else:
            info["int8_kept"] = True
    except Exception as exc:  # noqa: BLE001
        print("8-bit export failed:", exc)
        info["int8_kept"] = False
    log = RUNS / name / "log.json"
    if log.exists():
        info["training"] = json.loads(log.read_text(encoding="utf-8")).get("args")
    (out / "model.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print("written", out)


if __name__ == "__main__":
    main()
