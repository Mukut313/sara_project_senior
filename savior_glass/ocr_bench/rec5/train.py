"""Train the SARA line recogniser.

    cd savior_glass
    python -m ocr_bench.rec5.train --name r1 --steps 120000

Data per batch: rendered lines (synth.LineRenderer, train fonts only) and, with probability --real, real ICDAR 2019
MLT TRAINING words (train_real_words.split(): the same split as every earlier recogniser; the validation and test
words are never loaded for training).

Checkpoint choice, fixed before the first run: every --eval steps the model reads
  * the 300 real VALIDATION words (150 Bangla, 150 Latin script, rotated ones included), as train_real_words.py
    scored EasyOCR's recogniser (its pretrained and fine-tuned numbers are in results/ocr_bench/real_ft_*.json);
  * the rendered validation crops (train_task7.val_crops, validation fonts).
The kept checkpoint has the lowest  mean(real Bangla CER, real Latin CER, rendered CER).  The test words and the
rendered test set are read once, by eval.py, after training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from .. import bench as B
from .charset import encode, normalize, encodable, decode_greedy
from .model import SaraRec, count_params
from .paths import RUNS
from .synth import H, LineRenderer, to_height

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ real words
class RealCrops:
    """MLT training crops stored once as one flat uint8 file + index, memory-mapped by every data worker
    (a pickled list cost each of 12 workers ~0.5 GB of RAM and minutes to load)."""

    def __init__(self, items=None):
        d = RUNS / "mlt_train"
        if items is not None:
            d.mkdir(parents=True, exist_ok=True)
            flat = np.concatenate([im.reshape(-1) for _, im in items])
            np.save(d / "pixels.npy", flat)
            off, meta = 0, []
            for t, im in items:
                meta.append([off, im.shape[0], im.shape[1], t])
                off += im.size
            (d / "index.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.pix = np.load(d / "pixels.npy", mmap_mode="r")
        self.meta = json.loads((d / "index.json").read_text(encoding="utf-8"))
        # per script: only 2,989 of the 52,867 MLT training words are Bangla; drawing them half and half keeps the
        # real Bangla photos from being 1 % of what the model sees
        self.by_script = {"bn": [], "lat": []}
        for i, m in enumerate(self.meta):
            self.by_script["bn" if any("ঀ" <= c <= "৿" for c in m[3]) else "lat"].append(i)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        off, h, w, t = self.meta[i]
        return t, np.array(self.pix[off:off + h * w]).reshape(h, w)

    def choice(self, rng):
        pool = self.by_script["bn" if rng.random() < 0.5 else "lat"] or list(range(len(self.meta)))
        return self[pool[rng.randrange(len(pool))]]


def real_cache():
    """MLT training crops (RealCrops); built from the parquet files the first time (takes minutes)."""
    if (RUNS / "mlt_train" / "index.json").exists():
        return RealCrops()
    p = RUNS / "mlt_train_crops.pkl"
    if p.exists():
        items = pickle.loads(p.read_bytes())
    else:
        from ..train_real_words import load_crops, split
        tr, _, _ = split()
        items = [(normalize(t), im) for t, im in load_crops(tr, None, upright=True)]
    items = [(normalize(t), im) for t, im in items if t and encodable(normalize(t))]
    rc = RealCrops(items)
    p.unlink(missing_ok=True)
    return rc


def real_val():
    p = RUNS / "mlt_val_crops.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    from ..train_real_words import load_crops, split
    _, va, _ = split()
    out = []
    for lang, key in (("bn", "bn_IN"), ("en", "en_XX")):
        for t, im in load_crops({key: va[key]}, None, upright=False):
            out.append((lang, t, im))
    p.write_bytes(pickle.dumps(out, protocol=4))
    return out


def augment_real(im, rng):
    from ..train_real_words_v2 import augment
    im = augment(im, rng)
    if rng.random() < 0.3:                                            # detector-like extra margin
        p = int(im.shape[0] * rng.uniform(0.05, 0.3))
        im = cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_REPLICATE)
    return im


# ------------------------------------------------------------------ batches
def make_batch(samples):
    """samples [(text, grey H x w)] -> x (N,1,H,W), targets, target_lengths, input_lengths, texts."""
    ws = [im.shape[1] for _, im in samples]
    W = max(64, int(math.ceil(max(ws) / 64) * 64))
    x = np.zeros((len(samples), 1, H, W), np.float32)
    for i, (_, im) in enumerate(samples):
        w = im.shape[1]
        x[i, 0, :, :w] = im
        if w < W:
            x[i, 0, :, w:] = im[:, -1:]
    x = x / 127.5 - 1.0
    tg = [encode(t) for t, _ in samples]
    il = [max(1, w // 4) for w in ws]
    return (torch.from_numpy(x), torch.tensor([c for t in tg for c in t], dtype=torch.long),
            torch.tensor([len(t) for t in tg], dtype=torch.long), torch.tensor(il, dtype=torch.long),
            [t for t, _ in samples])


class Batches(IterableDataset):
    def __init__(self, bs, real_frac, seed):
        self.bs, self.real_frac, self.seed = bs, real_frac, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        wid = wi.id if wi else 0
        rng = random.Random(self.seed * 1000 + wid + int(time.time()) % 997)
        ren = LineRenderer(rng.randrange(1 << 30))
        real = real_cache() if self.real_frac > 0 else None
        # Fixed shapes: widths are padded up to a multiple of 64 and every width has one batch size (bs * 320
        # padded columns at most). The GPU picks a convolution plan once per shape (~0.6 s); with a new width every
        # batch that cost was paid on most steps (7 s/step instead of 0.1 s). Lines wait in their width's bucket
        # until it is full; none are thrown away.
        buckets = {}
        while True:
            for s in self._lines(rng, ren, real):
                W = max(64, -(-s[1].shape[1] // 64) * 64)
                b = buckets.setdefault(W, [])
                b.append(s)
                if len(b) >= max(8, min(self.bs, (self.bs * 320) // W)):
                    buckets[W] = []
                    yield make_batch(b)

    def _lines(self, rng, ren, real):
        while True:
            if True:
                if real is not None and rng.random() < self.real_frac:
                    t, im = real.choice(rng)
                    im = to_height(augment_real(im, rng), rng)
                    if im.shape[1] > 800:
                        continue
                    s = (t, im)
                else:
                    try:
                        s = ren.sample()
                    except Exception:  # noqa: BLE001
                        s = None
                    if s is None:
                        continue
                if len(encode(s[0])) * 1.0 > s[1].shape[1] // 4 - 1:        # not enough CTC steps for this label
                    continue
                yield s


# ------------------------------------------------------------------ evaluation
@torch.inference_mode()
def read(model, imgs, bs=64):
    """imgs: grey crops of any height -> [(text, conf)]."""
    model.eval()
    lines = [to_height(im) if im.shape[0] != H else im for im in imgs]
    order = sorted(range(len(lines)), key=lambda i: lines[i].shape[1])
    out = [None] * len(lines)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        x, _, _, il, _ = make_batch([("", lines[i]) for i in idx])
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            p = model(x.to(DEV)).float().softmax(-1)
        conf, ids = p.max(-1)
        for j, i in enumerate(idx):
            n = int(il[j])
            out[i] = decode_greedy(ids[j, :n].tolist(), conf[j, :n].tolist())
    model.train()
    return out


def evaluate(model, rv, rend):
    hyp = read(model, [im for _, _, im in rv])
    res = {}
    for lang in ("bn", "en"):
        c = [B.cer(t, h[0]) for (l, t, _), h in zip(rv, hyp) if l == lang]
        res["real_" + lang] = float(np.mean(c))
    hyp2 = read(model, [im for _, im in rend])
    res["rendered"] = float(np.mean([B.cer(t, h[0]) for (t, _), h in zip(rend, hyp2)]))
    res["real_mean"] = (res["real_bn"] + res["real_en"]) / 2
    res["score"] = (res["real_bn"] + res["real_en"] + res["rendered"]) / 3
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="r1")
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--real", type=float, default=0.2)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)

    from ..train_task7 import val_crops
    rv = real_val()
    rend = val_crops({"bn": B.BN_FONT["val"], "en": B.EN_FONT["val"]})
    real_cache()                                                        # build the cache before workers start

    model = SaraRec(a.width).to(DEV)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=DEV)["model"])
    print(f"{count_params(model) / 1e6:.2f} M parameters; real val {len(rv)} crops; rendered val {len(rend)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.04,
                                                anneal_strategy="cos", div_factor=20, final_div_factor=200)
    scaler = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    dl = DataLoader(Batches(a.bs, a.real, a.seed), batch_size=None, num_workers=a.workers,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    log, best, t0, run_loss = [], None, time.time(), []
    step = 0
    it = iter(dl)
    try:
        _loop(a, it, model, opt, sched, scaler, rv, rend, out, log, t0)
    finally:
        try:
            it._shutdown_workers()                 # no orphaned data workers after the run
        except Exception:  # noqa: BLE001
            pass


def _loop(a, it, model, opt, sched, scaler, rv, rend, out, log, t0):
    best, run_loss, step = None, [], 0
    for x, tg, tl, il, _ in it:
        x = x.to(DEV, non_blocking=True)
        tg, tl, il = tg.to(DEV), tl.to(DEV), il.to(DEV)
        with torch.autocast(DEV.type, enabled=DEV.type == "cuda"):
            y = model(x)
        lp = F.log_softmax(y.float(), -1).permute(1, 0, 2)
        loss = F.ctc_loss(lp, tg, il, tl, blank=0, reduction="mean", zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss.append(loss.item())
        step += 1
        if step % 200 == 0:
            print(f"step {step} loss {np.mean(run_loss):.3f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.3f}s/step", flush=True)
            run_loss = []
        if step % a.eval == 0 or step == a.steps:
            r = evaluate(model, rv, rend)
            r["step"] = step
            r["minutes"] = round((time.time() - t0) / 60, 1)
            log.append(r)
            print("EVAL", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
            torch.save({"model": model.state_dict(), "width": a.width, "step": step}, out / "last.pt")
            if best is None or r["score"] < best["score"]:
                best = r
                torch.save({"model": model.state_dict(), "width": a.width, "step": step}, out / "best.pt")
            (out / "log.json").write_text(json.dumps({"args": vars(a), "best": best, "log": log}, indent=1),
                                          encoding="utf-8")
        if step >= a.steps:
            break
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
