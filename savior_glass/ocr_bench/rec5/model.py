"""SARA line recogniser network.

Input  : (N, 1, 48, W) grey line, values (x / 127.5 - 1), W any multiple of 4.
Output : (N, W / 4, C) class scores, C = len(CHARS) + 1 (index 0 = CTC blank).

Convolution only (inverted-residual blocks as in MobileNetV2, then depthwise 1 x 7 context blocks along the line):
every layer is a plain conv / batch-norm / ReLU, so it exports to ONNX, quantises to 8 bits for the Pi's CPU, and
is the kind of network Hailo's Dataflow Compiler accepts (at a fixed width such as 48 x 320). One output step covers
4 input pixels: a Bangla conjunct with five code points in about 30 pixels still has enough steps for CTC.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .charset import CHARS


def _cbr(cin, cout, k=3, s=1, groups=1):
    pad = (k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, pad, groups=groups, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True))


class IR(nn.Module):
    """Inverted residual: 1x1 expand, 3x3 depthwise (stride s), 1x1 project."""

    def __init__(self, cin, cout, s=1, t=3):
        super().__init__()
        mid = cin * t
        self.res = s == 1 and cin == cout
        self.body = nn.Sequential(
            _cbr(cin, mid, 1),
            _cbr(mid, mid, 3, s, groups=mid),
            nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.body(x)
        return x + y if self.res else y


class Ctx(nn.Module):
    """Context along the line: depthwise 1 x k, then a 1x1 feed-forward, both residual."""

    def __init__(self, c, k=7, t=2):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(c, c, (1, k), 1, (0, k // 2), groups=c, bias=False), nn.BatchNorm2d(c),
                                nn.ReLU(inplace=True))
        self.ff = nn.Sequential(nn.Conv2d(c, c * t, 1, bias=False), nn.BatchNorm2d(c * t), nn.ReLU(inplace=True),
                                nn.Conv2d(c * t, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        x = x + self.dw(x)
        return x + self.ff(x)


class SaraRec(nn.Module):
    def __init__(self, width: float = 1.0, n_ctx: int = 4):
        super().__init__()
        c = lambda v: max(8, int(v * width + 0.5) // 8 * 8)                     # noqa: E731
        self.stem = _cbr(1, c(32), 3, 2)                                         # 24 x W/2
        self.stages = nn.Sequential(
            IR(c(32), c(48), 1, 2), IR(c(48), c(64), (2, 1)),                    # 12 x W/2
            IR(c(64), c(96), (2, 2)), IR(c(96), c(96)), IR(c(96), c(96)),        # 6 x W/4
            IR(c(96), c(160), (2, 1)), IR(c(160), c(160)), IR(c(160), c(160)), IR(c(160), c(160)),   # 3 x W/4
            IR(c(160), c(256), 1),
        )
        self.squash = nn.Sequential(nn.Conv2d(c(256), c(256), (3, 1), 1, 0, bias=False), nn.BatchNorm2d(c(256)),
                                    nn.ReLU(inplace=True))                       # 1 x W/4
        self.ctx = nn.Sequential(*[Ctx(c(256)) for _ in range(n_ctx)])
        self.drop = nn.Dropout(0.1)
        self.head = nn.Conv2d(c(256), len(CHARS) + 1, 1)

    def forward(self, x):
        x = self.squash(self.stages(self.stem(x)))
        x = self.head(self.drop(self.ctx(x)))                                    # N, C, 1, T
        return x.squeeze(2).permute(0, 2, 1)                                     # N, T, C


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = SaraRec()
    y = m(torch.zeros(2, 1, 48, 320))
    print(y.shape, f"{count_params(m) / 1e6:.2f} M parameters")
