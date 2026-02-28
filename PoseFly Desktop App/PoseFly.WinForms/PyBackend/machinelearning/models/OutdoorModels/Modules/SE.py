import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-and-Excitation (channel attention).

    Robust to Ultralytics width scaling:
    - The YAML 'c' arg is NOT reliably scaled for custom modules.
    - So we ignore it for building and infer channels from x at runtime.
    """

    def __init__(self, c=None, r=16, **kwargs):
        super().__init__()

        # keep your existing behavior: SE(c, r)
        if isinstance(r, bool) or r is None:
            r = 16
        try:
            r = int(r)
        except Exception:
            r = 16
        self.r = max(1, r)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Lazy-built layers (built on first forward from x.shape[1])
        self.c = None
        self.fc1 = None
        self.fc2 = None

        # NOTE: we intentionally do NOT build from 'c' here.
        # 'c' comes from YAML and is unscaled, which causes channel mismatch on YOLO26n/s/...

    def _build(self, c_in: int, device=None, dtype=None):
        c_in = max(1, int(c_in))
        c_mid = max(8, c_in // self.r)

        fc1 = nn.Conv2d(c_in, c_mid, 1, bias=True)
        fc2 = nn.Conv2d(c_mid, c_in, 1, bias=True)

        if device is not None:
            fc1 = fc1.to(device=device, dtype=dtype)
            fc2 = fc2.to(device=device, dtype=dtype)

        self.fc1 = fc1
        self.fc2 = fc2
        self.c = c_in

    def forward(self, x):
        c_in = int(x.shape[1])

        # Build/rebuild BEFORE calling fc1/fc2
        # - first run: fc1/fc2 are None
        # - scale change or YAML mismatch: rebuild when channels differ
        if (self.fc1 is None) or (self.fc2 is None) or (self.c != c_in) or (self.fc1.in_channels != c_in):
            self._build(c_in, device=x.device, dtype=x.dtype)

        w = self.pool(x)
        w = F.silu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w