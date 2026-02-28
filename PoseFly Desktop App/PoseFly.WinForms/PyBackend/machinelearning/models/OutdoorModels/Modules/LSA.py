import torch
import torch.nn as nn

class LSA(nn.Module):
    """
    Lightweight Self-Attention (channel-agnostic but NOT lazy)
    Requires channel count `c` at init time.
    """

    def __init__(self, c: int, reduction: int = 8):
        super().__init__()
        self.c = int(c)
        self.reduction = int(reduction)

        self.qkv  = nn.Conv2d(self.c, self.c * 3, 1, bias=False)
        self.dw   = nn.Conv2d(self.c * 3, self.c * 3, 3, padding=1, groups=self.c * 3, bias=False)
        self.proj = nn.Conv2d(self.c, self.c, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.c)
        self.act  = nn.SiLU()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        qkv = self.dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        prod = q * k
        attn_mean = prod.mean(dim=1, keepdim=True)
        attn_max, _ = prod.max(dim=1, keepdim=True)
        attn = self.spatial(torch.cat([attn_mean, attn_max], dim=1))

        out = v * attn
        out = self.proj(out)
        return self.act(self.norm(out + x))