from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class _DecoderSmall(nn.Module):
    def __init__(self, c_in: int, c_mid: int = 256):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(c_in, 256, 2, stride=2)
        self.c1  = nn.Conv2d(256, c_mid, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(c_mid, 128, 2, stride=2)
        self.c2  = nn.Conv2d(128, 64, 3, padding=1)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.c3  = nn.Conv2d(32, 32, 3, padding=1)
        self.up4 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.head= nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x = F.relu(self.c1(self.up1(x)))
        x = F.relu(self.c2(self.up2(x)))
        x = F.relu(self.c3(self.up3(x)))
        x = self.up4(x)
        return self.head(x)  # logits
    
class FloodChangeAdapter(nn.Module):
    """
    S1-only: features = backbone(x_t0), backbone(x_t1)
    Change encoding: abs-diff + concatenation
    """
    def __init__(self, c_backbone: int = 512, c_align: int = 256):
        super().__init__()
        self.align = nn.Conv2d(c_backbone, c_align, 1)
        self.dec   = _DecoderSmall(c_in=c_align*3)  # [t0,t1,|t1-t0|]

    def forward(self, x_t0: torch.Tensor, x_t1: torch.Tensor, backbone) -> dict:
        with torch.no_grad():
            f0 = backbone(x_t0)   # (B,512,H/32,W/32)
            f1 = backbone(x_t1)
        f0 = self.align(f0); f1 = self.align(f1)
        fd = torch.abs(f1 - f0)
        feats = torch.cat([f0, f1, fd], dim=1)
        logits = self.dec(feats)  # (B,1,H,W)
        return {"logits": logits}