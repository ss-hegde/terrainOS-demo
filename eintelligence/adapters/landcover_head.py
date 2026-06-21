from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eintelligence.fusion.kernel_base import FusionOutput

@dataclass
class LandCoverHeadConfig:
    in_channels: int            # channels of FusionOutput.fused
    num_classes: int = 6        # reduced WorldCover classes
    decoder_channels: int = 256
    dropout: float = 0.1

class LandCoverSegHead(nn.Module):
    def __init__(self, cfg: LandCoverHeadConfig):
        super().__init__()
        self.cfg = cfg

        self.block1 = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.decoder_channels, 3, padding=1),
            nn.BatchNorm2d(cfg.decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(cfg.decoder_channels, cfg.decoder_channels, 3, padding=1),
            nn.BatchNorm2d(cfg.decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(cfg.dropout)
        self.classifier = nn.Conv2d(cfg.decoder_channels, cfg.num_classes, kernel_size=1)

    def forward(
        self,
        fusion_out: FusionOutput,
        out_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            fusion_out: FusionOutput from fusion kernel
            out_size: optional (H, W) to upsample logits to tile size

        Returns:
            logits: [B, num_classes, H, W]
        """
        x = fusion_out.fused  # [B, C_in, Hf, Wf]
        x = self.block1(x)
        x = self.block2(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        if out_size is not None and logits.shape[-2:] != out_size:
            logits = F.interpolate(
                logits,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits