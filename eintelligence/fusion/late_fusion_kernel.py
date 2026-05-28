from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_base import BaseFusionKernel, FusionBatch, FusionOutput

class LateFusionKernel(BaseFusionKernel):
    """
    Feature-level late fusion kernel.

    - Accepts a backbone that returns per-modality feature maps (e.g. {"s1": ..., "s2": ...}).
    - Optionally supports extra encoders for additional modalities (DEM, LST, etc.) later.
    - Concatenates modality features (after masking), projects to fused_dim via 1x1 conv.
    """
     
    def __init__(
               self,
                backbone: nn.Module,
                fused_dim: int,
                use_modalities: Optional[list[str]] = None,
     ):
          
          super().__init__()
          self.backbone = backbone
          self.fused_dim = fused_dim
          self.use_modalities = use_modalities # e.g. ["s1", "s2"], or None

          self.proj: Optional[nn.Conv2d] = None 

    def _build_proj_if_needed(self, feats: Dict[str, torch.Tensor]) -> None:
        if self.proj is not None:
            return
        
        if self.use_modalities is None:
            keys = list(feats.keys())
        else:
            keys = [k for k in self.use_modalities if k in feats]

        if not keys:
            raise ValueError("No valid modalities found for fusion.")
        
        # Assume all modality features have the same spatial dimensions and channel counts for simplicity
        in_ch = sum(feats[k].shape[1] for k in keys)
        self.proj = nn.Conv2d(in_ch, self.fused_dim, kernel_size=1)

    def forward(self, batch: FusionBatch) -> FusionOutput:
        # 1) Call the underlying backbone to get per-modality features
        s1 = batch.imagery.get("s1", None)
        s2 = batch.imagery.get("s2", None)
        feats: Dict[str, torch.Tensor] = self.backbone(s1, s2) 

        # 2) Decide which modalities to use for fusion
        if self.use_modalities is None:
            keys = list(feats.keys()) 
        else:
            keys = [k for k in self.use_modalities if k in feats]

        if not keys:
            raise ValueError("LateFusionKernel: No valid modalities found for fusion.")
        
        # 3) Build the projection layer if we haven't already
        self._build_proj_if_needed(feats)

        # 4) Mask and concatenate features from the selected modalities
        feat_list = []
        for k in keys:
            f = feats[k] # B C_k H W
            mask = batch.masks.get(k, None) # B 1 H W or None
            if mask is not None:
                # Expand mask from (B, 1, H, W) to (B, C_k, H, W) to match feature channels
                if mask.shape[1] == 1:
                    mask = mask.expand(-1, f.shape[1], -1, -1)
                f = f * mask
            feat_list.append(f)

        fused_cat = torch.cat(feat_list, dim=1) # B (sum C_k) H W

        # 5) Project concatenated features to fused_dim
        fused = self.proj(fused_cat) # B fused_dim H W

        return FusionOutput(
            fused=fused,
            per_modality=feats,
            uncertainty=None,
            aux={},
        )