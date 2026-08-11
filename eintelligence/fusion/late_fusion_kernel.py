from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_base import BaseFusionKernel, FusionBatch, FusionOutput


class LateFusionKernel(BaseFusionKernel):
    """
    Feature-level late fusion kernel.

    - Accepts a backbone that returns per-modality feature maps, e.g. {"s1": ..., "s2": ...}
    - Selects the requested modalities
    - Applies modality masks
    - Concatenates features and projects them to fused_dim with a 1x1 conv
    """

    def __init__(
        self,
        backbone: nn.Module,
        fused_dim: int,
        use_modalities: Optional[list[str]] = None,
        debug_checks: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.fused_dim = fused_dim
        self.use_modalities = use_modalities
        self.debug_checks = debug_checks

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

        in_ch = sum(feats[k].shape[1] for k in keys)
        device = next(iter(feats.values())).device
        self.proj = nn.Conv2d(in_ch, self.fused_dim, kernel_size=1).to(device)

    def _check_finite(self, name: str, x: Optional[torch.Tensor]) -> None:
        if not self.debug_checks or x is None:
            return
        if not torch.isfinite(x).all():
            raise RuntimeError(f"Non-finite tensor detected in {name}: shape={tuple(x.shape)}")

    def forward(self, batch: FusionBatch) -> FusionOutput:
        s1 = batch.imagery.get("s1", None)
        s2 = batch.imagery.get("s2", None)

        self._check_finite("input.s1", s1)
        self._check_finite("input.s2", s2)

        feats: Dict[str, torch.Tensor] = self.backbone(s1, s2)

        if self.use_modalities is None:
            keys = list(feats.keys())
        else:
            keys = [k for k in self.use_modalities if k in feats]

        if not keys:
            raise ValueError("LateFusionKernel: No valid modalities found for fusion.")

        self._build_proj_if_needed(feats)

        feat_list = []
        for k in keys:
            f = feats[k]
            self._check_finite(f"backbone.{k}", f)

            mask = batch.masks.get(k, None)
            if mask is not None:
                if mask.shape[-2:] != f.shape[-2:]:
                    mask = F.interpolate(mask.float(), size=f.shape[-2:], mode="nearest")
                else:
                    mask = mask.float()

                if mask.shape[1] == 1:
                    mask = mask.expand(-1, f.shape[1], -1, -1)

                f = f * mask

            self._check_finite(f"post_mask.{k}", f)
            feat_list.append(f)

        fused_cat = torch.cat(feat_list, dim=1)
        self._check_finite("fused_cat", fused_cat)

        if self.proj.weight.device != fused_cat.device:
            self.proj = self.proj.to(fused_cat.device)

        fused = self.proj(fused_cat)
        self._check_finite("fused", fused)

        # per_modality must mirror `keys` (the modalities actually selected for
        # fusion), not the raw backbone output dict `feats` — the backbone may
        # have computed features for modalities the caller excluded via
        # use_modalities (e.g. the caller still passed both s1 and s2 in
        # batch.imagery), and reporting those would misrepresent which sensors
        # this kernel is actually fusing.
        per_modality = {k: feats[k] for k in keys}

        return FusionOutput(
            fused=fused,
            per_modality=per_modality,
            uncertainty=None,
            aux={},
        )