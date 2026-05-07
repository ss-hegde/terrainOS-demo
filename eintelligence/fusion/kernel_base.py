from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

@dataclass
class FusionBatch:
    """
    Container passed into the fusion kernel.

    imagery: dict of modality name -> BCHW tensor of image features
    masks: dict of modality name -> BCHW bool or (0,1)
    meta: arbitrary per tile metadata, e.g. geospatial coordinates, timestamps, etc.
    """

    imagery: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor]
    meta: Dict[str, Any]

@dataclass
class FusionOutput:
    """
    Output of the fusion kernel.

    fused: shared fused embeddings used by the downstream adapter heads
    per_modality: optional modality-specific feature maps
    uncertainty: optional per-pixel or per-tile uncertainty estimates
    aux: extra artifacts, e.g. attention maps, XAI Output, etc.
    """

    fused: torch.Tensor
    per_modality: Dict[str, torch.Tensor]
    uncertainty: Optional[torch.Tensor]
    aux: Dict[str, Any]

class BaseFusionKernel(nn.Module):
    """
    Abstract base class for all fusion kernels.
    """

    def forward(self, batch: FusionBatch) -> FusionOutput:
        """
        Forward pass of the fusion kernel.

        Args:
            batch: FusionBatch containing imagery, masks, and metadata.

        Returns:
            FusionOutput containing fused embeddings, per-modality features, uncertainty estimates, and auxiliary artifacts.
        """
        raise NotImplementedError("Fusion kernels must implement the forward method.")
    
class IdentityFusionKernel(BaseFusionKernel):
    """
    Minimal fusion kernel that just forwards one backbone's features as the fused output.

    This is used to introduce the fusion kernel abstraction without changing the behavior of the existing system, and can be used as a sanity check.
    """

    def __init__(self, encoder: nn.Module, modality_key: str):
        super().__init__()
        self.encoder = encoder
        self.modality_key = modality_key
    
    def forward(self, batch: FusionBatch) -> FusionOutput:
        # Expect chosen modality to be present in the batch
        x = batch.imagery[self.modality_key] # BCHW
        feats = self.encoder(x)              # Assume BCHW feature map

        per_modality = {self.modality_key: feats}
        fused = feats

        return FusionOutput(
            fused=fused,
            per_modality=per_modality,
            uncertainty=None,
            aux={}
        )

class MultisensorIdentityFusionKernel(BaseFusionKernel):
    """
    Wraps MultiSensorSSL4EOLiteBackbone and exposes its per-modality features
    via FusionOutput.per_modality. For now, .fused can be a simple choice
    (e.g., concatenation or one branch) and is not used by this adapter yet.

    Expects a backbone(s1, s2) -> Dict[str, Tensor] (e.g. MultiSensorSSL4EOLiteBackbone).
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: FusionBatch) -> FusionOutput:
        s1 = batch.imagery.get("s1", None)
        s2 = batch.imagery.get("s2", None)

        feats = self.backbone(s1, s2)   # {"s1": ..., "s2": ...}

        if "s2" in feats:
            fused = feats["s2"]
        else:
            fused = next(iter(feats.values()))

        return FusionOutput(
            fused=fused,
            per_modality=feats,
            uncertainty=None,
            aux={},
        )