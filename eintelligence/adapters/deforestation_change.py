from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# Kernel building blocks
from eintelligence.fusion.kernel_base import BaseFusionKernel, FusionBatch, FusionOutput

#----------- building blocks -----------

class ConvAlign(nn.Module):
    """
    Convolutional feature alignment module from SSL4EO-Multisensor.
    """
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x): return F.relu(self.bn(self.conv(x)))

class CrossAttentionFuse(nn.Module):
    """
    Attends 'a' over 'b' and returns fused map (same C,H,W).
    """
    def __init__(self, c: int):
        super().__init__()
        self.query_conv = nn.Conv2d(c, c, 1, bias=False)
        self.key_conv   = nn.Conv2d(c, c, 1, bias=False)
        self.value_conv = nn.Conv2d(c, c, 1, bias=False)
        self.scale = c ** -0.5
    def forward(self, a, b):
        Q,K,V = self.query_conv(a), self.key_conv(b), self.value_conv(b)
        attn = torch.softmax((Q * self.scale) * K, dim=1)
        return a + (attn * V)
    
class UpBlock(nn.Module):
    def __init__(self, in_ch:int, out_ch:int, skip_ch:int=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x, skip:Optional[torch.Tensor]=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ChangeUNetDecoder(nn.Module):
    """
    U-Net decoder for change detection head.
    Lightweight decoder: (2*C) -> 1 logit.
    """
    def __init__(self, c: int):
        super().__init__()
        self.bottleneck = nn.Conv2d(2*c, c, 1)
        self.up1 = UpBlock(c, c//2)
        self.up2 = UpBlock(c//2, c//4)
        self.up3 = UpBlock(c//4, c//8)  
        self.out = nn.Conv2d(c//8, 1, 1)
    def forward(self, f0, f1):
        x = F.relu(self.bottleneck(torch.cat([f0, f1], dim=1)))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.out(x)
    
#----------- the Adapter -----------

class DeforestationChangeAdapter(nn.Module):
    """
    Plug-and-play adapter:
    - Expects per-time features for S1 and/or S2 from a shared SSL4EO backbone.
    - Fuses S1<->S2 mid-level, then Siamese differencing (t0 vs t1), then decodes.
    I/O:
        forward(batch, backbone) -> {"logits": (B,1,H,W)}
        where batch = {
            "t0": {"s1": Tensor or None, "s2": Tensor or None},
            "t1": {"s1": Tensor or None, "s2": Tensor or None}
        }
    """
    def __init__(self, c_backbone: int, c_align: int = 256):
        super().__init__()
        # backbone outputs 512 per branch; if two branches (s1+s2) we align anf fuse to c_align
        self.align_s1 = ConvAlign(c_backbone, c_align)
        self.align_s2 = ConvAlign(c_backbone, c_align)
        self.fuse = CrossAttentionFuse(c_align)
        self.decoder = ChangeUNetDecoder(c_align)

    def _fuse_sensors(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        feats may contain 's1' and/or 's2'. If one is missing, just align the present one.
        """
        if "s1" in feats and "s2" in feats:
            f1 = self.align_s1(feats["s1"])        
            f2 = self.align_s2(feats["s2"])
            # symmetric fuse: a<-b and b<-a then average
            a = self.fuse(f1, f2)
            b = self.fuse(f2, f1)
            return 0.5 * (a + b)
        elif "s1" in feats:
            return self.align_s1(feats["s1"])
        elif "s2" in feats:
            return self.align_s2(feats["s2"])
        else:
            raise ValueError("No sensor features provided for fusion.")
        
    def _make_fusion_batch(self, time_dict: Dict[str, torch.Tensor]) -> FusionBatch:
        """
        Wraps the per-time sensor features into a FusionBatch for potential use in more complex fusion kernels.
        
        time_dict: {"s1": Tensor?, "s2": Tensor?}
        """
        imagery = {}
        masks = {}

        for modality in ("s1", "s2"):
            if modality in time_dict and time_dict[modality] is not None:
                x = time_dict[modality]
                imagery[modality] = x
                masks[modality] = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], 
                                             dtype=torch.bool, 
                                                device=x.device)  # assume full valid mask for present modalities
                
        return FusionBatch(imagery=imagery, masks=masks, meta={})
    

    def forward(self, batch: Dict, fusion_kernel: BaseFusionKernel) -> Dict[str, torch.Tensor]:
        """
        fusion_kernel: BaseFusionKernel
        batch:
          "t0": {"s1": Tensor or None, "s2": Tensor or None}
          "t1": {"s1": Tensor or None, "s2": Tensor or None}
        """
        # # encode each time with the same backbones (weights frozen by backbone class)
        # feats_t0 = backbone(batch["t0"].get("s1"), batch["t0"].get("s2"))
        # feats_t1 = backbone(batch["t1"].get("s1"), batch["t1"].get("s2"))

        # # fuse sensors per time
        # ft0 = self._fuse_sensors(feats_t0)  # (B, C_align, H/32, W/32)
        # ft1 = self._fuse_sensors(feats_t1)

        # # decode
        # logits = self.decoder(ft0, ft1) # (B,1,H,W) 
        # return {"logits": logits}
        
        # Build FusionBatch objects for t0 and t1
        fb_t0 = self._make_fusion_batch(batch["t0"])
        fb_t1 = self._make_fusion_batch(batch["t1"])

        # call fusion kernel for each time step to get fused features
        fusion_t0: FusionOutput = fusion_kernel(fb_t0)
        fusion_t1: FusionOutput = fusion_kernel(fb_t1)

        feats_t0 = fusion_t0.per_modality # e.g. {"s1": Tensor, "s2": Tensor} 
        feats_t1 = fusion_t1.per_modality

        ft0 = self._fuse_sensors(feats_t0)  # (B, C_align, H/32, W/32)
        ft1 = self._fuse_sensors(feats_t1)

        logits = self.decoder(ft0, ft1) # (B,1,H,W)
        return {"logits": logits}
