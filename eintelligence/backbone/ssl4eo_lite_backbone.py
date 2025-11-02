from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn as nn
import torchvision.models as tv

#----------- helpers -----------

def _adapt_conv1_weight(w: torch.Tensor, in_ch:int) -> torch.Tensor:
    """
    Adapt (out_ch,3,k,k) -> (out_ch,in_ch,k,k) by tiling/averaging with varience-preserving scale.
    """
    out_ch, old_in, k, _ = w.shape
    if in_ch == old_in:
        return w
    elif in_ch < old_in:
        w = w.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1)
        return w
    reps = (in_ch + old_in - 1) // old_in
    w_exp = w.repeat(1, reps, 1, 1)[:, :in_ch]
    scale = (old_in / float(in_ch))**0.5
    return w_exp * scale

def _make_resnet18_backbone(in_ch:int, state_dict:Optional[Dict[str, torch.Tensor]]=None) -> nn.Module:
    """
    Build a ResNet18 trunk (no classifier) with flexible input channels.
    Optionally load a state_dict (e.g., SSL4EO). We ignore conv1/fc when shapes mismatch.
    
    NOTE: REPLACE WITH RESNET50 DURING FINAL IMPLEMENTATION TO MATCH SSL4EO MODEL
    """
    try:
        model = tv.resnet18(weights=None)
    except TypeError:
        model = tv.resnet18(pretrained=False)
        
    if state_dict:
        model.load_state_dict(state_dict, strict=False)
    # adapt conv1
    with torch.no_grad():
        w = model.conv1.weight.detach().clone()
        model.conv1 = nn.Conv2d(in_ch, model.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.copy_(_adapt_conv1_weight(w, in_ch))

    # expose C*H*W feature map
    trunk = nn.Sequential(
        model.conv1, model.bn1, nn.ReLU(inplace=True), model.maxpool,
        model.layer1, model.layer2, model.layer3, model.layer4 # (B, 512, H/32, W/32)
    )
    trunk.out_channels = 512
    return trunk

@dataclass
class SSL4EOLiteConfig:
    in_ch: int # input channels for this backbone
    freeze: bool = True  # freeze by default; adapters train on top
    state_dict: Optional[Dict] = None  # pre-trained weights

class SSL4EOLiteBackbone(nn.Module):
    """
    Single-Sensor SSL4EO-Lite Backbone (ResNet18 trunk) for S-1 or S-2.
    """
    def __init__(self, cfg: SSL4EOLiteConfig):
        super().__init__()
        self.trunk = _make_resnet18_backbone(cfg.in_ch, cfg.state_dict)
        if cfg.freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False
        self.out_channels = self.trunk.out_channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,C_out=512,H/32,W/32)
        return self.trunk(x)
    
class MultiSensorSSL4EOLiteBackbone(nn.Module):
    """
    Multi-Sensor SSL4EO-Lite Backbone for S-1 + S-2.
    You can pass None for one of the branches to run single-sensor mode.
    """
    def __init__(
        self,
        s1_cfg: Optional[SSL4EOLiteConfig] = None,
        s2_cfg: Optional[SSL4EOLiteConfig] = None,
    ):
        super().__init__()
        assert s1_cfg is not None or s2_cfg is not None, "At least one sensor config must be provided."
        self.s1 = SSL4EOLiteBackbone(s1_cfg) if s1_cfg is not None else None
        self.s2 = SSL4EOLiteBackbone(s2_cfg) if s2_cfg is not None else None

        # define output channels for downstream heads
        out_ch = 0

        if self.s1 is not None:
            out_ch += self.s1.out_channels
        if self.s2 is not None:
            out_ch += self.s2.out_channels

        self.out_channels = out_ch

    @torch.no_grad()
    def forward(self, x_s1: Optional[torch.Tensor], x_s2: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Returns dict of features that can be fused in adapters
        feats = {}
        if self.s1 is not None and x_s1 is not None:
            feats["s1"] = self.s1(x_s1)  # (B,512,H/32,W/32)
        if self.s2 is not None and x_s2 is not None:
            feats["s2"] = self.s2(x_s2)  # (B,512,H/32,W/32)
        return feats
    

