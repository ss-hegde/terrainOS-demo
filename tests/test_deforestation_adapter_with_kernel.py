import torch

from eintelligence.fusion.kernel_base import FusionBatch, MultisensorIdentityFusionKernel
from eintelligence.backbone.ssl4eo_lite_backbone import MultiSensorSSL4EOLiteBackbone
from eintelligence.adapters.deforestation_change import DeforestationChangeAdapter

def test_adapter_with_identity_kernel_forward():
    device = torch.device("cpu")

    s1_ch = 2 # VV, VH
    s2_ch = 4 # B02, B03, B04, B08
    batch_size = 2
    H = W = 256

    backbone = MultiSensorSSL4EOLiteBackbone(
        s1_cfg=None,
        s2_cfg=None,
    ).to(device)
    
    fusion_kernel = MultisensorIdentityFusionKernel(backbone).to(device)
    adapter = DeforestationChangeAdapter(c_backbone=512, c_align=256).to(device)

    # synthetic t0/t1 tensors

    t0_s1 = torch.randn(batch_size, s1_ch, H, W, device=device)
    t0_s2 = torch.randn(batch_size, s2_ch, H, W, device=device)
    t1_s1 = torch.randn(batch_size, s1_ch, H, W, device=device)
    t1_s2 = torch.randn(batch_size, s2_ch, H, W, device=device)

    batch = {
        "t0": {"s1":t0_s1, "s2":t0_s2},
        "t1": {"s1":t1_s1, "S2":t1_s2},
        "target": torch.zeros(batch_size, 1, H, W, device=device),

    }

    fusion_kernel.eval()
    adapter.eval()

    with torch.no_grad():
        out = adapter(batch, fusion_kernel)
        logits = out["logits"]

    assert logits.ndim == 4
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == 1
    # Height/width can be different due to decoder; just assert > 0
    assert logits.shape[2] > 0 and logits.shape[3] > 0