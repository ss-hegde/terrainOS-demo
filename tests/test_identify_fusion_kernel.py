import torch

from eintelligence.fusion.kernel_base import FusionBatch, FusionOutput, MultisensorIdentityFusionKernel
from eintelligence.backbone.ssl4eo_lite_backbone import MultiSensorSSL4EOLiteBackbone

def test_identify_kernel_matches_backbone():
    device = torch.device("cpu")

    s1_ch = 2 # VV, VH
    s2_ch = 4 # B02, B03, B04, B08
    batch_size = 2
    H = W = 256

    backbone = MultiSensorSSL4EOLiteBackbone(
        s1_cfg=None,
        s2_cfg=None,
    ).to(device)
    
    kernel = MultisensorIdentityFusionKernel(backbone).to(device)

    s1 = torch.randn(batch_size, s1_ch, H, W, device=device)
    s2 = torch.randn(batch_size, s2_ch, H, W, device=device)

    # Build FusionBatch for "s1s2"
    fb = FusionBatch(
        imagery={"s1":s1, "s2":s2},
        masks = {"s1": torch.ones(batch_size, 1, H, W, dtype = torch.bool, device=device),
                 "s1": torch.ones(batch_size, 1, H, W, dtype = torch.bool, device=device),
        },
        meta = {},

    )

    backbone.eval()
    kernel.eval()

    with torch.no_grad():
        # Direct backbone output
        feats_raw = backbone(s1, s2) # expected: {"s1": Tensor, "s2": Tensor}

        # Through fusion kernel
        fusion_out: FusionOutput = kernel(fb)

    # Check keys and shapes
    assert set(fusion_out.per_modality.keys()) == set(feats_raw.keys())
    for k in feats_raw:
        print("kernel out shape:", fusion_out.per_modality[k].shape)
        print("backbone out shape:", feats_raw[k].shape)
        assert fusion_out.per_modality[k].shape == feats_raw[k].shape
        # print("kernel out shape:", fusion_out.per_modality[k].shape)
        # print("backbone out shape:", feats_raw[k].shape)

        # values should match exactly
        assert torch.allclose(fusion_out.per_modality[k], feats_raw[k], atol=1e-6)