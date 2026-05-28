import torch

from eintelligence.fusion.kernel_base import FusionBatch
from eintelligence.fusion.late_fusion_kernel import LateFusionKernel
from eintelligence.backbone.ssl4eo_lite_backbone import MultiSensorSSL4EOLiteBackbone

def test_late_fusion_kernel_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s1_channels = 2
    s2_channels = 4
    B, H, W = 2, 256, 256
    fused_dim = 128

    backbone = MultiSensorSSL4EOLiteBackbone(
        s1_cfg=None,
        s2_cfg=None,
    ).to(device)

    kernel = LateFusionKernel(
        backbone=backbone,
        fused_dim=fused_dim,
    ).to(device)

    kernel.eval()

    s1 = torch.randn(B, s1_channels, H, W, device=device)
    s2 = torch.randn(B, s2_channels, H, W, device=device)

    batch = FusionBatch(
        imagery={"s1": s1, "s2": s2},
        masks={
            "s1": torch.ones(B, 1, H, W, dtype=torch.bool, device=device),
            "s2": torch.ones(B, 1, H, W, dtype=torch.bool, device=device),
        },
        meta={},
    )

    with torch.no_grad():
        out = kernel(batch)

    assert out.fused.ndim == 4
    assert out.fused.shape[0] == B
    assert out.fused.shape[1] == fused_dim
    assert out.fused.shape[2] > 0 and out.fused.shape[3] > 0


    # Ensure per modality matches backbone keys
    assert isinstance(out.per_modality, dict)
    assert "s1" in out.per_modality or "s2" in out.per_modality
    