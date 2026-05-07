import torch
from pathlib import Path

from orchestrator.workflow_manager_multisensor_v1 import _TrainerMS, TrainingConfig
from eintelligence.fusion.kernel_base import MultisensorIdentityFusionKernel
from eintelligence.backbone.ssl4eo_lite_backbone import MultiSensorSSL4EOLiteBackbone
from eintelligence.adapters.deforestation_change import DeforestationChangeAdapter


def test_trainer_step_with_fusion_kernel():
    device = torch.device("cpu")

    s1_ch = 2
    s2_ch = 4
    batch_size = 2
    H = W = 256

    backbone = MultiSensorSSL4EOLiteBackbone(
        s1_cfg=None,
        s2_cfg=None,
    ).to(device)
    fusion_kernel = MultisensorIdentityFusionKernel(backbone).to(device)
    adapter = DeforestationChangeAdapter(c_backbone=512, c_align=256).to(device)

    batch = {
        "t0": {"s1": torch.randn(batch_size, s1_ch, H, W),
               "s2": torch.randn(batch_size, s2_ch, H, W)},
        "t1": {"s1": torch.randn(batch_size, s1_ch, H, W),
               "s2": torch.randn(batch_size, s2_ch, H, W)},
        "target": torch.randint(0, 2, (batch_size, 1, H, W)).float(),
    }

    cfg = TrainingConfig(
        num_epochs=1,
        batch_size=batch_size,
        val_fraction=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        amp=False,
        num_workers=0,
    )
    trainer = _TrainerMS(device, cfg)

    # Build a simple optimizer just for adapter parameters as in your fit()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, adapter.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    loss_value = trainer._step(batch, fusion_kernel, adapter, optimizer=optimizer, train=True)
    assert isinstance(loss_value, float)
    assert loss_value > 0.0