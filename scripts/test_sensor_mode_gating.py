"""
Standalone behavioral test for the sensor_mode -> use_modalities gating fix in
LandCoverWorkflowMS (CLAUDE.md "Open" item 1: sensor_mode used to be inert).

Not a pytest file -- run directly:

    EARTH_PROJECT_ROOT=$(pwd) .venv/bin/python scripts/test_sensor_mode_gating.py

Builds LandCoverWorkflowMS with a freshly-initialized (untrained) model -- the
existing landcover_s2.pt/landcover_s1s2.pt checkpoints were trained under the old
always-both-modalities code and are expected to go stale once this fix lands
(retraining is explicitly out of scope for this script).

Exercises self.fusion_kernel directly against a synthetic FusionBatch (random
tensors in the shapes LateFusionKernel.forward expects: batch.imagery/masks keyed
by "s1"/"s2", BCHW). This tests the fusion kernel in isolation -- no STAC/tiling/
real imagery needed.

Two checks, for sensor_mode="s2" and as a control for sensor_mode="s1s2":
  1. FusionOutput.per_modality's keys match exactly what that sensor_mode should
     expose (not "every modality the backbone happened to compute").
  2. The REAL proof, not just "didn't crash" or "key is missing": zero out the s1
     tensor and rerun. In "s2" mode, FusionOutput.fused must be numerically
     identical (torch.allclose) -- s1 must have zero influence on the output. In
     "s1s2" mode (control), corrupting s1 MUST change fused -- otherwise the test
     itself would be vacuous.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from eintelligence.fusion.kernel_base import FusionBatch
from orchestrator.workflow_manager_landcover import (
    LandCoverWorkflowMS,
    TilingConfigLandCover,
    TrainingConfigLandCover,
)


def _make_batch(tcfg: TilingConfigLandCover, device: torch.device, seed: int) -> FusionBatch:
    g = torch.Generator().manual_seed(seed)
    B, H, W = 2, tcfg.tile_size, tcfg.tile_size
    s1 = torch.randn(B, len(tcfg.bands_s1), H, W, generator=g).to(device)
    s2 = torch.randn(B, len(tcfg.bands_s2), H, W, generator=g).to(device)
    masks = {
        "s1": torch.ones(B, 1, H, W, dtype=torch.bool, device=device),
        "s2": torch.ones(B, 1, H, W, dtype=torch.bool, device=device),
    }
    return FusionBatch(imagery={"s1": s1, "s2": s2}, masks=masks, meta={})


def _run(wf: LandCoverWorkflowMS, fb: FusionBatch):
    wf.fusion_kernel.eval()
    with torch.no_grad():
        return wf.fusion_kernel(fb)


def check_sensor_mode(sensor_mode: str, expect_gated: bool) -> None:
    print(f"\n=== sensor_mode={sensor_mode!r} (expect_gated={expect_gated}) ===")

    tcfg = TilingConfigLandCover(sensor_mode=sensor_mode)
    train_cfg = TrainingConfigLandCover(amp=False)
    wf = LandCoverWorkflowMS(PROJECT_ROOT, tcfg, train_cfg)
    device = wf.device

    fb = _make_batch(tcfg, device, seed=0)
    out1 = _run(wf, fb)

    print(f"per_modality keys: {sorted(out1.per_modality.keys())}")
    expected_keys = {"s2"} if sensor_mode == "s2" else ({"s1"} if sensor_mode == "s1" else {"s1", "s2"})
    assert set(out1.per_modality.keys()) == expected_keys, (
        f"per_modality keys {set(out1.per_modality.keys())} != expected {expected_keys} "
        f"for sensor_mode={sensor_mode!r}"
    )
    print(f"[OK] per_modality keys == {expected_keys}")

    # Corrupt s1 (zero it out) and rerun through the SAME kernel instance (so the
    # lazily-built proj layer's weights are identical across both calls).
    fb_corrupt = FusionBatch(
        imagery={"s1": torch.zeros_like(fb.imagery["s1"]), "s2": fb.imagery["s2"]},
        masks=fb.masks,
        meta={},
    )
    out2 = _run(wf, fb_corrupt)

    is_identical = torch.allclose(out1.fused, out2.fused)
    print(f"fused identical after zeroing s1: {is_identical}")
    print(f"  max abs diff: {(out1.fused - out2.fused).abs().max().item():.6g}")

    if expect_gated:
        assert is_identical, (
            f"sensor_mode={sensor_mode!r} should be gated to s2 only -- fused must be "
            "invariant to s1, but it changed when s1 was zeroed."
        )
        print("[OK] fused is invariant to s1 corruption -- s1 has zero influence, as expected")
    else:
        assert not is_identical, (
            f"sensor_mode={sensor_mode!r} control check failed -- fused did NOT change when "
            "s1 was zeroed, which would mean this test is vacuous (s1 never mattered here "
            "either way)."
        )
        print("[OK] fused DID change when s1 was corrupted -- confirms the test is meaningful")


def main() -> None:
    # s2-only: per_modality must be exactly {"s2"}, and fused must be invariant to s1.
    check_sensor_mode("s2", expect_gated=True)

    # s1s2 control: per_modality must be {"s1", "s2"}, and corrupting s1 MUST move fused --
    # otherwise the invariance check above wouldn't be proving anything.
    check_sensor_mode("s1s2", expect_gated=False)

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
