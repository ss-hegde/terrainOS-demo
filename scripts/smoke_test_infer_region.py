"""
Standalone smoke test for LandCoverWorkflowMS.infer_region().

Not a pytest file -- run directly:

    EARTH_PROJECT_ROOT=$(pwd) .venv/bin/python scripts/smoke_test_infer_region.py

Exercises infer_region() end-to-end against the existing Munich AOI/dates and the
already-trained s2-only checkpoint, and checks (design call 1 from the infer_region
review) that the shared training-pool manifest is never touched by an infer_region()
call -- that guarantee is asserted here in code, not just trusted from reading the
source.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from eintelligence.data_prep.aoi import square_aoi
from orchestrator.workflow_manager_landcover import (
    LandCoverWorkflowMS,
    TilingConfigLandCover,
    TrainingConfigLandCover,
)


POOLED_MANIFEST = PROJECT_ROOT / "data" / "corpus" / "landcover_manifest_multisensor.json"
CKPT_PATH = PROJECT_ROOT / "models" / "landcover_s2.pt"
OUT_DIR = PROJECT_ROOT / "data" / "corpus" / "smoke_test_infer_region"


def _pooled_manifest_fingerprint() -> tuple[float, int]:
    mtime = POOLED_MANIFEST.stat().st_mtime
    n_tiles = len(json.loads(POOLED_MANIFEST.read_text()).get("tiles", []))
    return mtime, n_tiles


def main() -> None:
    if not POOLED_MANIFEST.is_file():
        raise RuntimeError(f"Shared pooled manifest not found, can't fingerprint it: {POOLED_MANIFEST}")
    if not CKPT_PATH.is_file():
        raise RuntimeError(f"Expected checkpoint not found: {CKPT_PATH}")

    before_mtime, before_tiles = _pooled_manifest_fingerprint()
    print(f"[before] pooled manifest mtime={before_mtime} tiles={before_tiles}")

    tiling_cfg = TilingConfigLandCover(
        bands_s2=("B02", "B03", "B04", "B08"),
        bands_s1=("vv", "vh"),
        tile_size=256,
        stride=256,
        max_cloud=50,
        sensor_mode="s2",
    )
    train_cfg = TrainingConfigLandCover(amp=False)  # unused by infer_region, kept consistent with notebook

    wf = LandCoverWorkflowMS(PROJECT_ROOT, tiling_cfg, train_cfg)

    result = wf.infer_region(
        aoi_geojson=square_aoi(48.1351, 11.5820),
        start="2023-06-01",
        end="2023-08-01",
        ckpt_path=CKPT_PATH,
        region_name="munich_smoke_test_infer_region",
        out_dir=OUT_DIR,
    )

    after_mtime, after_tiles = _pooled_manifest_fingerprint()
    print(f"[after]  pooled manifest mtime={after_mtime} tiles={after_tiles}")

    assert after_mtime == before_mtime, (
        f"Shared pooled manifest mtime changed: {before_mtime} -> {after_mtime}. "
        "infer_region() must never write to the training corpus's pooled manifest."
    )
    assert after_tiles == before_tiles, (
        f"Shared pooled manifest tile count changed: {before_tiles} -> {after_tiles}. "
        "infer_region() must never write to the training corpus's pooled manifest."
    )
    print("[OK] shared pooled manifest untouched by infer_region()")

    print("\n=== infer_region() payload ===")
    print("keys:", list(result.keys()))
    print("region_name:", result["region_name"])
    print("manifest_path:", result["manifest_path"])
    print("stitched_manifest_path:", result["stitched_manifest_path"])

    tiles = result["tiles"]
    n_compare = sum(1 for t in tiles if t.compare_quicklook_path is not None)
    print(f"tile count: {len(tiles)}")
    print(f"comparison PNGs produced: {n_compare} / {len(tiles)}")

    print("\nfirst few tiles (uncertainty / per_modality):")
    for t in tiles[:3]:
        print(
            f"  tile_id={t.tile_id!r} scene_id={t.scene_id!r} "
            f"pred_mask_path={t.pred_mask_path} "
            f"uncertainty={t.uncertainty} per_modality={t.per_modality}"
        )


if __name__ == "__main__":
    main()
