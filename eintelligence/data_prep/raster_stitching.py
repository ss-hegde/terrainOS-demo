from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Iterable, Optional, Any
import json

import numpy as np
import rasterio
from rasterio.merge import merge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def _mask_to_rgb(mask: np.ndarray, color_map: dict[int, tuple[float, float, float]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in color_map.items():
        rgb[mask == cls] = color
    return rgb


def save_mosaic_quicklook(
    mask_tif_path: Path,
    out_png_path: Path,
    color_map: dict[int, tuple[float, float, float]],
) -> Path:
    with rasterio.open(mask_tif_path) as src:
        arr = src.read(1)

    rgb = _mask_to_rgb(arr, color_map)

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax.imshow(rgb)
    ax.axis("off")
    fig.savefig(out_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png_path


def stitch_geotiff_tiles(
    tile_paths: Iterable[str | Path],
    out_path: str | Path,
    *,
    method: str = "first",
    compress: str = "deflate",
    nodata: Optional[int] = 255,
) -> Path:
    tile_paths = [Path(p) for p in tile_paths]
    out_path = Path(out_path)

    if not tile_paths:
        raise RuntimeError("No tile paths provided for stitching.")

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        first_crs = srcs[0].crs
        if first_crs is None:
            raise RuntimeError(f"First tile has no CRS: {tile_paths[0]}")

        bad = []
        good_srcs = []
        for src, path in zip(srcs, tile_paths):
            if src.crs != first_crs:
                bad.append(f"{path} (crs={src.crs})")
            else:
                good_srcs.append(src)

        if bad:
            raise RuntimeError(
                "CRS mismatch in stitch group. "
                f"Expected {first_crs}, got: " + "; ".join(bad)
            )

        mosaic, out_transform = merge(good_srcs, method=method)

        out_meta = good_srcs[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "compress": compress,
                "crs": first_crs,
            }
        )
        if nodata is not None:
            out_meta["nodata"] = nodata

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        return out_path
    finally:
        for src in srcs:
            src.close()


def group_prediction_tiles_by_scene(
    pred_root: str | Path,
    *,
    suffix: str = "_landcover_pred.tif",
) -> Dict[str, List[Path]]:
    pred_root = Path(pred_root)
    groups: Dict[str, List[Path]] = {}

    for tif_path in pred_root.rglob(f"*{suffix}"):
        if tif_path.parent.name == "quicklooks":
            continue

        rel = tif_path.relative_to(pred_root)

        # expected layout:
        # pred_root/<stitch_namespace>/<scene_id>/<tile_pred.tif>
        if len(rel.parts) < 3:
            continue

        stitch_namespace = rel.parts[0]
        scene_id = rel.parts[1]
        group_key = f"{stitch_namespace}/{scene_id}"

        groups.setdefault(group_key, []).append(tif_path)

    return groups

from typing import Any

def stitch_prediction_tree_by_scene(
    pred_root: str | Path,
    out_dir: str | Path,
    *,
    suffix: str = "_landcover_pred.tif",
    color_map: Optional[dict[int, tuple[float, float, float]]] = None,
    manifest_path: Optional[str | Path] = None,
    manifest_out_name: str = "stitched_scenes.json",
) -> Path:
    pred_root = Path(pred_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_groups = group_prediction_tiles_by_scene(pred_root, suffix=suffix)
    manifest_groups = group_manifest_tiles_by_scene(manifest_path) if manifest_path is not None else {}

    stitched = []
    skipped = []

    for group_key, tile_paths in sorted(pred_groups.items()):
        try:
            stitch_namespace, scene_id = group_key.split("/", 1)
        except ValueError:
            stitch_namespace = "unknown"
            scene_id = group_key.replace("/", "_")

        scene_dir = out_dir / stitch_namespace / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        pred_mosaic_path = scene_dir / f"{scene_id}_pred_mosaic.tif"

        try:
            stitch_geotiff_tiles(tile_paths, pred_mosaic_path)
        except Exception as e:
            skipped.append(
                {
                    "group_key": group_key,
                    "tile_count": len(tile_paths),
                    "tile_paths": [str(p) for p in sorted(tile_paths)],
                    "error": str(e),
                }
            )
            print(f"[WARN] skipping stitch group {group_key}: {e}")
            continue

        pred_quicklook_path = None
        if color_map is not None:
            pred_quicklook_path = scene_dir / f"{scene_id}_pred_mosaic.png"
            save_mosaic_quicklook(pred_mosaic_path, pred_quicklook_path, color_map)

        gt_mosaic_path = None
        s2_mosaic_path = None
        compare_png_path = None

        records = manifest_groups.get(group_key, [])
        if records:
            s2_paths = [Path(r["s2_path"]) for r in records if r.get("s2_path")]
            wc_paths = [Path(r["worldcover_path"]) for r in records if r.get("worldcover_path")]

            if s2_paths:
                s2_mosaic_path = scene_dir / f"{scene_id}_s2_mosaic.tif"
                try:
                    stitch_geotiff_tiles(s2_paths, s2_mosaic_path, nodata=None)
                except Exception as e:
                    print(f"[WARN] could not stitch S2 mosaic for {group_key}: {e}")
                    s2_mosaic_path = None

            if wc_paths:
                gt_mosaic_path = scene_dir / f"{scene_id}_gt_mosaic.tif"
                try:
                    stitch_geotiff_tiles(wc_paths, gt_mosaic_path, nodata=255)
                except Exception as e:
                    print(f"[WARN] could not stitch GT mosaic for {group_key}: {e}")
                    gt_mosaic_path = None

            if (
                color_map is not None
                and s2_mosaic_path is not None
                and gt_mosaic_path is not None
            ):
                compare_png_path = scene_dir / f"{scene_id}_s2_gt_pred.png"
                try:
                    save_scene_comparison_png(
                        s2_mosaic_tif=s2_mosaic_path,
                        gt_mosaic_tif=gt_mosaic_path,
                        pred_mosaic_tif=pred_mosaic_path,
                        out_png=compare_png_path,
                        color_map=color_map,
                    )
                except Exception as e:
                    print(f"[WARN] could not save scene comparison for {group_key}: {e}")
                    compare_png_path = None

        stitched.append(
            {
                "group_key": group_key,
                "stitch_namespace": stitch_namespace,
                "scene_id": scene_id,
                "tile_count": len(tile_paths),
                "tile_paths": [str(p) for p in sorted(tile_paths)],
                "pred_mosaic_path": str(pred_mosaic_path),
                "pred_quicklook_path": str(pred_quicklook_path) if pred_quicklook_path is not None else None,
                "s2_mosaic_path": str(s2_mosaic_path) if s2_mosaic_path is not None else None,
                "gt_mosaic_path": str(gt_mosaic_path) if gt_mosaic_path is not None else None,
                "compare_png_path": str(compare_png_path) if compare_png_path is not None else None,
            }
        )

    manifest_path_out = out_dir / manifest_out_name
    manifest_path_out.write_text(json.dumps({"scenes": stitched, "skipped": skipped}, indent=2))
    return manifest_path_out

def save_scene_comparison_png(
    s2_mosaic_tif: Path,
    gt_mosaic_tif: Path,
    pred_mosaic_tif: Path,
    out_png: Path,
    color_map: dict[int, tuple[float, float, float]],
) -> Path:
    with rasterio.open(s2_mosaic_tif) as src:
        s2 = src.read()

    with rasterio.open(gt_mosaic_tif) as src:
        gt = src.read(1)

    with rasterio.open(pred_mosaic_tif) as src:
        pred = src.read(1)

    # assume S2 bands are B02, B03, B04, B08 in dataset order [B02, B03, B04, B08]
    # build RGB as R=B04, G=B03, B=B02
    if s2.shape[0] < 3:
        raise RuntimeError(f"S2 mosaic has fewer than 3 bands: {s2_mosaic_tif}")

    blue = s2[0].astype(np.float32)
    green = s2[1].astype(np.float32)
    red = s2[2].astype(np.float32)

    rgb = np.stack([red, green, blue], axis=-1)

    # robust visualization stretch
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    p2, p98 = np.percentile(rgb, [2, 98])
    if p98 > p2:
        rgb = np.clip((rgb - p2) / (p98 - p2), 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    gt_rgb = _mask_to_rgb(gt, color_map)
    pred_rgb = _mask_to_rgb(pred, color_map)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title("S2 RGB")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png

def group_manifest_tiles_by_scene(
    manifest_path: str | Path,
) -> Dict[str, List[Dict[str, Any]]]:
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    tiles = data.get("tiles", [])

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in tiles:
        scene_id = str(rec.get("scene_id") or "")
        group_id = str(rec.get("group_id") or rec.get("aoi_id") or scene_id)
        key = f"{group_id}/{scene_id}"
        groups.setdefault(key, []).append(rec)

    return groups