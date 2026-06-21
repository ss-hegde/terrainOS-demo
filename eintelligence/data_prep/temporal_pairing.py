import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np

import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds, reproject, Resampling

from shapely.geometry import shape, box
from shapely.strtree import STRtree


from rasterio.windows import from_bounds
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer

WORLD_COVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

def _tile_key(props: dict) -> tuple:
    """
    Produce a stable key for matching tiles between scenes.
    Supports both old schema (row/col) and new AOI schema (row_off/col_off).

    Preference order:
    1) explicit grid indices: (row, col)
    2) derive grid indices from pixel offsets: (row_off // stride, col_off // stride)
       falling back to // size if stride is missing
    3) last resort: exact pixel offsets (row_off, col_off)
    """
    if "row" in props and "col" in props:
        return (int(props["row"]), int(props["col"]))

    if "row_off" in props and "col_off" in props:
        row_off = int(props["row_off"])
        col_off = int(props["col_off"])
        # derive grid step
        step = int(props.get("stride", props.get("size", 1)))
        if step > 0:
            return (row_off // step, col_off // step)
        # fallback if step missing/bad
        return (row_off, col_off)

    # ultimate fallback: use bounds (rounded) if nothing else available
    if "geometry" in props:  # not typical; geometry is on the feature, not props
        pass
    # Unusual manifest; return something deterministic to avoid crash
    return (hash(frozenset(props.items())), 0)


def build_rowcol_index(manifest_path: Path) -> dict:
    """
    Build an index mapping tile-key -> absolute tile path for a single scene manifest.
    """
    m = json.loads(Path(manifest_path).read_text())
    idx = {}
    tiles_base = Path(manifest_path).parent
    for ft in m["features"]:
        props = ft["properties"]
        path = tiles_base / props["path"]
        key = _tile_key(props)
        idx[key] = str(path)
    return idx

def build_temporal_pairs(
    collection_manifest_path: Path, 
) -> Path:
    
    """
    For consecutive scenes s_i, s_(i+1), pair tiles with same (row, col).
    Returns a JSON file, pairs_manifest.json, with entries:
    { t1_path, t2_path, row, col, s1_id, s2_id}
    """

    collection = json.loads(Path(collection_manifest_path).read_text())["scenes"]
    collection = sorted(collection, key=lambda e: e["datetime"])
    pairs = []
    for i in range(len(collection) - 1):
        A, B = collection[i], collection[i+1]
        index_A = build_rowcol_index(Path(A["manifest_path"]))
        index_B = build_rowcol_index(Path(B["manifest_path"]))
        common_keys = set(index_A.keys()).intersection(index_B.keys())
        for (r, c) in common_keys:
            pairs.append({
                "row": r, "col": c,
                "t1_path": index_A[(r, c)], "t2_path": index_B[(r, c)],
                "s1_id": A["scene_id"], "s2_id": B["scene_id"]
            })
    
    out_path = Path(collection_manifest_path).parent / "pairs_manifest.json"
    out_path.write_text(json.dumps({"pairs": pairs}, indent=2))
    return out_path

def _read_scene_tiles(manifest_path: Path):
    """
    Returns a list of dicts:
      [{"geom": shapely_polygon, "path": abs_path, "props": props_dict}, ...]
    """
    base = Path(manifest_path).parent
    m = json.loads(Path(manifest_path).read_text())
    tiles = []
    for ft in m["features"]:
        geom = shape(ft["geometry"])
        props = ft["properties"]
        tiles.append({
            "geom": geom,
            "path": str(base / props["path"]),
            "props": props
        })
    return tiles

def build_temporal_pairs_relaxed_s1(
    collection_manifest_path: Path,
    iou_thresh: float = 0.90
) -> Path:
    """
    Pair consecutive S1 scenes by matching tiles whose geographic bounds overlap
    with IoU >= iou_thresh. Outputs pairs_manifest.json with entries:
      { "row": <int|None>, "col": <int|None>,
        "t1_path": <str>, "t2_path": <str>,
        "s1_id": <scene_id_prev>, "s2_id": <scene_id_next> }

    Notes:
    - Falls back to row/col=None when we don't have stable grid indices.
    - Much more tolerant to tiny grid misalignments between scenes.
    """
    coll = json.loads(Path(collection_manifest_path).read_text())["scenes"]
    coll = sorted(coll, key=lambda e: e["datetime"])
    out = []

    for i in range(len(coll) - 1):
        A, B = coll[i], coll[i+1]
        tiles_A = _read_scene_tiles(Path(A["manifest_path"]))
        tiles_B = _read_scene_tiles(Path(B["manifest_path"]))

        # index B tiles by simple bbox hash buckets for speed (optional, simple O(N^2) is fine for small sets)
        for ta in tiles_A:
            ga = ta["geom"]
            best_iou = 0.0
            best_tb = None

            # brute-force match (usually a few hundred tiles max)
            for tb in tiles_B:
                gb = tb["geom"]
                inter = ga.intersection(gb).area
                if inter <= 0.0:
                    continue
                union = ga.union(gb).area
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_tb = iou, tb

            if best_tb is not None and best_iou >= iou_thresh:
                # try to keep row/col if present (won't be identical across scenes necessarily)
                r = ta["props"].get("row")
                c = ta["props"].get("col")

                out.append({
                    "row": int(r) if r is not None else None,
                    "col": int(c) if c is not None else None,
                    "t1_path": ta["path"],
                    "t2_path": best_tb["path"],
                    "s1_id": A["scene_id"],
                    "s2_id": B["scene_id"],
                    "iou": float(best_iou)
                })

    out_path = Path(collection_manifest_path).parent / "pairs_manifest.json"
    out_path.write_text(json.dumps({"pairs": out}, indent=2))
    return out_path



def _read_manifest_features(manifest_path: Path):
    m = json.loads(Path(manifest_path).read_text())
    base = Path(manifest_path).parent
    feats = []
    for ft in m["features"]:
        geom = shape(ft["geometry"])
        props = ft["properties"]
        abs_path = str(base / props["path"])
        feats.append((geom, props, abs_path))
    return feats


def _build_spatial_index(feats):
    geoms = [g for g, _, _ in feats]
    tree = STRtree(geoms)
    return tree, geoms


def _best_overlap(target_geom, tree: STRtree, geoms: List, feats: List, iou_min=0.8):
    cand_idxs = tree.query(target_geom)
    best_iou, best_idx = 0.0, None
    tg_area = target_geom.area if target_geom.area > 0 else 1e-9
    for idx in cand_idxs:
        g = geoms[idx]
        inter = target_geom.intersection(g).area
        union = tg_area + g.area - inter
        iou = inter / max(union, 1e-9)
        if iou > best_iou:
            best_iou, best_idx = iou, idx
    if best_idx is None or best_iou < iou_min:
        return None, None
    _geom, props, abs_path = feats[best_idx]
    return abs_path, props


def _worldcover_tile_name(lon: float, lat: float) -> str:
    lon0 = math.floor(lon / 3) * 3
    lat0 = math.floor(lat / 3) * 3
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def _worldcover_urls_for_bounds(bounds_4326, version="v200", year="2021") -> List[str]:
    minx, miny, maxx, maxy = bounds_4326
    lon_starts = range(math.floor(minx / 3) * 3, math.floor(maxx / 3) * 3 + 1, 3)
    lat_starts = range(math.floor(miny / 3) * 3, math.floor(maxy / 3) * 3 + 1, 3)

    urls = []
    seen = set()
    for lon0 in lon_starts:
        for lat0 in lat_starts:
            tile = _worldcover_tile_name(lon0 + 0.001, lat0 + 0.001)
            if tile in seen:
                continue
            seen.add(tile)
            fname = f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
            urls.append(f"{WORLD_COVER_BASE}/{version}/{year}/map/{fname}")
    return urls


def _write_aligned_worldcover_label(
    *,
    s2_tile_path: Path,
    out_label_path: Path,
    worldcover_version: str = "v200",
    worldcover_year: str = "2021",
) -> Optional[Path]:
    with rasterio.open(s2_tile_path) as s2:
        s2_bounds = s2.bounds
        s2_crs = s2.crs
        s2_transform = s2.transform
        s2_height = s2.height
        s2_width = s2.width
        s2_profile = s2.profile.copy()

        bounds_4326 = transform_bounds(s2_crs, "EPSG:4326", *s2_bounds, densify_pts=21)
        wc_urls = _worldcover_urls_for_bounds(
            bounds_4326, version=worldcover_version, year=worldcover_year
        )

        srcs = []
        for url in wc_urls:
            try:
                srcs.append(rasterio.open(url))
            except Exception:
                continue

        if not srcs:
            return None

        try:
            mosaic, mosaic_transform = merge(srcs, bounds=bounds_4326)
            src_arr = mosaic[0]

            dst_arr = np.zeros((s2_height, s2_width), dtype=np.uint8)
            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=mosaic_transform,
                src_crs="EPSG:4326",
                dst_transform=s2_transform,
                dst_crs=s2_crs,
                resampling=Resampling.nearest,
            )

            out_label_path.parent.mkdir(parents=True, exist_ok=True)
            s2_profile.update(count=1, dtype="uint8", compress="deflate", nodata=0)

            with rasterio.open(out_label_path, "w", **s2_profile) as dst:
                dst.write(dst_arr, 1)

            return out_label_path
        finally:
            for src in srcs:
                src.close()


def build_temporal_pairs_multisensor(
    *,
    s2_collection_manifest_path: Path,
    s1_collection_manifest_path: Path,
    iou_min: float = 0.8
) -> Path:
    """
    For adjacent S2 scenes (t0,t1):
      - Find nearest-time S1 scenes (t0',t1')
      - For each S2 tile at t0/t1, find best-overlap S1 tiles via IoU
      - Emit pairs where ALL FOUR tiles overlap sufficiently.
    Outputs: pairs_manifest_multisensor.json adjacent to S2 collection.
    """
    def _parse_iso(t): return datetime.fromisoformat(str(t).replace("Z","+00:00"))

    s2_collection = json.loads(Path(s2_collection_manifest_path).read_text())["scenes"]
    s1_collection = json.loads(Path(s1_collection_manifest_path).read_text())["scenes"]
    if not s2_collection or not s1_collection:
        raise RuntimeError("Empty S2 or S1 collection for multisensor pairing.")

    # sort
    s2_collection = sorted(s2_collection, key=lambda e: e["datetime"])
    s1_collection = sorted(s1_collection, key=lambda e: e["datetime"])

    # nearest S1 helper
    def nearest_scene(sc_list, target_iso):
        tgt = _parse_iso(target_iso)
        best, best_dt = None, None
        for s in sc_list:
            dt = abs((_parse_iso(s["datetime"]) - tgt).total_seconds())
            if best is None or dt < best_dt:
                best, best_dt = s, dt
        return best

    pairs = []

    for i in range(len(s2_collection) - 1):
        A2, B2 = s2_collection[i], s2_collection[i+1]
        A1 = nearest_scene(s1_collection, A2["datetime"])
        B1 = nearest_scene(s1_collection, B2["datetime"])

        # read features for all four scenes
        feats_A2 = _read_manifest_features(Path(A2["manifest_path"]))
        feats_B2 = _read_manifest_features(Path(B2["manifest_path"]))
        feats_A1 = _read_manifest_features(Path(A1["manifest_path"]))
        feats_B1 = _read_manifest_features(Path(B1["manifest_path"]))

        # spatial indexes for S1 sets
        tree_A1, geoms_A1 = _build_spatial_index(feats_A1)
        tree_B1, geoms_B1 = _build_spatial_index(feats_B1)

        # index S2 tiles by a stable key (use their own grid key)
        # and store geometry to test overlap consistency between t0 and t1
        def _s2_key(props: dict) -> Tuple[int,int]:
            if "row" in props and "col" in props:
                return (int(props["row"]), int(props["col"]))
            if "row_off" in props and "col_off" in props:
                step = int(props.get("stride", props.get("size", 1)))
                return (int(props["row_off"]) // max(step,1), int(props["col_off"]) // max(step,1))
            # fallback: rounded bounds hash (rare)
            return (hash(frozenset(props.items())), 0)

        idx_A2 = { _s2_key(p): (g, p, path) for (g,p,path) in feats_A2 }
        idx_B2 = { _s2_key(p): (g, p, path) for (g,p,path) in feats_B2 }

        # For each S2 tile that exists at both t0 and t1,
        # find best-overlap S1 tiles at t0' and t1'
        common_keys = set(idx_A2.keys()) & set(idx_B2.keys())
        for k in common_keys:
            gA2, pA2, pathA2 = idx_A2[k]
            gB2, pB2, pathB2 = idx_B2[k]

            s1A_path, _ = _best_overlap(gA2, tree_A1, geoms_A1, feats_A1, iou_min=iou_min)
            if s1A_path is None: continue
            s1B_path, _ = _best_overlap(gB2, tree_B1, geoms_B1, feats_B1, iou_min=iou_min)
            if s1B_path is None: continue

            pairs.append({
                "row": k[0], "col": k[1],
                "t0": {"s2": pathA2, "s1": s1A_path},
                "t1": {"s2": pathB2, "s1": s1B_path},
                "scene_ids": {
                    "s2_t0": A2["scene_id"], "s2_t1": B2["scene_id"],
                    "s1_t0": A1["scene_id"], "s1_t1": B1["scene_id"],
                }
            })

    out_path = Path(s2_collection_manifest_path).parent / "pairs_manifest_multisensor.json"
    out_path.write_text(json.dumps({"pairs": pairs}, indent=2))
    # helpful log if empty
    if not pairs:
        print("[WARN] multisensor pairing produced 0 pairs. Consider lowering iou_min or checking AOI alignment.")
    return out_path

def build_landcover_multisensor_manifest(
    *,
    s2_collection_manifest_path: Path,
    s1_collection_manifest_path: Path,
    iou_min: float = 0.8,
    worldcover_version: str = "v200",
    worldcover_year: str = "2021",
) -> Path:
    def _parse_iso(t: str) -> datetime:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00"))

    s2_collection = json.loads(Path(s2_collection_manifest_path).read_text())["scenes"]
    s1_collection = json.loads(Path(s1_collection_manifest_path).read_text())["scenes"]

    if not s2_collection or not s1_collection:
        raise RuntimeError("Empty S2 or S1 collection for landcover multisensor manifest.")

    s2_collection = sorted(s2_collection, key=lambda e: e["datetime"])
    s1_collection = sorted(s1_collection, key=lambda e: e["datetime"])

    def nearest_s1_scene(target_iso: str) -> Dict:
        tgt = _parse_iso(target_iso)
        best, best_dt = None, None
        for s in s1_collection:
            dt = abs((_parse_iso(s["datetime"]) - tgt).total_seconds())
            if best is None or dt < best_dt:
                best, best_dt = s, dt
        return best

    tiles_out: List[Dict] = []

    for s2_scene in s2_collection:
        s1_scene = nearest_s1_scene(s2_scene["datetime"])
        feats_s2 = _read_manifest_features(Path(s2_scene["manifest_path"]))
        feats_s1 = _read_manifest_features(Path(s1_scene["manifest_path"]))
        tree_s1, geoms_s1 = _build_spatial_index(feats_s1)

        for geom_s2, props_s2, s2_path in feats_s2:
            s1_path, _ = _best_overlap(
                target_geom=geom_s2,
                tree=tree_s1,
                geoms=geoms_s1,
                feats=feats_s1,
                iou_min=iou_min,
            )
            if s1_path is None:
                continue

            s2_tile_path = Path(s2_path)
            wc_tile_path = s2_tile_path.with_name(s2_tile_path.stem + "_worldcover_aligned.tif")

            wc_written = _write_aligned_worldcover_label(
                s2_tile_path=s2_tile_path,
                out_label_path=wc_tile_path,
                worldcover_version=worldcover_version,
                worldcover_year=worldcover_year,
            )
            if wc_written is None:
                continue

            with rasterio.open(s2_tile_path) as s2_src:
                H, W = s2_src.height, s2_src.width

            row_off = int(props_s2["row_off"])
            col_off = int(props_s2["col_off"])
            step = int(props_s2.get("stride", props_s2.get("size", 1)))
            step = max(step, 1)
            row_idx = row_off // step
            col_idx = col_off // step

            tiles_out.append({
                "tile_id": f"{s2_scene['scene_id']}_{row_idx}_{col_idx}",
                "s2_path": s2_path,
                "s1_path": s1_path,
                "worldcover_path": str(wc_written),
                "height": int(H),
                "width": int(W),
                "datetime": s2_scene["datetime"],
                "row": row_idx,
                "col": col_idx,
            })

    out_path = Path(s2_collection_manifest_path).parent / "landcover_manifest_multisensor.json"
    out_path.write_text(json.dumps({"tiles": tiles_out}, indent=2))

    if not tiles_out:
        print("[WARN] build_landcover_multisensor_manifest produced 0 tiles.")
    else:
        print(f"[INFO] Landcover multisensor manifest written to {out_path} with {len(tiles_out)} tiles.")
    return out_path