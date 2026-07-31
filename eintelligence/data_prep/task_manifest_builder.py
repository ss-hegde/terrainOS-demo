from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

from .registry_manager import load_jsonl, grouped_split, save_splits, write_json


def _load_collection(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    return data.get("scenes", [])


def _index_registry_by_scene_and_sensor(
    registry_rows: List[Dict[str, Any]],
    *,
    aoi_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in registry_rows:
        if aoi_id is not None and row.get("aoi_id") != aoi_id:
            continue
        if job_id is not None and row.get("job_id") != job_id:
            continue
        scene_id = row.get("scene_id")
        sensor = row.get("sensor")
        if scene_id is None or sensor is None:
            continue
        out[(scene_id, sensor.upper())] = row
    return out


def _index_collection_by_scene(collection_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row["scene_id"]: row for row in collection_rows if "scene_id" in row}


def build_landcover_task_manifest_from_registry(
    *,
    registry_path: Path,
    s1_collection_path: Path,
    s2_collection_path: Path,
    out_dir: Path,
    worldcover_version: str = "v200",
    worldcover_year: str = "2021",
    aoi_id: Optional[str] = None,
    job_id: Optional[str] = None,
    group_key: str = "group_id",
    split_seed: int = 42,
) -> Path:
    """
    Build a task-level land cover manifest from the scene registry.

    Output samples are scene-level records that point to:
      - S2 tile manifest
      - S1 tile manifest
      - WorldCover tile manifest

    The dataset will later expand these into aligned tile triplets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = load_jsonl(registry_path)
    s1_collection_rows = _load_collection(s1_collection_path)
    s2_collection_rows = _load_collection(s2_collection_path)

    reg_index = _index_registry_by_scene_and_sensor(
        registry_rows,
        aoi_id=aoi_id,
        job_id=job_id,
    )
    s1_index = _index_collection_by_scene(s1_collection_rows)
    s2_index = _index_collection_by_scene(s2_collection_rows)

    samples: List[Dict[str, Any]] = []

    for scene_id, s2_row in s2_index.items():
        reg_s2 = reg_index.get((scene_id, "S2"))
        if reg_s2 is None:
            continue

        reg_s1 = reg_index.get((scene_id, "S1"))

        s2_tile_manifest_path = reg_s2.get("tile_manifest_path")
        s1_tile_manifest_path = reg_s1.get("tile_manifest_path") if reg_s1 else None

        # Placeholder convention:
        # we assume WorldCover tiles for this scene will be materialized and stored
        # next to the S2 scene assets under a predictable folder.
        #
        # Example:
        #   .../S2/<scene_id>/tiles_s2/manifest.json
        # -> .../S2/<scene_id>/tiles_worldcover/manifest.json
        wc_tile_manifest_path = None
        if s2_tile_manifest_path is not None:
            s2_manifest_path_obj = Path(s2_tile_manifest_path)
            wc_tile_manifest_path = str(
                s2_manifest_path_obj.parent.parent / "tiles_worldcover" / "manifest.json"
            )

        sample = {
            "scene_id": scene_id,
            "sensor_mode": "s1s2" if s1_tile_manifest_path is not None else "s2",
            "aoi_id": reg_s2.get("aoi_id"),
            "job_id": reg_s2.get("job_id"),
            "group_id": reg_s2.get("group_id"),
            "datetime": reg_s2.get("datetime"),
            "mgrs_tile": reg_s2.get("mgrs_tile"),

            "s2_collection_manifest_path": str(s2_collection_path),
            "s1_collection_manifest_path": str(s1_collection_path),

            "s2_scene_manifest_path": s2_row.get("manifest_path"),
            "s1_scene_manifest_path": s1_index.get(scene_id, {}).get("manifest_path"),

            "s2_tile_manifest_path": s2_tile_manifest_path,
            "s1_tile_manifest_path": s1_tile_manifest_path,
            "worldcover_tile_manifest_path": wc_tile_manifest_path,

            "worldcover_version": worldcover_version,
            "worldcover_year": worldcover_year,
        }

        samples.append(sample)

    manifest = {
        "task": "landcover",
        "worldcover_version": worldcover_version,
        "worldcover_year": worldcover_year,
        "group_key": group_key,
        "samples": samples,
    }

    task_manifest_path = out_dir / "landcover_task_manifest.json"
    write_json(task_manifest_path, manifest)

    splits = grouped_split(samples, group_key=group_key, seed=split_seed)
    save_splits(out_dir / "landcover_splits.json", splits, group_key=group_key)

    return task_manifest_path