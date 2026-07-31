from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Optional

import planetary_computer

from .tiler_streaming import tile_stac_item_to_cogs
from .registry_manager import build_scene_record, append_jsonl, write_json, safe_id


def _collection_payload(entries):
    return {"scenes": entries}


def _build_collection(
    *,
    items: Sequence,
    out_dir: Path,
    sensor: str,
    bands,
    tile_size: int,
    stride: int,
    aoi_geojson=None,
    aoi_id: str = "aoi",
    job_id: str = "job",
    registry_path: Optional[Path] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    registry_rows = []
    for item in items:
        scene_id = safe_id(item.id)
        scene_dir = out_dir / scene_id
        tiles_dir = scene_dir / f"tiles_{sensor.lower()}"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        signed_item = planetary_computer.sign(item)
        manifest_path = tile_stac_item_to_cogs(
            signed_item,
            sensor=sensor,
            bands=bands,
            out_dir=tiles_dir,
            tile_size=tile_size,
            stride=stride,
            min_valid_fraction=0.3,
            web_optimized=False,
            aoi_geojson=aoi_geojson,
        )
        entries.append(
            {
                "scene_id": scene_id,
                "datetime": item.properties.get("datetime"),
                "mgrs_tile": item.properties.get("s2:mgrs_tile"),
                "manifest_path": str(manifest_path),
            }
        )
        registry_rows.append(
            build_scene_record(
                sensor=sensor,
                item=item,
                aoi_id=aoi_id,
                job_id=job_id,
                collection_manifest_path=out_dir / f"collection_manifest_{sensor.lower()}.json",
                tile_manifest_path=manifest_path,
                source_dir=scene_dir,
                extra={"bands": list(bands), "tile_size": tile_size, "stride": stride},
            )
        )
    collection_path = out_dir / f"collection_manifest_{sensor.lower()}.json"
    write_json(collection_path, _collection_payload(entries))
    if registry_path is not None and registry_rows:
        append_jsonl(registry_path, registry_rows)
    return collection_path


def build_s2_data_collection(
    items: Sequence,
    out_dir: Path,
    bands=("B02", "B03", "B04", "B08"),
    tile_size: int = 512,
    stride: int = 256,
    aoi_geojson=None,
    aoi_id: str = "aoi",
    job_id: str = "job",
    registry_path: Optional[Path] = None,
) -> Path:
    return _build_collection(
        items=items,
        out_dir=out_dir,
        sensor="S2",
        bands=bands,
        tile_size=tile_size,
        stride=stride,
        aoi_geojson=aoi_geojson,
        aoi_id=aoi_id,
        job_id=job_id,
        registry_path=registry_path,
    )


def build_s1_data_collection(
    items: Sequence,
    out_dir: Path,
    bands=("VV", "VH"),
    tile_size: int = 512,
    stride: int = 256,
    aoi_geojson=None,
    aoi_id: str = "aoi",
    job_id: str = "job",
    registry_path: Optional[Path] = None,
) -> Path:
    return _build_collection(
        items=items,
        out_dir=out_dir,
        sensor="S1",
        bands=bands,
        tile_size=tile_size,
        stride=stride,
        aoi_geojson=aoi_geojson,
        aoi_id=aoi_id,
        job_id=job_id,
        registry_path=registry_path,
    )