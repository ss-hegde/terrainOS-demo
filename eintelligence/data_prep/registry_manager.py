from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _as_dict(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "to_dict"):
        return item.to_dict()
    raise TypeError(f"Unsupported record type: {type(item)!r}")


def _read_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return default


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return path


def safe_id(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace(" ", "_")


def build_scene_record(
    *,
    sensor: str,
    item: Any,
    aoi_id: str,
    job_id: str,
    collection_manifest_path: Path,
    tile_manifest_path: Path,
    source_dir: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    item_d = _as_dict(item)
    props = item_d.get("properties", {})
    record = {
        "sensor": sensor.upper(),
        "scene_id": safe_id(item_d.get("id", "")),
        "datetime": props.get("datetime"),
        "mgrs_tile": props.get("s2:mgrs_tile"),
        "aoi_id": aoi_id,
        "job_id": job_id,
        "group_id": f'{aoi_id}__{props.get("s2:mgrs_tile", "na")}',
        "collection_manifest_path": str(collection_manifest_path),
        "tile_manifest_path": str(tile_manifest_path),
        "source_dir": str(source_dir),
    }
    if extra:
        record.update(extra)
    return record


def index_scene_registry(registry_path: Path) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(registry_path)
    return {r["scene_id"]: r for r in rows if "scene_id" in r}


def filter_registry(rows: Sequence[Dict[str, Any]], **filters: Any) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        ok = True
        for key, value in filters.items():
            if value is None:
                continue
            if row.get(key) != value:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def grouped_split(
    records: Sequence[Dict[str, Any]],
    group_key: str = "group_id",
    seed: int = 42,
    fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Dict[str, List[Dict[str, Any]]]:
    import random

    rnd = random.Random(seed)
    groups = {}
    for r in records:
        g = r.get(group_key)
        groups.setdefault(g, []).append(r)

    group_ids = list(groups)
    rnd.shuffle(group_ids)
    n = len(group_ids)

    n_train = max(1, int(n * fractions[0])) if n else 0
    n_val = max(1, int(n * fractions[1])) if n >= 3 else max(0, n - n_train)

    train_groups = set(group_ids[:n_train])
    val_groups = set(group_ids[n_train:n_train + n_val])
    test_groups = set(group_ids[n_train + n_val:])

    return {
        "train": [r for g in train_groups for r in groups[g]],
        "val": [r for g in val_groups for r in groups[g]],
        "test": [r for g in test_groups for r in groups[g]],
    }


def save_splits(path: Path, splits: Dict[str, List[Dict[str, Any]]], group_key: str = "group_id") -> Path:
    payload = {
        "group_key": group_key,
        "splits": {
            name: [r["scene_id"] for r in rows] for name, rows in splits.items()
        },
    }
    return write_json(path, payload)


def load_splits(path: Path) -> Dict[str, List[str]]:
    data = _read_json(path, {"splits": {}})
    return data.get("splits", {})