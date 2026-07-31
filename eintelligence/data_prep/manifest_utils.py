# eintelligence/data_prep/manifest_utils.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json


def load_manifest_records(
    manifest_path: str | Path,
    *,
    record_key: str,
) -> List[Dict[str, Any]]:
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())

    records = data.get(record_key)
    if records is None:
        raise RuntimeError(
            f"Manifest missing top-level key '{record_key}': {manifest_path}"
        )
    if not isinstance(records, list):
        raise RuntimeError(
            f"Manifest key '{record_key}' must be a list: {manifest_path}"
        )
    return records


def write_manifest_records(
    out_path: str | Path,
    *,
    record_key: str,
    records: Sequence[Dict[str, Any]],
    task_name: Optional[str] = None,
    extra_top_level: Optional[Dict[str, Any]] = None,
) -> Path:
    out_path = Path(out_path)

    payload: Dict[str, Any] = {}
    if task_name is not None:
        payload["task"] = task_name
    if extra_top_level:
        payload.update(extra_top_level)

    payload[record_key] = list(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def namespace_record(
    record: Dict[str, Any],
    *,
    namespace: Optional[str] = None,
    fields_to_prefix: Sequence[str] = ("group_id",),
    set_default_aoi_id: bool = False,
    aoi_field: str = "aoi_id",
) -> Dict[str, Any]:
    out = dict(record)

    prefix = (namespace or "").strip()
    if not prefix:
        return out

    for field in fields_to_prefix:
        value = out.get(field)
        if value is not None and value != "":
            out[field] = f"{prefix}:{value}"

    if set_default_aoi_id and not out.get(aoi_field):
        out[aoi_field] = prefix

    return out


def merge_record_manifests(
    manifest_paths: Iterable[str | Path],
    out_path: str | Path,
    *,
    record_key: str,
    task_name: Optional[str] = None,
    extra_top_level: Optional[Dict[str, Any]] = None,
    namespaces: Optional[Iterable[Optional[str]]] = None,
    fields_to_prefix: Sequence[str] = ("group_id",),
    set_default_aoi_id: bool = False,
    aoi_field: str = "aoi_id",
    deduplicate_on: Optional[str] = None,
    sort_by: Optional[Sequence[str]] = None,
) -> Path:
    manifest_paths = [Path(p) for p in manifest_paths]
    if not manifest_paths:
        raise RuntimeError("No manifest paths were provided.")

    if namespaces is None:
        namespaces = [None] * len(manifest_paths)
    else:
        namespaces = list(namespaces)

    if len(manifest_paths) != len(namespaces):
        raise RuntimeError("namespaces must have the same length as manifest_paths.")

    merged: List[Dict[str, Any]] = []
    seen = set()

    for manifest_path, namespace in zip(manifest_paths, namespaces):
        records = load_manifest_records(manifest_path, record_key=record_key)

        for record in records:
            rec = namespace_record(
                record,
                namespace=namespace,
                fields_to_prefix=fields_to_prefix,
                set_default_aoi_id=set_default_aoi_id,
                aoi_field=aoi_field,
            )

            if deduplicate_on is not None:
                dedup_value = rec.get(deduplicate_on)
                if dedup_value is None:
                    raise RuntimeError(
                        f"Record in {manifest_path} missing deduplicate key '{deduplicate_on}'."
                    )
                if dedup_value in seen:
                    continue
                seen.add(dedup_value)

            merged.append(rec)

    if sort_by:
        merged.sort(
            key=lambda r: tuple("" if r.get(k) is None else str(r.get(k)) for k in sort_by)
        )

    return write_manifest_records(
        out_path,
        record_key=record_key,
        records=merged,
        task_name=task_name,
        extra_top_level=extra_top_level,
    )