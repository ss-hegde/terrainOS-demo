from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from eintelligence.fusion.kernel_base import FusionBatch
from eintelligence.data_prep.worldcover_labels import (
    worldcover_to_reduced,
    REDUCED_LC_IGNORE,
)


CURL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="YES",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    CPL_VSIL_CURL_NON_CACHED=".tif",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="1000000",
)


@dataclass
class LandCoverTileRecord:
    tile_id: str
    scene_id: str
    s1_path: Path
    s2_path: Path
    worldcover_path: Path
    height: int
    width: int
    datetime: str
    row: int
    col: int
    aoi_id: Optional[str] = None
    job_id: Optional[str] = None
    group_id: Optional[str] = None


class LandCoverDataset(Dataset):
    """
    Tile-level land-cover dataset.

    Expected manifest format:
    {
      "tiles": [
        {
          "tile_id": "...",
          "scene_id": "...",
          "s1_path": "...",
          "s2_path": "...",
          "worldcover_path": "...",
          "height": 256,
          "width": 256,
          "datetime": "...",
          "row": 0,
          "col": 0,
          "aoi_id": "...",      # optional
          "job_id": "...",      # optional
          "group_id": "..."     # optional
        }
      ]
    }

    Yields:
      {
        "fusion_batch": FusionBatch,
        "labels": LongTensor[H, W],
        "valid_mask": BoolTensor[H, W],
      }
    """

    def __init__(
        self,
        manifest_path: str | Path,
        transforms: Optional[Any] = None,
        tile_size: Optional[int] = None,
        normalize_s2: bool = True,
        s2_mean: Tuple[float, ...] = (0.12, 0.14, 0.16, 0.28),
        s2_std: Tuple[float, ...] = (0.08, 0.07, 0.08, 0.10),
        s1_clip_min: Optional[float] = 0.0,
        s1_clip_max: Optional[float] = None,
        s1_log1p: bool = True,
        s1_replace_value: float = 0.0,
    ):
        self.manifest_path = Path(manifest_path)
        self.transforms = transforms
        self.records: List[LandCoverTileRecord] = []

        self.normalize_s2 = normalize_s2
        self.s2_mean = np.array(s2_mean, dtype=np.float32)[:, None, None]
        self.s2_std = np.array(s2_std, dtype=np.float32)[:, None, None]

        self.s1_clip_min = s1_clip_min
        self.s1_clip_max = s1_clip_max
        self.s1_log1p = s1_log1p
        self.s1_replace_value = float(s1_replace_value)

        data = json.loads(self.manifest_path.read_text())
        entries = data.get("tiles")
        if entries is None:
            raise RuntimeError(
                f"Unsupported landcover manifest format in {self.manifest_path}: expected top-level key 'tiles'."
            )

        for e in entries:
            h = int(e["height"])
            w = int(e["width"])
            if tile_size is not None and (h != tile_size or w != tile_size):
                continue

            self.records.append(
                LandCoverTileRecord(
                    tile_id=e["tile_id"],
                    scene_id=e.get("scene_id", ""),
                    s1_path=Path(e["s1_path"]),
                    s2_path=Path(e["s2_path"]),
                    worldcover_path=Path(e["worldcover_path"]),
                    height=h,
                    width=w,
                    datetime=e["datetime"],
                    row=int(e["row"]),
                    col=int(e["col"]),
                    aoi_id=e.get("aoi_id"),
                    job_id=e.get("job_id"),
                    group_id=e.get("group_id"),
                )
            )

        if not self.records:
            raise RuntimeError(
                f"No valid tile records found in {self.manifest_path}. "
                f"Check tile_size filtering and manifest contents."
            )

    def __len__(self) -> int:
        return len(self.records)

    def _read_raster_masked(self, path: Path) -> Tuple[np.ma.MaskedArray, np.ndarray, Any]:
        with rasterio.Env(**CURL_ENV):
            with rasterio.open(path) as src:
                arr = src.read(masked=True)
                dataset_mask = src.dataset_mask() > 0
                nodata = src.nodata
        return arr, dataset_mask, nodata

    @staticmethod
    def _masked_to_float(arr: np.ma.MaskedArray, fill_value: float = np.nan) -> np.ndarray:
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        return arr.filled(fill_value).astype(np.float32, copy=False)

    def _prepare_s2(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        arr_ma, dataset_valid, _ = self._read_raster_masked(path)
        arr = self._masked_to_float(arr_ma, fill_value=np.nan)

        arr = arr * (1.0 / 10000.0)

        if self.normalize_s2:
            c = min(arr.shape[0], len(self.s2_mean))
            arr[:c] = (arr[:c] - self.s2_mean[:c]) / (self.s2_std[:c] + 1e-6)

        arr[~np.isfinite(arr)] = 0.0
        valid = dataset_valid.astype(bool)

        return arr.astype(np.float32, copy=False), valid

    def _prepare_s1(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        arr_ma, dataset_valid, _ = self._read_raster_masked(path)
        arr = self._masked_to_float(arr_ma, fill_value=np.nan)

        arr = arr.astype(np.float32, copy=False)

        # mark obvious junk as invalid early
        arr[~np.isfinite(arr)] = np.nan
        arr[(arr <= -1e10) | (arr >= 1e10)] = np.nan

        # optional pre-log clipping
        if self.s1_clip_min is not None or self.s1_clip_max is not None:
            lo = self.s1_clip_min if self.s1_clip_min is not None else -np.inf
            hi = self.s1_clip_max if self.s1_clip_max is not None else np.inf
            arr = np.clip(arr, lo, hi)

        # stabilize positive-heavy S1 magnitudes
        # if inputs are linear-power-like, log1p compresses the dynamic range strongly
        if self.s1_log1p:
            arr = np.log1p(np.clip(arr, 0.0, None))
        else:
            arr = np.clip(arr, 0.0, None)

        # optional post-log clipping to keep a sane range
        arr = np.clip(arr, 0.0, 25.0)

        valid = dataset_valid.astype(bool) & np.isfinite(arr).all(axis=0)

        arr = np.nan_to_num(
            arr,
            nan=self.s1_replace_value,
            posinf=self.s1_replace_value,
            neginf=self.s1_replace_value,
        ).astype(np.float32, copy=False)

        if not np.isfinite(arr).all():
            raise RuntimeError(f"Non-finite S1 after preprocessing: {path}")

        if float(np.abs(arr).max()) > 100.0:
            raise RuntimeError(
                f"S1 still too large after preprocessing: {path} max={float(np.abs(arr).max())}"
            )

        return arr, valid

    def _prepare_labels(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        with rasterio.open(path) as src:
            wc = src.read(1)

        labels_np = worldcover_to_reduced(wc).astype(np.uint8, copy=False)
        valid_mask_np = labels_np != REDUCED_LC_IGNORE
        return labels_np, valid_mask_np

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]

        s1, s1_valid = self._prepare_s1(rec.s1_path)
        s2, s2_valid = self._prepare_s2(rec.s2_path)
        labels_np, label_valid = self._prepare_labels(rec.worldcover_path)

        if s1.shape[1:] != s2.shape[1:]:
            raise RuntimeError(
                f"S1/S2 shape mismatch for tile {rec.tile_id}: {s1.shape} vs {s2.shape}"
            )
        if labels_np.shape != s2.shape[1:]:
            raise RuntimeError(
                f"Label/S2 shape mismatch for tile {rec.tile_id}: {labels_np.shape} vs {s2.shape[1:]}"
            )

        valid_mask_np = label_valid & s1_valid & s2_valid

        imagery = {
            "s1": torch.from_numpy(s1).float(),
            "s2": torch.from_numpy(s2).float(),
        }
        masks = {
            "s1": torch.from_numpy(s1_valid[None, ...]).bool(),
            "s2": torch.from_numpy(s2_valid[None, ...]).bool(),
        }
        meta = {
            "tile_id": rec.tile_id,
            "scene_id": rec.scene_id,
            "datetime": rec.datetime,
            "row": rec.row,
            "col": rec.col,
            "height": rec.height,
            "width": rec.width,
            "aoi_id": rec.aoi_id,
            "job_id": rec.job_id,
            "group_id": rec.group_id,
            "s1_path": str(rec.s1_path),
            "s2_path": str(rec.s2_path),
            "worldcover_path": str(rec.worldcover_path),
        }

        fusion_batch = FusionBatch(
            imagery=imagery,
            masks=masks,
            meta=meta,
        )

        sample = {
            "fusion_batch": fusion_batch,
            "labels": torch.from_numpy(labels_np).long(),
            "valid_mask": torch.from_numpy(valid_mask_np).bool(),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample