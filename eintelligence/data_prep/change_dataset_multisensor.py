# eintelligence/data_prep/change_dataset_multisensor.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.env import Env

CURL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="YES",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    CPL_VSIL_CURL_NON_CACHED=".tif",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="1000000",
)

def _read(path: Path) -> np.ndarray:
    with Env(**CURL_ENV):
        with rasterio.open(path) as src:
            return src.read(), src.transform, src.crs

def _s2_to_float01(arr: np.ndarray) -> np.ndarray:
    return (arr.astype(np.float32) * (1.0 / 10000.0))

def _compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    denom = (nir + red)
    ndvi = (nir - red) / np.where(denom != 0.0, denom, np.finfo(np.float32).eps)
    return ndvi.astype(np.float32)

class MultiSensorChangeDataset(Dataset):
    """
    Returns dict for SSL4EO-based change model:
    {
      "t0": {"s1": Tensor[C1,H,W], "s2": Tensor[C2,H,W]},
      "t1": {"s1": Tensor[C1,H,W], "s2": Tensor[C2,H,W]},
      "target": Tensor[1,H,W],   # NDVI-drop weak label (optional)
      "meta": {...}
    }
    """
    def __init__(
        self,
        pairs_manifest_path: Path,
        ndvi_drop_threshold: float = 0.2,
        tile_size: int = 256,
        normalize_s2: bool = True,
        s2_mean=(0.12,0.14,0.16,0.28),
        s2_std=(0.08,0.07,0.08,0.10),
    ):
        self.pairs = json.loads(Path(pairs_manifest_path).read_text())["pairs"]
        self.ndvi_drop_threshold = ndvi_drop_threshold
        self.tile_size = tile_size
        self.normalize_s2 = normalize_s2
        self.s2_mean = np.array(s2_mean, np.float32)[:,None,None]
        self.s2_std  = np.array(s2_std,  np.float32)[:,None,None]

    def __len__(self): return len(self.pairs)

    @staticmethod
    def _pad(arr, Ht, Wt):
        C,H,W = arr.shape
        if H==Ht and W==Wt: return arr
        return np.pad(arr, ((0,0),(0,max(0,Ht-H)),(0,max(0,Wt-W))), constant_values=0)

    def _prepare_s2(self, path: str) -> Tuple[np.ndarray, Dict]:
        arr, transform, crs = _read(Path(path))
        arr = _s2_to_float01(arr)
        if self.normalize_s2:
            C = arr.shape[0]
            arr[:C] = (arr[:C] - self.s2_mean) / (self.s2_std + 1e-6)
        meta = {"transform": transform, "crs": str(crs)}
        return arr, meta

    def _prepare_s1(self, path: str) -> Tuple[np.ndarray, Dict]:
        arr, transform, crs = _read(Path(path))
        # Assuming tiles were written as float32 linear sigma0 by your tiler.
        arr = arr.astype(np.float32)
        meta = {"transform": transform, "crs": str(crs)}
        return arr, meta

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        s2_t0, m0 = self._prepare_s2(pair["t0"]["s2"])
        s2_t1, m1 = self._prepare_s2(pair["t1"]["s2"])
        s1_t0, _  = self._prepare_s1(pair["t0"]["s1"])
        s1_t1, _  = self._prepare_s1(pair["t1"]["s1"])

        # weak labels via NDVI drop (S2: B04=red idx 2, B08=nir idx 3)
        ndvi0 = _compute_ndvi(s2_t0[2], s2_t0[3])
        ndvi1 = _compute_ndvi(s2_t1[2], s2_t1[3])
        target = ((ndvi0 - ndvi1) > self.ndvi_drop_threshold).astype(np.float32)[None,...]

        # pad to fixed size
        Ht = Wt = self.tile_size
        s2_t0 = self._pad(s2_t0, Ht, Wt); s2_t1 = self._pad(s2_t1, Ht, Wt)
        s1_t0 = self._pad(s1_t0, Ht, Wt); s1_t1 = self._pad(s1_t1, Ht, Wt)
        target = self._pad(target, Ht, Wt)

        batch = {
            "t0": {"s1": torch.from_numpy(s1_t0), "s2": torch.from_numpy(s2_t0)},
            "t1": {"s1": torch.from_numpy(s1_t1), "s2": torch.from_numpy(s2_t1)},
            "target": torch.from_numpy(target),
            "meta": {
                "row": pair["row"], "col": pair["col"],
                "scene_ids": pair["scene_ids"],
                "transform": m0["transform"], "crs": m0["crs"]
            }
        }
        return batch
