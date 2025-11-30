# eintelligence/data_prep/flood_s1_dataset.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

class FloodS1ChangeDataset(Dataset):
    """
    Map-style dataset for S1 flood change detection using paired tiles.

    Expects pairs_manifest.json with entries like:
      {"t1_path": ".../rXXXX_cXXXX.tif", "t2_path": ".../rXXXX_cXXXX.tif", "s1_id": "...", "s2_id": "...", ...}

    Returns: (xb, yb, meta)
      xb   : {"t0": FloatTensor[2,H,W], "t1": FloatTensor[2,H,W]} where channels are [VV, VH] in normalized dB
      yb   : FloatTensor[1,H,W] weak label (0/1) if use_weak_labels else zeros
      meta : {"t1_path": str, "t2_path": str, "s1_id": str, "s2_id": str}
    """

    def __init__(self,
                 pairs_manifest_path: Path | str,
                 tile_size: Optional[int] = None,
                 use_weak_labels: bool = True,
                 vv_mean_db: float = -12.0,
                 vv_std_db: float  = 6.0,
                 vh_mean_db: float = -18.0,
                 vh_std_db: float  = 6.0,
                 weak_thr_db: float = 2.5):
        """
        Args:
            tile_size: if set, pad/crop to this size (optional, usually tiles are already uniform)
            use_weak_labels: if True, produce a simple weak flood mask from backscatter decrease
            *_mean_db/std_db: normalization constants for S1 channels in dB
            weak_thr_db: t0 - t1 drop (in dB) threshold for weak flood label (higher = stricter)
        """
        self.pairs_path = Path(pairs_manifest_path)
        data = json.loads(self.pairs_path.read_text())
        self.pairs: List[Dict[str, Any]] = data.get("pairs", [])
        self.tile_size = tile_size
        self.use_weak = use_weak_labels

        # normalization constants
        self.m = np.array([vv_mean_db, vh_mean_db], np.float32)[:, None, None]
        self.s = np.array([vv_std_db, vh_std_db], np.float32)[:, None, None]
        self.weak_thr_db = float(weak_thr_db)

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _read_s1(path: str) -> np.ndarray:
        # returns float32 linear backscatter [2,H,W] (VV,VH)
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
        return arr

    @staticmethod
    def _to_db(x: np.ndarray) -> np.ndarray:
        # x linear -> dB, safe log
        return 10.0 * np.log10(np.clip(x, 1e-6, None))

    def _norm_db(self, x_db: np.ndarray) -> np.ndarray:
        # channel-wise normalize using fixed mean/std in dB
        return (x_db - self.m) / (self.s + 1e-6)

    def _maybe_resize(self, x: np.ndarray, Ht: int, Wt: int) -> np.ndarray:
        # simple pad/crop to [*,Ht,Wt]
        C, H, W = x.shape
        if H == Ht and W == Wt:
            return x
        out = np.zeros((C, Ht, Wt), dtype=x.dtype)
        hh = min(H, Ht); ww = min(W, Wt)
        out[:, :hh, :ww] = x[:, :hh, :ww]
        return out

    def _weak_label(self, t0_db: np.ndarray, t1_db: np.ndarray) -> np.ndarray:
        """
        Very simple weak label: flooded areas get darker in VV (sometimes VH).
        Label = 1 if (VV_drop_db >= thr) OR (VH_drop_db >= thr/2)
        """
        vv_drop = t0_db[0] - t1_db[0]  # positive if t1 darker
        vh_drop = t0_db[1] - t1_db[1]
        mask = (vv_drop >= self.weak_thr_db) | (vh_drop >= (0.5 * self.weak_thr_db))
        return mask.astype(np.float32)[None, ...]  # [1,H,W]

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        t0_path = p["t1_path"]  # previous
        t1_path = p["t2_path"]  # next

        A = self._read_s1(t0_path)  # [2,H,W] linear
        B = self._read_s1(t1_path)  # [2,H,W] linear

        # to dB then normalize
        A_db = self._to_db(A)
        B_db = self._to_db(B)
        A_n  = self._norm_db(A_db)
        B_n  = self._norm_db(B_db)

        if self.tile_size is not None:
            A_n = self._maybe_resize(A_n, self.tile_size, self.tile_size)
            B_n = self._maybe_resize(B_n, self.tile_size, self.tile_size)

        if self.use_weak:
            y = self._weak_label(A_db, B_db)
            if self.tile_size is not None:
                y = self._maybe_resize(y, self.tile_size, self.tile_size)
        else:
            H, W = B_n.shape[1], B_n.shape[2]
            y = np.zeros((1, H, W), np.float32)

        xb = {
            "t0": torch.from_numpy(A_n),  # [2,H,W]
            "t1": torch.from_numpy(B_n),  # [2,H,W]
        }
        yb = torch.from_numpy(y)          # [1,H,W]
        meta = {
            "t1_path": t0_path,
            "t2_path": t1_path,
            "s1_id": p.get("s1_id"),
            "s2_id": p.get("s2_id"),
        }
        return xb, yb, meta


