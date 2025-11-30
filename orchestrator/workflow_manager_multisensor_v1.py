from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Sequence, Tuple, Literal
import json
import os

import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----- Data prep -----
from eintelligence.data_prep.fetch_multi_data import search_s1_items, search_s2_items
from eintelligence.data_prep.build_data_collection import (
    build_s1_data_collection, build_s2_data_collection
)
from eintelligence.data_prep.temporal_pairing import (
    build_temporal_pairs,                   # single-sensor (S2 or S1)
    build_temporal_pairs_multisensor,      # S1+S2
    build_temporal_pairs_relaxed_s1        # relaxed S1 pairing
)
from eintelligence.data_prep.resume import (
    discover_s1_collection, discover_s2_collection
)

# Datasets
from eintelligence.data_prep.change_dataset import DeforestationChangeDataset  # S2 single-sensor dataset
from eintelligence.data_prep.change_dataset_multisensor import MultiSensorChangeDataset
from eintelligence.data_prep.collate import collate_change

# Backbone/Adapter/Analytics
from eintelligence.backbone.ssl4eo_lite_backbone import SSL4EOLiteConfig, MultiSensorSSL4EOLiteBackbone
from eintelligence.adapters.deforestation_change import DeforestationChangeAdapter
from eintelligence.analytics.deforestation import deforestation_summary

# ---------- small utils copied from your single-sensor ----------
def _save_mask_like(src_tile_path: Path, mask_uint8: np.ndarray, out_path: Path) -> None:
    with rasterio.open(src_tile_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8", compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask_uint8, 1)

def _pick_rgb_indices(band_names: Sequence[str]) -> Tuple[int, int, int]:
    band_map = {name: idx for idx, name in enumerate(band_names)}
    for need in ("B04", "B03", "B02"):
        if need not in band_map:
            raise ValueError(f"Band {need} not in band names: {band_names}")
    return band_map["B04"], band_map["B03"], band_map["B02"]

def _load_rgb_reflectance(tile_path: Path, band_names: Sequence[str]) -> np.ndarray:
    with rasterio.open(tile_path) as src:
        r_idx, g_idx, b_idx = _pick_rgb_indices(band_names)
        R = src.read(r_idx + 1).astype(np.float32) / 10000.0
        G = src.read(g_idx + 1).astype(np.float32) / 10000.0
        B = src.read(b_idx + 1).astype(np.float32) / 10000.0
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)

def _save_quicklook_png(rgb: np.ndarray, mask_uint8: np.ndarray, out_png: Path, alpha: float = 0.3):
    h, w, _ = rgb.shape
    plt.figure(figsize=(max(4, w/512), max(4, h/512)), dpi=150)
    plt.imshow(rgb)
    m = (mask_uint8 > 0).astype(np.float32)
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 1] = 1.0
    overlay[..., 3] = m * alpha
    plt.imshow(overlay)
    plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

# ---------- Configs ----------

SensorMode = Literal["s1", "s2", "s1s2"]

@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 5
    val_fraction: float = 0.2
    amp: bool = True

@dataclass
class TilingConfigMS:
    bands_s2: Tuple[str, ...] = ("B02","B03","B04","B08")  # S2 10m
    bands_s1: Tuple[str, ...] = ("VV","VH")                # S1 GRD/RTC
    tile_size: int = 256
    stride: Optional[int] = 256
    max_cloud: int = 20
    same_mgrs_tile: bool = True
    sensor_mode: SensorMode = "s1s2"   # <-- the switch

# ---------- Trainer (dict-style batches) ----------

class _TrainerMS:
    def __init__(self, device: torch.device, cfg: TrainingConfig):
        self.device = device
        self.cfg = cfg
        self.scaler = GradScaler(enabled=(device.type == "cuda" and cfg.amp))
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def _loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # >>> NEW: upsample logits to target size if needed
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2,3))
        dice = 1 - (2*inter + eps) / (probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps)
        return bce + dice.mean()


    def _step(self, batch: Dict, backbone: MultiSensorSSL4EOLiteBackbone, adapter: DeforestationChangeAdapter,
              optimizer=None, train=True) -> float:
        target = batch["target"].to(self.device).float()
        for t in ("t0","t1"):
            for m in ("s1","s2"):
                if m in batch[t]:
                    batch[t][m] = batch[t][m].to(self.device, non_blocking=True)

        with torch.set_grad_enabled(train), autocast(enabled=(self.device.type=="cuda" and self.cfg.amp)):
            out = adapter(batch, backbone)
            logits = out["logits"]
            # 🔧 ensure logits match target HxW
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss = self._loss(logits, target)
            # loss = self._loss(out["logits"], target)

        if train:
            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

        return float(loss.item())

    def fit(self, dataset, backbone: MultiSensorSSL4EOLiteBackbone,
            adapter: DeforestationChangeAdapter, ckpt_path: Path, collate_fn=None) -> None:
        n_val = int(len(dataset) * self.cfg.val_fraction)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True,
                                  num_workers=self.cfg.num_workers, pin_memory=True,
                                  persistent_workers=self.cfg.num_workers>0,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=False,
                                num_workers=self.cfg.num_workers, pin_memory=True,
                                persistent_workers=self.cfg.num_workers>0,
                                collate_fn=collate_fn)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, list(adapter.parameters())),
            lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        best = float("inf")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for e in range(self.cfg.num_epochs):
            # train
            tr_total = 0.0
            backbone.eval(); adapter.train()
            for batch in train_loader:
                tr_total += self._step(batch, backbone, adapter, optimizer, train=True) * batch["target"].size(0)
            tr_loss = tr_total / len(train_loader.dataset)

            # val
            vl_total = 0.0
            backbone.eval(); adapter.eval()
            with torch.no_grad():
                for batch in val_loader:
                    vl_total += self._step(batch, backbone, adapter, optimizer=None, train=False) * batch["target"].size(0)
            vl_loss = vl_total / len(val_loader.dataset)

            print(f"[epoch {e:02d}] train={tr_loss:.4f}  val={vl_loss:.4f}")
            if vl_loss < best:
                best = vl_loss
                torch.save({"adapter": adapter.state_dict()}, ckpt_path)
                print(f"  ↳ saved: {ckpt_path}")

# ---------- Simple wrappers to get dict-style batches per mode ----------

class S2DictDataset(torch.utils.data.Dataset):
    """
    Wraps your DeforestationChangeDataset so the adapter sees a dict:
      {"t0":{"s2":...}, "t1":{"s2":...}, "target":..., "meta":...}
    """
    def __init__(self, base: DeforestationChangeDataset):
        self.base = base

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        x, y, meta = self.base[i]          # x: [2C,H,W], y: [1,H,W]
        C = x.shape[0] // 2
        t0 = x[:C]                         # [C,H,W]
        t1 = x[C:]
        return {
            "t0": {"s2": t0},
            "t1": {"s2": t1},
            "target": y,
            "meta": meta
        }

def _to_device_batch(batch: Dict, device: torch.device) -> Dict:
    out = {"t0":{}, "t1":{}, "meta": batch.get("meta")}
    for t in ("t0","t1"):
        for m in ("s1","s2"):
            if m in batch[t]:
                out[t][m] = batch[t][m].to(device, non_blocking=True)
    out["target"] = batch["target"].to(device, non_blocking=True)
    return out

# ---------- Workflow with sensor switch ----------

class DeforestationWorkflowMS:
    """
    sensor_mode:
      - "s2"    : S2-only (uses S2 dataset + SSL4EO S2 branch; train+infer supported)
      - "s1"    : S1-only (uses S1 tiles; inference supported; training requires labels)
      - "s1s2"  : Multi-sensor (train+infer using MultiSensorChangeDataset)
    """

    def __init__(self,
                 project_root: Path | str,
                 tiling_cfg: TilingConfigMS = TilingConfigMS(),
                 train_cfg: TrainingConfig = TrainingConfig(),
                 skip_to_pairing: bool = False):
        self.root = Path(project_root)
        self.tcfg = tiling_cfg
        self.skip_to_pairing = skip_to_pairing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = _TrainerMS(self.device, train_cfg)

        # Backbone branches depend on mode
        mode = self.tcfg.sensor_mode
        s1_cfg = SSL4EOLiteConfig(in_ch=len(self.tcfg.bands_s1), freeze=True, state_dict=None) if mode in ("s1","s1s2") else None
        s2_cfg = SSL4EOLiteConfig(in_ch=len(self.tcfg.bands_s2), freeze=True, state_dict=None) if mode in ("s2","s1s2") else None
        self.backbone = MultiSensorSSL4EOLiteBackbone(s1_cfg, s2_cfg).to(self.device)

        self.adapter = DeforestationChangeAdapter(c_backbone=512, c_align=256).to(self.device)

    # ---- Build / Resume ----
    def build_data(self, aoi_geojson: Dict[str, Any], start: str, end: str, region_name: str) -> Path:
        mode = self.tcfg.sensor_mode
        region_dir = self.root / "data" / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        if self.skip_to_pairing:
            # s2_coll = discover_s2_collection(region_dir / ("S2" if mode!="s2" else "")) if mode in ("s2","s1s2") else None
            # s1_coll = discover_s1_collection(region_dir / ("S1" if mode!="s1" else "")) if mode in ("s1","s1s2") else None
            s2_coll = discover_s2_collection(region_dir / ("S2")) if mode in ("s2","s1s2") else None
            s1_coll = discover_s1_collection(region_dir / ("S1")) if mode in ("s1","s1s2") else None
        else:
            s2_coll = None; s1_coll = None
            if mode in ("s2","s1s2"):
                s2_items = search_s2_items(aoi_geojson, start, end,
                                           max_cloud=self.tcfg.max_cloud,
                                           same_mgrs_tile=self.tcfg.same_mgrs_tile)
                if not s2_items and mode != "s1":
                    raise RuntimeError("No S2 items found.")
                if s2_items:
                    s2_coll = build_s2_data_collection(
                        s2_items, out_dir=region_dir / "S2",
                        bands=self.tcfg.bands_s2,
                        tile_size=self.tcfg.tile_size, stride=self.tcfg.stride,
                        aoi_geojson=aoi_geojson
                    )
            if mode in ("s1","s1s2"):
                s1_items = search_s1_items(aoi_geojson, start, end)
                if not s1_items and mode != "s2":
                    raise RuntimeError("No S1 items found.")
                if s1_items:
                    s1_coll = build_s1_data_collection(
                        s1_items, out_dir=region_dir / "S1",
                        bands=self.tcfg.bands_s1,
                        tile_size=self.tcfg.tile_size, stride=self.tcfg.stride,
                        aoi_geojson=aoi_geojson
                    )

        # Pairing depending on mode
        if mode == "s1s2":
            if not (s1_coll and s2_coll):
                raise RuntimeError("Both S1 and S2 collections are required for 's1s2' mode.")
            return build_temporal_pairs_multisensor(
                s2_collection_manifest_path=s2_coll,
                s1_collection_manifest_path=s1_coll
            )
        elif mode == "s2":
            if not s2_coll:
                raise RuntimeError("S2 collection required for 's2' mode.")
            return build_temporal_pairs(s2_coll)
        else:  # "s1"
            if not s1_coll:
                raise RuntimeError("S1 collection required for 's1' mode.")
            return build_temporal_pairs(s1_coll)

    # ---- Train (adapter only when labels available) ----
    def train(self, pairs_manifest: Path, ckpt_path: Path) -> None:
        mode = self.tcfg.sensor_mode
        print(f"Training adapter (mode={mode})...")

        if mode == "s1s2":
            ds = MultiSensorChangeDataset(pairs_manifest, tile_size=self.tcfg.tile_size)
            self.trainer.fit(ds, self.backbone, self.adapter, ckpt_path, collate_fn=collate_change)

        elif mode == "s2":
            # Wrap your existing S2 dataset into dict form
            base = DeforestationChangeDataset(str(pairs_manifest),
                                              band_names=self.tcfg.bands_s2,
                                              ndvi_drop_threshold=0.2,
                                              tile_size=self.tcfg.tile_size)
            ds = S2DictDataset(base)
            # simple collate (PyTorch default is fine for this dict of tensors)
            self.trainer.fit(ds, self.backbone, self.adapter, ckpt_path, collate_fn=None)

        else:  # "s1"
            # We typically don't have labels for S1-only deforestation (no NDVI).
            # If you want training here, provide a labeled dataset or SAR-based labels.
            raise RuntimeError("S1-only training requires labels; not provided. Use inference or supply labeled dataset.")

    # ---- Inference + analytics for all modes ----
    @torch.inference_mode()
    def infer_latest(self,
                     pairs_manifest: Path,
                     ckpt_path: Path,
                     out_dir: Path,
                     max_tiles: int = 32,
                     prob_thresh: float = 0.5) -> Path:

        mode = self.tcfg.sensor_mode

        # load adapter
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.adapter.load_state_dict(state["adapter"])
        self.adapter.eval(); self.backbone.eval()

        out_dir.mkdir(parents=True, exist_ok=True)
        pairs = json.loads(Path(pairs_manifest).read_text())["pairs"]
        if not pairs:
            raise RuntimeError("No pairs in the manifest.")

        # pick last set consistently
        if mode == "s1s2":
            last_key = pairs[-1]["scene_ids"]["s2_t1"]
            sel = [p for p in pairs if p["scene_ids"]["s2_t1"] == last_key]
        else:
            # single-sensor pair manifest uses generic s1_id/s2_id fields (prev/next)
            last_key = pairs[-1]["s2_id"]
            sel = [p for p in pairs if p["s2_id"] == last_key]

        saved = 0
        for p in sel:
            if mode == "s1s2":
                # ----- read S2 & S1 tiles
                with rasterio.open(p["t0"]["s2"]) as s2_a, rasterio.open(p["t1"]["s2"]) as s2_b:
                    A_s2 = (s2_a.read().astype(np.float32) * (1.0/10000.0))
                    B_s2 = (s2_b.read().astype(np.float32) * (1.0/10000.0))
                    aff = s2_b.transform
                with rasterio.open(p["t0"]["s1"]) as s1_a, rasterio.open(p["t1"]["s1"]) as s1_b:
                    A_s1 = s1_a.read().astype(np.float32)
                    B_s1 = s1_b.read().astype(np.float32)

                H, W = B_s2.shape[1], B_s2.shape[2]

                batch = {
                    "t0": {"s1": torch.from_numpy(A_s1)[None, ...],
                           "s2": torch.from_numpy(A_s2)[None, ...]},
                    "t1": {"s1": torch.from_numpy(B_s1)[None, ...],
                           "s2": torch.from_numpy(B_s2)[None, ...]},
                    # "target": torch.zeros(1,1,A_s2.shape[1],A_s2.shape[2])
                    "target": torch.zeros(1,1,H,W)
                }
                for t in ("t0","t1"):
                    for m in ("s1","s2"):
                        batch[t][m] = batch[t][m].to(self.device, non_blocking=True)

                with autocast(enabled=(self.device.type=="cuda")):
                    # out = self.adapter(batch, self.backbone)
                    # prob = torch.sigmoid(out["logits"]).float().cpu().numpy()[0,0]
                    out = self.adapter(batch, self.backbone)
                    logits = out["logits"]
                    # 🔧 upsample logits to full tile size
                    if logits.shape[-2:] != (H, W):
                        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                    prob = torch.sigmoid(logits).float().cpu().numpy()[0,0]

                # NDVI pre for gating/analytics from S2 t0
                red0, nir0 = A_s2[2], A_s2[3]
                ndvi_pre = (nir0 - red0) / np.clip(nir0 + red0, 1e-6, None)
                summary = deforestation_summary(change_prob=prob, meta={"transform": aff},
                                                threshold=prob_thresh, ndvi_pre=ndvi_pre)

                src_tile_path = Path(p["t1"]["s2"])
                mask = (prob >= prob_thresh).astype(np.uint8) * 255
                out_mask = out_dir / (src_tile_path.stem + "_deforest_ms.tif")
                _save_mask_like(src_tile_path, mask, out_mask)

                # quicklook
                try:
                    rgb = _load_rgb_reflectance(src_tile_path, self.tcfg.bands_s2)
                    png_dir = out_dir / "quicklooks"
                    png_path = png_dir / (src_tile_path.stem + "_deforest.png")
                    _save_quicklook_png(rgb, mask, png_path)
                except Exception as e:
                    print(f"Quicklook failed for {src_tile_path}: {e}")

                with open(out_mask.with_suffix(".json"), "w") as f:
                    json.dump(summary, f, indent=2)

            elif mode == "s2":
                # pairs schema: {"t1_path","t2_path",...}
                with rasterio.open(p["t1_path"]) as a, rasterio.open(p["t2_path"]) as b:
                    A = (a.read().astype(np.float32) * (1.0/10000.0))
                    B = (b.read().astype(np.float32) * (1.0/10000.0))
                    aff = b.transform
                H, W = B.shape[1], B.shape[2]

                batch = {
                    "t0": {"s2": torch.from_numpy(A)[None, ...].to(self.device)},
                    "t1": {"s2": torch.from_numpy(B)[None, ...].to(self.device)},
                    "target": torch.zeros(1,1,H,W, device=self.device)  # not used
                }
                with autocast(enabled=(self.device.type=="cuda")):
                    out = self.adapter(batch, self.backbone)
                    logits = out["logits"]
                    # 🔧 upsample logits to full tile size
                    if logits.shape[-2:] != (H, W):
                        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                    prob = torch.sigmoid(logits).float().cpu().numpy()[0,0]  # HxW

                red0, nir0 = A[2], A[3]
                ndvi_pre = (nir0 - red0) / np.clip(nir0 + red0, 1e-6, None)
                summary = deforestation_summary(change_prob=prob, meta={"transform": aff},
                                                threshold=prob_thresh, ndvi_pre=ndvi_pre)

                src_tile_path = Path(p["t2_path"])
                mask = (prob >= prob_thresh).astype(np.uint8) * 255
                out_mask = out_dir / (src_tile_path.stem + "_deforest_s2.tif")
                _save_mask_like(src_tile_path, mask, out_mask)

                try:
                    rgb = _load_rgb_reflectance(src_tile_path, self.tcfg.bands_s2)
                    png_dir = out_dir / "quicklooks"
                    png_path = png_dir / (src_tile_path.stem + "_deforest.png")
                    _save_quicklook_png(rgb, mask, png_path)
                except Exception as e:
                    print(f"Quicklook failed for {src_tile_path}: {e}")

                with open(out_mask.with_suffix(".json"), "w") as f:
                    json.dump(summary, f, indent=2)

            else:  # mode == "s1"
                # pairs schema: {"t1_path","t2_path",...} but they are S1 tiles
                with rasterio.open(p["t1_path"]) as a, rasterio.open(p["t2_path"]) as b:
                    A = a.read().astype(np.float32)   # S1: float32 linear
                    B = b.read().astype(np.float32)
                    aff = b.transform

                batch = {
                    "t0": {"s1": torch.from_numpy(A)[None, ...].to(self.device)},
                    "t1": {"s1": torch.from_numpy(B)[None, ...].to(self.device)},
                    "target": torch.zeros(1,1,A.shape[1],A.shape[2]).to(self.device)
                }
                with autocast(enabled=(self.device.type=="cuda")):
                    out = self.adapter(batch, self.backbone)
                    prob = torch.sigmoid(out["logits"]).float().cpu().numpy()[0,0]

                # No NDVI; analytics without forest gating (or pass external landcover)
                summary = deforestation_summary(change_prob=prob, meta={"transform": aff},
                                                threshold=prob_thresh, ndvi_pre=None)

                src_tile_path = Path(p["t2_path"])
                mask = (prob >= prob_thresh).astype(np.uint8) * 255
                out_mask = out_dir / (src_tile_path.stem + "_deforest_s1.tif")
                _save_mask_like(src_tile_path, mask, out_mask)

                # no RGB quicklook for S1; you may render SAR amplitude if you like

                with open(out_mask.with_suffix(".json"), "w") as f:
                    json.dump(summary, f, indent=2)

            saved += 1
            if saved >= max_tiles:
                break

        print(f"wrote {saved} deforestation tiles (mode={mode}) -> {out_dir}")
        return out_dir

    # ---- Full run ----
    def run(self, pairs_manifest: Path, ckpt_path: Path, out_dir: Path,
            retrain: bool = False, **infer_kwargs):
        mode = self.tcfg.sensor_mode
        if retrain or not ckpt_path.exists():
            if mode == "s1":
                print("Skipping training for S1-only: labels not provided. Using existing or randomly-initialized adapter.")
            else:
                self.train(pairs_manifest, ckpt_path)
        else:
            print(f"Using existing adapter checkpoint: {ckpt_path}")
        self.infer_latest(pairs_manifest, ckpt_path, out_dir, **infer_kwargs)


# ================================================================================
#--------------------- FLOOD WORKFLOW --------------------------------------------
# ================================================================================

# local helper: SAR quicklook (VV in dB)
def _sar_quicklook_vv_db(tile_path: Path, out_png: Path):
    with rasterio.open(tile_path) as src:
        vv = src.read(1).astype(np.float32)
    vv_db = 10.0 * np.log10(np.clip(vv, 1e-6, None))
    # simple stretch to [0,1]
    img = np.clip((vv_db + 20.0)/20.0, 0.0, 1.0)
    plt.figure(figsize=(4,4), dpi=150)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

@dataclass
class TilingConfigS1:
    bands_s1: tuple[str, ...] = ("vv","vh")
    tile_size: int = 256
    stride: Optional[int] = 256

class _TrainerFloodS1:
    """Small trainer for S1-only flood adapter (keeps backbone frozen)."""
    def __init__(self, device: torch.device, cfg: TrainingConfig):
        self.device = device
        self.cfg = cfg
        self.scaler = GradScaler(enabled=(device.type=="cuda" and cfg.amp))
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def _loss(logits, target, eps=1e-6):
        # upsample logits if needed to match target size
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        bce  = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter= (prob*target).sum(dim=(2,3))
        dice = 1 - (2*inter + eps)/(prob.sum(dim=(2,3))+target.sum(dim=(2,3))+eps)
        return bce + dice.mean()

    def fit(self, dataset, backbone, adapter, ckpt_path: Path):
        if len(dataset) == 0:
            raise RuntimeError("Flood dataset is empty (0 pairs). Check S1 pairing or time window.")

        n_val = max(1, int(len(dataset)*self.cfg.val_fraction))
        n_train = max(1, len(dataset) - n_val)
        train_set, val_set = random_split(dataset, [n_train, n_val])

        tl = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True,
                        num_workers=self.cfg.num_workers, pin_memory=True,
                        persistent_workers=self.cfg.num_workers>0)
        vl = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=False,
                        num_workers=self.cfg.num_workers, pin_memory=True,
                        persistent_workers=self.cfg.num_workers>0)

        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, adapter.parameters()),
                                lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        best = float("inf")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        backbone.eval()  # keep frozen
        for e in range(self.cfg.num_epochs):
            # train
            adapter.train()
            tot = 0.0
            for xb, yb, _ in tl:
                x0 = xb["t0"].to(self.device); x1 = xb["t1"].to(self.device); yb = yb.to(self.device)
                with autocast(enabled=(self.device.type=="cuda" and self.cfg.amp)):
                    out = adapter(x0, x1, backbone)
                    loss = self._loss(out["logits"], yb)
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                tot += float(loss.item()) * yb.size(0)
            tr = tot / len(train_set)

            # val
            adapter.eval()
            tot = 0.0
            with torch.no_grad():
                for xb, yb, _ in vl:
                    x0 = xb["t0"].to(self.device); x1 = xb["t1"].to(self.device); yb = yb.to(self.device)
                    out = adapter(x0, x1, backbone)
                    loss = self._loss(out["logits"], yb)
                    tot += float(loss.item()) * yb.size(0)
            vloss = tot / len(val_set)

            print(f"[epoch {e:02d}] train={tr:.4f}  val={vloss:.4f}")
            if vloss < best:
                best = vloss
                torch.save({"adapter": adapter.state_dict()}, ckpt_path)
                print(f"  ↳ saved {ckpt_path}")

class FloodWorkflowS1:
    """
    S1-only flood workflow:
      - build/resume S1 tiles
      - temporal pair (S1-only)
      - train (weak labels by default)
      - infer (COG mask, quicklook) + analytics (area/patches/polygons)
    """
    def __init__(self, project_root: Path | str,
                 tiling_cfg: TilingConfigS1 = TilingConfigS1(),
                 train_cfg: TrainingConfig = TrainingConfig(),
                 skip_to_pairing: bool = False):
        self.root = Path(project_root)
        self.tcfg = tiling_cfg
        self.skip_to_pairing = skip_to_pairing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = _TrainerFloodS1(self.device, train_cfg)

        # lazy imports to avoid modifying existing import section
        from eintelligence.backbone.ssl4eo_lite_backbone import SSL4EOLiteConfig, SSL4EOLiteBackbone
        from eintelligence.adapters.flood_change import FloodChangeAdapter

        bb_cfg = SSL4EOLiteConfig(in_ch=len(self.tcfg.bands_s1), freeze=True, state_dict=None)
        self.backbone = SSL4EOLiteBackbone(bb_cfg).to(self.device)
        self.adapter  = FloodChangeAdapter(c_backbone=self.backbone.out_channels, c_align=256).to(self.device)

    def build_data(self, aoi_geojson: Dict[str, Any], start: str, end: str, region_name: str) -> Path:
        region_dir = self.root / "data" / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        if self.skip_to_pairing:
            coll = discover_s1_collection(region_dir / "S1")
        else:
            s1_items = search_s1_items(aoi_geojson, start, end)
            if not s1_items:
                raise RuntimeError("No Sentinel-1 items found for given AOI/time.")
            coll = build_s1_data_collection(
                s1_items, out_dir=region_dir / "S1",
                bands=self.tcfg.bands_s1, 
                tile_size=self.tcfg.tile_size, stride=self.tcfg.stride,
                aoi_geojson=aoi_geojson
            )
        # pairs_manifest = build_temporal_pairs(coll)  # S1-only consecutive pairing
        # print("Collection manifest:", coll)
        pairs_manifest = build_temporal_pairs_relaxed_s1(coll, iou_thresh=0.90)
        print(f"S1 pairs manifest: {pairs_manifest}")
        return pairs_manifest

    def train(self, pairs_manifest: Path, ckpt_path: Path, use_weak_labels: bool = True):
        # lazy import to avoid edits above
        from eintelligence.data_prep.flood_s1_dataset import FloodS1ChangeDataset
        ds = FloodS1ChangeDataset(Path(pairs_manifest),
                                   tile_size=self.tcfg.tile_size,
                                   use_weak_labels=use_weak_labels)
        self.trainer.fit(ds, self.backbone, self.adapter, ckpt_path)

    @torch.inference_mode()
    def infer_latest(self, pairs_manifest: Path, ckpt_path: Path, out_dir: Path,
                     max_tiles: int = 32, prob_thresh: float = 0.5,
                     slope_deg: Optional[np.ndarray] = None, max_slope_deg: float = 5.0) -> Path:
        # lazy imports
        from eintelligence.analytics.flood import flood_summary

        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.adapter.load_state_dict(state["adapter"])
        self.adapter.eval(); self.backbone.eval()

        pairs = json.loads(Path(pairs_manifest).read_text())["pairs"]
        if not pairs:
            raise RuntimeError("No S1 pairs in manifest.")
        last_key = pairs[-1]["s2_id"] if "s2_id" in pairs[-1] else None
        last_pairs = [p for p in pairs if p.get("s2_id", last_key) == last_key] if last_key else pairs

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for p in last_pairs:
            with rasterio.open(p["t1_path"]) as a, rasterio.open(p["t2_path"]) as b:
                A = a.read().astype(np.float32)   # [2,H,W] linear VV,VH
                B = b.read().astype(np.float32)
                aff = b.transform

            # preprocess to normalized dB (same as dataset)
            def _norm_db(Z):
                Zdb = 10.0*np.log10(np.clip(Z, 1e-6, None))
                m = np.array([-12.0, -18.0], np.float32)[:,None,None]
                s = np.array([  6.0,   6.0], np.float32)[:,None,None]
                return (Zdb - m) / (s + 1e-6)

            A_n = _norm_db(A); B_n = _norm_db(B)
            x0 = torch.from_numpy(A_n)[None].to(self.device)
            x1 = torch.from_numpy(B_n)[None].to(self.device)

            with autocast(enabled=(self.device.type=="cuda")):
                out = self.adapter(x0, x1, self.backbone)
                logits = out["logits"]
                # upsample logits to full tile HxW if decoder output is smaller
                H, W = B.shape[1], B.shape[2]
                if logits.shape[-2:] != (H, W):
                    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
                prob = torch.sigmoid(logits).float().cpu().numpy()[0,0]

            # analytics
            summary = flood_summary(prob, transform=aff, thr=prob_thresh,
                                    slope_deg=slope_deg, max_slope_deg=max_slope_deg)
            mask = (prob >= prob_thresh).astype(np.uint8) * 255

            src_t2 = Path(p["t2_path"])
            out_mask = out_dir / (src_t2.stem + "_flood_s1.tif")
            _save_mask_like(src_t2, mask, out_mask)

            # quicklook (VV dB)
            ql_dir = out_dir / "quicklooks"
            _sar_quicklook_vv_db(src_t2, ql_dir / (src_t2.stem + "_vv.png"))

            with open(out_mask.with_suffix(".json"), "w") as f:
                json.dump(summary, f, indent=2)

            saved += 1
            if saved >= max_tiles:
                break

        print(f"wrote {saved} flood tiles -> {out_dir}")
        return out_dir

    def run(self, pairs_manifest: Path, ckpt_path: Path, out_dir: Path,
            retrain: bool = False, use_weak_labels: bool = True, **infer_kwargs):
        if retrain or not ckpt_path.exists():
            self.train(pairs_manifest, ckpt_path, use_weak_labels=use_weak_labels)
        else:
            print(f"Using existing flood adapter checkpoint: {ckpt_path}")
        return self.infer_latest(pairs_manifest, ckpt_path, out_dir, **infer_kwargs)