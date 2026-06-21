from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Literal

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data prep
from eintelligence.data_prep.fetch_multi_data import search_s1_items, search_s2_items
from eintelligence.data_prep.build_data_collection import (
    build_s1_data_collection,
    build_s2_data_collection,
)
from eintelligence.data_prep.temporal_pairing import (
    build_landcover_multisensor_manifest,
)
from eintelligence.data_prep.landcover_dataset import LandCoverDataset
from eintelligence.data_prep.worldcover_labels import REDUCED_LC_IGNORE

from eintelligence.data_prep.collate import collate_landcover

# Backbone / Fusion / Adapter
from eintelligence.backbone.ssl4eo_lite_backbone import (
    SSL4EOLiteConfig,
    MultiSensorSSL4EOLiteBackbone,
)
from eintelligence.fusion.late_fusion_kernel import LateFusionKernel
from eintelligence.adapters.landcover_head import LandCoverSegHead, LandCoverHeadConfig

# Metrics
from eintelligence.analytics.segmentation_metrics import compute_segmentation_metrics

SensorMode = Literal["s1s2", "s2", "s1"]  # future-proof, but we'll mainly use "s1s2"
TrainMode = Literal["worldcover_supervised", "self_supervised"]
RunMode = Literal["train", "infer", "train_and_infer"]

# ---------  Debugging mode: if True, prints batch shapes 

DEBUG_MODE = False

# ---------- small utils (save mask, RGB, quicklook) ----------


def _save_mask_like(src_tile_path: Path, mask_uint8: np.ndarray, out_path: Path) -> None:
    with rasterio.open(src_tile_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8", compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask_uint8, 1)


def _pick_rgb_indices(band_names: Tuple[str, ...]) -> Tuple[int, int, int]:
    band_map = {name: idx for idx, name in enumerate(band_names)}
    for need in ("B04", "B03", "B02"):
        if need not in band_map:
            raise ValueError(f"Band {need} not in band names: {band_names}")
    return band_map["B04"], band_map["B03"], band_map["B02"]


def _load_rgb_reflectance(tile_path: Path, band_names: Tuple[str, ...]) -> np.ndarray:
    with rasterio.open(tile_path) as src:
        r_idx, g_idx, b_idx = _pick_rgb_indices(band_names)
        R = src.read(r_idx + 1).astype(np.float32) / 10000.0
        G = src.read(g_idx + 1).astype(np.float32) / 10000.0
        B = src.read(b_idx + 1).astype(np.float32) / 10000.0
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def _save_quicklook_png_landcover(pred_classes: np.ndarray, out_png: Path) -> None:
    """
    Simple quicklook: show predicted classes with a qualitative colormap.
    """
    plt.figure(figsize=(4, 4), dpi=150)
    plt.imshow(pred_classes, cmap="tab20", interpolation="nearest", vmin=0)
    plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


# ---------- Configs ----------


@dataclass
class TrainingConfigLandCover:
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    val_fraction: float = 0.2
    amp: bool = True
    # For future extension: allow switching to SSL pretraining
    train_mode: TrainMode = "worldcover_supervised"


@dataclass
class TilingConfigLandCover:
    bands_s2: Tuple[str, ...] = ("B02", "B03", "B04", "B08")
    bands_s1: Tuple[str, ...] = ("vv", "vh")
    tile_size: int = 256
    stride: Optional[int] = 256
    max_cloud: int = 20
    same_mgrs_tile: bool = True
    sensor_mode: SensorMode = "s1s2"  # for now, focus on multisensor


# ---------- Trainer for land cover ----------
def check_tensor(name, t):
    if t is None:
        return
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NaN/Inf detected in {name}: shape={tuple(t.shape)}")

class _TrainerLandCover:
    def __init__(self, device: torch.device, cfg: TrainingConfigLandCover):
        self.device = device
        self.cfg = cfg
        self.scaler = GradScaler(enabled=(device.type == "cuda" and cfg.amp))
        torch.backends.cudnn.benchmark = True



    def _step(
        self,
        batch,
        fusion_kernel: LateFusionKernel,
        head: LandCoverSegHead,
        optimizer: Optional[torch.optim.Optimizer] = None,
        train: bool = True,
    ) -> float:
        if self.cfg.train_mode != "worldcover_supervised":
            # Placeholder for future self-supervised training path
            raise NotImplementedError("Self-supervised training is not implemented yet.")

        fb = batch["fusion_batch"]
        labels = batch["labels"].to(self.device)

        for k in fb.imagery:
            fb.imagery[k] = fb.imagery[k].to(self.device, non_blocking=True)
            fb.masks[k] = fb.masks[k].to(self.device, non_blocking=True)
        
        valid = labels != REDUCED_LC_IGNORE
        if valid.sum() == 0:
            return 0.0
        
        with torch.set_grad_enabled(train), autocast(
            enabled=(self.device.type == "cuda" and self.cfg.amp)
        ):
            fusion_out = fusion_kernel(fb)
            if DEBUG_MODE:
                check_tensor("fusion_out.fused", fusion_out.fused)
            logits = head(fusion_out, out_size=labels.shape[-2:])
            if DEBUG_MODE:
                check_tensor("logits", logits)
            loss = nn.CrossEntropyLoss(ignore_index=REDUCED_LC_IGNORE)(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

        return float(loss.item())

    def fit(
        self,
        dataset: LandCoverDataset,
        fusion_kernel: LateFusionKernel,
        head: LandCoverSegHead,
        ckpt_path: Path,
    ) -> None:
        if self.cfg.train_mode != "worldcover_supervised":
            raise NotImplementedError("Self-supervised training is not implemented yet.")

        n_val = int(len(dataset) * self.cfg.val_fraction)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(
            train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_landcover,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_landcover,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

        
        if DEBUG_MODE:
            dbg_batch = next(iter(train_loader))
            fb = dbg_batch["fusion_batch"]
            labels = dbg_batch["labels"]
            valid_mask = dbg_batch["valid_mask"]

            print("s1 nan:", torch.isnan(fb.imagery["s1"]).any().item())
            print("s2 nan:", torch.isnan(fb.imagery["s2"]).any().item())
            print("labels min/max:", labels.min().item(), labels.max().item())
            print("valid pixels:", valid_mask.sum().item())
            print("unique labels:", torch.unique(labels))


        optimizer = torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                list(fusion_kernel.parameters()) + list(head.parameters()),
            ),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        best_mIoU = 0.0
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for e in range(self.cfg.num_epochs):
            # train
            fusion_kernel.train()
            head.train()
            tr_total = 0.0
            for batch in train_loader:
                tr_total += (
                    self._step(batch, fusion_kernel, head, optimizer, train=True)
                    * batch["labels"].size(0)
                )
            tr_loss = tr_total / len(train_loader.dataset)

            # val
            fusion_kernel.eval()
            head.eval()
            vl_total = 0.0
            all_logits, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    fb = batch["fusion_batch"]
                    labels = batch["labels"].to(self.device)

                    for k in fb.imagery:
                        fb.imagery[k] = fb.imagery[k].to(
                            self.device, non_blocking=True
                        )
                        fb.masks[k] = fb.masks[k].to(
                            self.device, non_blocking=True
                        )

                    with autocast(
                        enabled=(self.device.type == "cuda" and self.cfg.amp)
                    ):
                        fusion_out = fusion_kernel(fb)
                        logits = head(fusion_out, out_size=labels.shape[-2:])

                        if DEBUG_MODE:
                            print("logits nan:", torch.isnan(logits).any().item())
                            print("logits inf:", torch.isinf(logits).any().item())
                            print("logits shape:", logits.shape)

                        loss = nn.CrossEntropyLoss(
                            ignore_index=REDUCED_LC_IGNORE
                        )(logits, labels)

                    vl_total += loss.item() * labels.size(0)
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())

            vl_loss = vl_total / len(val_loader.dataset)

            # compute metrics on full val set
            logits_cat = torch.cat(all_logits, dim=0)
            labels_cat = torch.cat(all_labels, dim=0)
            metrics = compute_segmentation_metrics(
                logits=logits_cat,
                labels=labels_cat,
                num_classes=6,
                ignore_index=REDUCED_LC_IGNORE,
            )

            print(
                f"[epoch {e:02d}] train_loss={tr_loss:.4f}  val_loss={vl_loss:.4f}  "
                f"mIoU={metrics.mean_iou:.3f}  macroF1={metrics.macro_f1:.3f}  OA={metrics.overall_accuracy:.3f}"
            )

            if metrics.mean_iou > best_mIoU:
                best_mIoU = metrics.mean_iou
                torch.save(
                    {
                        "fusion_kernel": fusion_kernel.state_dict(),
                        "head": head.state_dict(),
                    },
                    ckpt_path,
                )
                print(
                    f"  ↳ saved best land-cover model to {ckpt_path} (mIoU={best_mIoU:.3f})"
                )


# ---------- Land-cover workflow ----------


class LandCoverWorkflowMS:
    """
    End-to-end land-cover workflow using S1+S2 + WorldCover labels.

    - build_data(...) : search S1/S2, tile, build landcover_multisensor manifest (train-time).
    - train(...)      : train fusion + head on LandCoverDataset (WorldCover-supervised).
    - infer_latest(...) : run land-cover inference tile-wise from a manifest (labels optional).
    - run(...)        : orchestration entry with explicit run mode (train / infer / both).
    """

    def __init__(
        self,
        project_root: Path | str,
        tiling_cfg: TilingConfigLandCover = TilingConfigLandCover(),
        train_cfg: TrainingConfigLandCover = TrainingConfigLandCover(),
        skip_to_landcover_manifest: bool = False,
    ):
        self.root = Path(project_root)
        self.tcfg = tiling_cfg
        self.skip_to_landcover_manifest = skip_to_landcover_manifest
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = _TrainerLandCover(self.device, train_cfg)

        # Backbone and fusion kernel
        s1_cfg = SSL4EOLiteConfig(
            in_ch=len(self.tcfg.bands_s1), freeze=False, state_dict=None
        )
        s2_cfg = SSL4EOLiteConfig(
            in_ch=len(self.tcfg.bands_s2), freeze=False, state_dict=None
        )
        backbone = MultiSensorSSL4EOLiteBackbone(s1_cfg=s1_cfg, s2_cfg=s2_cfg)
        self.fusion_kernel = LateFusionKernel(
            backbone=backbone, fused_dim=256, use_modalities=["s1", "s2"]
        ).to(self.device)

        # Land-cover head (in_channels will be determined lazily)
        self.head: Optional[LandCoverSegHead] = None

    # ---- Build data for supervised training ----
    def build_data(
        self,
        aoi_geojson: Dict[str, Any],
        start: str,
        end: str,
        region_name: str,
        worldcover_version: str = "v200",
        worldcover_year: str = "2021",
    ) -> Path:
        """
        Build S1/S2 collections and a multisensor land-cover manifest including WorldCover labels.
        This is used for supervised training.
        """
        region_dir = self.root / "data" / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        if self.skip_to_landcover_manifest:
            # assume S1/S2 collections already built
            s2_coll = region_dir / "S2" / "collection_manifest_s2.json"
            s1_coll = region_dir / "S1" / "collection_manifest_s1.json"
        else:
            # build S2 collection
            s2_items = search_s2_items(
                aoi_geojson,
                start,
                end,
                max_cloud=self.tcfg.max_cloud,
                same_mgrs_tile=self.tcfg.same_mgrs_tile,
            )
            if not s2_items:
                raise RuntimeError("No S2 items found for land-cover workflow.")
            s2_coll = build_s2_data_collection(
                s2_items,
                out_dir=region_dir / "S2",
                bands=self.tcfg.bands_s2,
                tile_size=self.tcfg.tile_size,
                stride=self.tcfg.stride,
                aoi_geojson=aoi_geojson,
            )

            # build S1 collection
            s1_items = search_s1_items(aoi_geojson, start, end)
            if not s1_items:
                raise RuntimeError("No S1 items found for land-cover workflow.")
            s1_coll = build_s1_data_collection(
                s1_items,
                out_dir=region_dir / "S1",
                bands=self.tcfg.bands_s1,
                tile_size=self.tcfg.tile_size,
                stride=self.tcfg.stride,
                aoi_geojson=aoi_geojson,
            )

        # build S1+S2+WorldCover tile manifest
        landcover_manifest = build_landcover_multisensor_manifest(
            s2_collection_manifest_path=s2_coll,
            s1_collection_manifest_path=s1_coll,
            iou_min=0.8,
            worldcover_version=worldcover_version,
            worldcover_year=worldcover_year,
        )
        return landcover_manifest

    # ---- Train (WorldCover-supervised) ----
    def train(self, landcover_manifest: Path, ckpt_path: Path) -> None:
        print("Training land-cover head (WorldCover-supervised)...")
        dataset = LandCoverDataset(landcover_manifest)

        # probe one batch to define head
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_landcover)
        example = next(iter(loader))
        fb = example["fusion_batch"]

        # send to device and run fusion once
        for k in fb.imagery:
            fb.imagery[k] = fb.imagery[k].to(self.device)
            fb.masks[k] = fb.masks[k].to(self.device)

        with torch.no_grad():
            fusion_out = self.fusion_kernel(fb)
        in_channels = fusion_out.fused.shape[1]

        head_cfg = LandCoverHeadConfig(
            in_channels=in_channels,
            num_classes=6,
            decoder_channels=256,
            dropout=0.1,
        )
        self.head = LandCoverSegHead(head_cfg).to(self.device)

        self.trainer.fit(dataset, self.fusion_kernel, self.head, ckpt_path)

    # ---- Inference (can be used on training AOIs or other AOIs sharing same manifest schema) ----
    @torch.inference_mode()
    def infer_latest(
        self,
        landcover_manifest: Path,
        ckpt_path: Path,
        out_dir: Path,
        max_tiles: int = 32,
    ) -> Path:
        """
        Tile-wise land-cover inference from a manifest. For now it assumes the same
        manifest structure as training (i.e., built by build_landcover_multisensor_manifest),
        but it ignores WorldCover labels at inference.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        # load checkpoint
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device)
            self.fusion_kernel.load_state_dict(state["fusion_kernel"])
            # Recreate head if needed
            if self.head is None:
                dataset = LandCoverDataset(landcover_manifest)
                loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_landcover)
                example = next(iter(loader))
                fb = example["fusion_batch"]
                for k in fb.imagery:
                    fb.imagery[k] = fb.imagery[k].to(self.device)
                    fb.masks[k] = fb.masks[k].to(self.device)
                with torch.no_grad():
                    fusion_out = self.fusion_kernel(fb)
                in_channels = fusion_out.fused.shape[1]
                head_cfg = LandCoverHeadConfig(
                    in_channels=in_channels,
                    num_classes=6,
                    decoder_channels=256,
                    dropout=0.1,
                )
                self.head = LandCoverSegHead(head_cfg).to(self.device)
            self.head.load_state_dict(state["head"])
        else:
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        self.fusion_kernel.eval()
        self.head.eval()

        # simple tile-wise inference
        dataset = LandCoverDataset(landcover_manifest)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_landcover)

        saved = 0
        for sample in loader:
            fb = sample["fusion_batch"]

            # meta must contain at least "s2_path" to anchor georeferencing
            s2_path_str = fb.meta.get("s2_path", None)
            if s2_path_str is None:
                raise RuntimeError(
                    "FusionBatch.meta must contain 's2_path' for land-cover inference; "
                    "ensure LandCoverDataset populates it."
                )

            if isinstance(s2_path_str, list):
                if len(s2_path_str) != 1:
                    raise RuntimeError(f"Expected one s2_path for batch_size=1, got {len(s2_path_str)}")
                s2_path_str = s2_path_str[0]
            
            s2_path = Path(s2_path_str)

            # Determine output size from S2 tile
            with rasterio.open(s2_path) as src:
                H, W = src.height, src.width

            for k in fb.imagery:
                fb.imagery[k] = fb.imagery[k].to(self.device)
                fb.masks[k] = fb.masks[k].to(self.device)

            with autocast(enabled=(self.device.type == "cuda")):
                fusion_out = self.fusion_kernel(fb)
                logits = self.head(fusion_out, out_size=(H, W))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [C, H, W]

            pred_classes = probs.argmax(axis=0).astype(np.uint8)  # [H,W]

            # Save predicted map as a GeoTIFF aligned to the S2 tile
            out_mask = out_dir / (s2_path.stem + "_landcover_pred.tif")
            _save_mask_like(s2_path, pred_classes, out_mask)

            # Optional quicklook: class map as PNG
            png_dir = out_dir / "quicklooks"
            png_path = png_dir / (s2_path.stem + "_landcover_pred.png")
            _save_quicklook_png_landcover(pred_classes, png_path)

            saved += 1
            if saved >= max_tiles:
                break

        print(f"wrote {saved} land-cover tiles -> {out_dir}")
        return out_dir

    # ---- Full run with explicit mode ----
    def run(
        self,
        landcover_manifest: Path,
        ckpt_path: Path,
        out_dir: Path,
        mode: RunMode = "train_and_infer",
        retrain: bool = False,
        **infer_kwargs,
    ) -> None:
        """
        Orchestrate training and/or inference:

        - mode="train": only train (WorldCover-supervised), do not run inference.
        - mode="infer": only run inference, requires an existing checkpoint (or retrain=True).
        - mode="train_and_infer": train if needed (or retrain=True), then run inference.
        """
        if mode == "train":
            if retrain or not ckpt_path.exists():
                self.train(landcover_manifest, ckpt_path)
            else:
                print(f"Checkpoint already exists, skipping training: {ckpt_path}")
            return

        if mode == "infer":
            if retrain or not ckpt_path.exists():
                # If retrain=True, train first even in infer mode
                if retrain:
                    self.train(landcover_manifest, ckpt_path)
                else:
                    raise RuntimeError(
                        "mode='infer' but checkpoint does not exist. "
                        "Set retrain=True or use mode='train_and_infer'."
                    )
            self.infer_latest(landcover_manifest, ckpt_path, out_dir, **infer_kwargs)
            return

        # mode == "train_and_infer"
        if retrain or not ckpt_path.exists():
            self.train(landcover_manifest, ckpt_path)
        else:
            print(f"Using existing land-cover checkpoint: {ckpt_path}")
        self.infer_latest(landcover_manifest, ckpt_path, out_dir, **infer_kwargs)