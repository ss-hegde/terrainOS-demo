from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Literal, List

import hashlib
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eintelligence.data_prep.fetch_multi_data import search_s1_items, search_s2_items
from eintelligence.data_prep.build_data_collection import (
    build_s1_data_collection,
    build_s2_data_collection,
)
from eintelligence.data_prep.landcover_dataset import LandCoverDataset
from eintelligence.data_prep.worldcover_labels import REDUCED_LC_IGNORE
from eintelligence.data_prep.collate import collate_landcover
from eintelligence.data_prep.registry_manager import grouped_split, save_splits, load_splits
from eintelligence.data_prep.raster_stitching import stitch_prediction_tree_by_scene
from eintelligence.data_prep.temporal_pairing import build_landcover_multisensor_manifest
from eintelligence.data_prep.manifest_utils import merge_record_manifests

from eintelligence.backbone.ssl4eo_lite_backbone import (
    SSL4EOLiteConfig,
    MultiSensorSSL4EOLiteBackbone,
)
from eintelligence.fusion.late_fusion_kernel import LateFusionKernel
from eintelligence.adapters.landcover_head import LandCoverSegHead, LandCoverHeadConfig
from eintelligence.analytics.segmentation_metrics import compute_segmentation_metrics


SensorMode = Literal["s1s2", "s2", "s1"]
TrainMode = Literal["worldcover_supervised", "self_supervised"]
RunMode = Literal["train", "infer", "train_and_infer"]

DEBUG_MODE = False

LANDCOVER_COLORS = {
    0: (0.0, 0.4, 0.0),
    1: (0.9, 0.9, 0.2),
    2: (0.8, 0.2, 0.2),
    3: (0.2, 0.4, 0.9),
    4: (0.6, 0.4, 0.2),
    5: (0.6, 0.8, 0.6),
}


def _save_mask_like(src_tile_path: Path, mask_uint8: np.ndarray, out_path: Path) -> None:
    with rasterio.open(src_tile_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="uint8", compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask_uint8, 1)


def _percentile_stretch(rgb: np.ndarray, low=2, high=98) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[2]):
        band = rgb[:, :, c]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            out[:, :, c] = 0
            continue
        lo, hi = np.percentile(finite, [low, high])
        if hi <= lo:
            out[:, :, c] = 0
        else:
            out[:, :, c] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def _read_s2_rgb(s2_path: Path) -> np.ndarray:
    with rasterio.open(s2_path) as src:
        count = src.count
        if count >= 4:
            blue = src.read(1).astype(np.float32)
            green = src.read(2).astype(np.float32)
            red = src.read(3).astype(np.float32)
            rgb = np.dstack([red, green, blue])
        elif count >= 3:
            arr = src.read([1, 2, 3]).astype(np.float32)
            rgb = np.transpose(arr, (1, 2, 0))
        else:
            raise RuntimeError(f"Expected at least 3 bands in {s2_path}, got {count}")
    return _percentile_stretch(rgb)


def _mask_to_rgb(mask: np.ndarray, color_map: dict[int, tuple[float, float, float]]) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls, color in color_map.items():
        rgb[mask == cls] = color
    return rgb


def _save_quicklook_png_landcover(pred_classes: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(4, 4), dpi=150)
    plt.imshow(pred_classes, cmap="tab20", interpolation="nearest", vmin=0)
    plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()


def _save_side_by_side_quicklook_landcover(
    s2_path: Path,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_png: Path,
    color_map: dict[int, tuple[float, float, float]] = LANDCOVER_COLORS,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    s2_rgb = _read_s2_rgb(s2_path)
    gt_rgb = _mask_to_rgb(gt_mask, color_map)
    pred_rgb = _mask_to_rgb(pred_mask, color_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axes[0].imshow(s2_rgb)
    axes[0].set_title("S2 RGB")
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


@dataclass
class TrainingConfigLandCover:
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    amp: bool = True
    train_mode: TrainMode = "worldcover_supervised"


@dataclass
class TrainMetrics:
    """
    What _TrainerLandCover.fit()/LandCoverWorkflowMS.train() return -- the *final*
    epoch's numbers (not necessarily the best-mIoU epoch, which is what actually gets
    checkpointed; see fit()'s best_miou tracking below), so a caller can show
    something concrete after a short canvas-triggered training run rather than a
    bare 200.
    """

    train_loss: float
    val_loss: float
    mean_iou: float


@dataclass
class TileInferenceResult:
    """One tile's worth of infer_region() output."""

    tile_id: str
    scene_id: str
    pred_mask_path: Path
    quicklook_path: Path
    compare_quicklook_path: Optional[Path]  # only set if the tile had a comparable label
    uncertainty: Optional[float]            # mean of FusionOutput.uncertainty; None if the kernel didn't produce one
    per_modality: Dict[str, float]          # modality -> mean of FusionOutput.per_modality[modality]


@dataclass
class TilingConfigLandCover:
    bands_s2: Tuple[str, ...] = ("B02", "B03", "B04", "B08")
    bands_s1: Tuple[str, ...] = ("VV", "VH")
    tile_size: int = 256
    stride: Optional[int] = 256
    max_cloud: int = 20
    same_mgrs_tile: bool = True
    sensor_mode: SensorMode = "s1s2"


def _collection_manifest_is_valid(path: Path) -> bool:
    """
    Minimal-validity check for a build_s1_data_collection()/
    build_s2_data_collection() output (`{"scenes": [...]}`, written by
    eintelligence/data_prep/build_data_collection.py::_build_collection) --
    exists, parses as JSON, and has at least one scene entry. This is what
    ingest_region() checks per-sensor to decide whether that sensor's STAC
    search + tiling can be skipped and the existing manifest reused as-is.
    """
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    scenes = payload.get("scenes")
    return isinstance(scenes, list) and len(scenes) > 0


def ingestion_fingerprint(
    aoi_geojson: Dict[str, Any],
    start: str,
    end: str,
    tiling_cfg: TilingConfigLandCover,
) -> str:
    """
    Deterministic fingerprint over everything that actually determines what
    ingest_region() fetches/tiles: AOI geometry, date range, and the tiling
    params that feed search_s2_items/search_s1_items and
    build_s1_data_collection/build_s2_data_collection (max_cloud,
    same_mgrs_tile, bands_s1, bands_s2, tile_size, stride). Same inputs always
    produce the same fingerprint; different inputs produce a different one
    (barring a SHA-256 collision).

    Deliberately excludes sensor_mode: sensor_mode doesn't change what
    ingest_region() fetches today -- it always pulls both S1 and S2 regardless
    (see root CLAUDE.md's open item on this) -- so two requests differing only
    in sensor_mode should, correctly, share the same ingested S1/S2 data.

    Prefixed with "canvas_ingest_" so this can never collide with a real named
    training region (e.g. "munich_lc_2023_summer") living alongside it under
    data/.
    """
    payload = {
        "aoi_geojson": aoi_geojson,
        "start": start,
        "end": end,
        "max_cloud": tiling_cfg.max_cloud,
        "same_mgrs_tile": tiling_cfg.same_mgrs_tile,
        "bands_s1": list(tiling_cfg.bands_s1),
        "bands_s2": list(tiling_cfg.bands_s2),
        "tile_size": tiling_cfg.tile_size,
        "stride": tiling_cfg.stride,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"canvas_ingest_{digest}"


def check_tensor(name, t):
    if t is None:
        return
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"NaN/Inf detected in {name}: shape={tuple(t.shape)}")


def _load_tile_records(manifest_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(manifest_path.read_text())
    return data.get("tiles", [])


def _record_group_id(record: Dict[str, Any]) -> str:
    return str(record.get("group_id") or record.get("scene_id") or record.get("tile_id"))


def _indices_from_group_ids(records: List[Dict[str, Any]], group_ids: List[str]) -> List[int]:
    keep = set(group_ids)
    return [i for i, r in enumerate(records) if _record_group_id(r) in keep]


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
            raise NotImplementedError("Self-supervised training is not implemented yet.")

        fb = batch["fusion_batch"]
        labels = batch["labels"].to(self.device)

        for k in fb.imagery:
            fb.imagery[k] = fb.imagery[k].to(self.device, non_blocking=True)
            fb.masks[k] = fb.masks[k].to(self.device, non_blocking=True)

        if (labels != REDUCED_LC_IGNORE).sum() == 0:
            return 0.0

        with torch.set_grad_enabled(train), autocast(enabled=(self.device.type == "cuda" and self.cfg.amp)):
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
        train_dataset,
        val_dataset,
        fusion_kernel: LateFusionKernel,
        head: LandCoverSegHead,
        ckpt_path: Path,
    ) -> TrainMetrics:
        if self.cfg.train_mode != "worldcover_supervised":
            raise NotImplementedError("Self-supervised training is not implemented yet.")
        if self.cfg.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.cfg.num_epochs}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_landcover,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_landcover,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, list(fusion_kernel.parameters()) + list(head.parameters())),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        best_miou = 0.0
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for e in range(self.cfg.num_epochs):
            fusion_kernel.train()
            head.train()
            tr_total = 0.0
            for batch in train_loader:
                tr_total += self._step(batch, fusion_kernel, head, optimizer, train=True) * batch["labels"].size(0)
            tr_loss = tr_total / max(1, len(train_loader.dataset))

            fusion_kernel.eval()
            head.eval()
            vl_total = 0.0
            all_logits, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    fb = batch["fusion_batch"]
                    labels = batch["labels"].to(self.device)

                    for k in fb.imagery:
                        fb.imagery[k] = fb.imagery[k].to(self.device, non_blocking=True)
                        fb.masks[k] = fb.masks[k].to(self.device, non_blocking=True)

                    with autocast(enabled=(self.device.type == "cuda" and self.cfg.amp)):
                        fusion_out = fusion_kernel(fb)
                        logits = head(fusion_out, out_size=labels.shape[-2:])
                        loss = nn.CrossEntropyLoss(ignore_index=REDUCED_LC_IGNORE)(logits, labels)

                    vl_total += loss.item() * labels.size(0)
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())

            vl_loss = vl_total / max(1, len(val_loader.dataset))
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

            if metrics.mean_iou > best_miou:
                best_miou = metrics.mean_iou
                torch.save(
                    {
                        "fusion_kernel": fusion_kernel.state_dict(),
                        "head": head.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"  ↳ saved best land-cover model to {ckpt_path} (mIoU={best_miou:.3f})")

        # Final epoch's numbers, not best_miou's -- see TrainMetrics' docstring.
        # tr_loss/vl_loss/metrics are guaranteed bound here since num_epochs >= 1 is
        # enforced above, so the loop always runs at least once.
        return TrainMetrics(train_loss=tr_loss, val_loss=vl_loss, mean_iou=metrics.mean_iou)


class LandCoverWorkflowMS:
    def __init__(
        self,
        project_root: Path | str,
        tiling_cfg: TilingConfigLandCover = TilingConfigLandCover(),
        train_cfg: TrainingConfigLandCover = TrainingConfigLandCover(),
    ):
        self.root = Path(project_root)
        self.tcfg = tiling_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = _TrainerLandCover(self.device, train_cfg)

        s1_cfg = SSL4EOLiteConfig(in_ch=len(self.tcfg.bands_s1), freeze=False, state_dict=None)
        s2_cfg = SSL4EOLiteConfig(in_ch=len(self.tcfg.bands_s2), freeze=False, state_dict=None)
        backbone = MultiSensorSSL4EOLiteBackbone(s1_cfg=s1_cfg, s2_cfg=s2_cfg)
        # use_modalities must actually gate fusion by sensor_mode — previously this
        # was hardcoded to ["s1", "s2"] regardless of tcfg.sensor_mode, so swapping
        # checkpoints/sensor_mode never changed which modalities were fused (see
        # CLAUDE.md "Open" item 1).
        sensor_mode_to_modalities: Dict[SensorMode, List[str]] = {
            "s1": ["s1"],
            "s2": ["s2"],
            "s1s2": ["s1", "s2"],
        }
        use_modalities = sensor_mode_to_modalities[self.tcfg.sensor_mode]
        self.fusion_kernel = LateFusionKernel(backbone=backbone, fused_dim=256, use_modalities=use_modalities).to(self.device)
        self.head: Optional[LandCoverSegHead] = None

    def ingest_region(
        self,
        aoi_geojson: Dict[str, Any],
        start: str,
        end: str,
        region_name: str,
        aoi_id: Optional[str] = None,
        job_id: Optional[str] = None,
        registry_relpath: str = "data/corpus/registry/scenes.jsonl",
    ) -> Tuple[Path, Path, Path]:
        region_dir = self.root / "data" / region_name
        region_dir.mkdir(parents=True, exist_ok=True)

        aoi_id = aoi_id or region_name
        job_id = job_id or f"{region_name}_{start}_{end}"
        registry_path = self.root / registry_relpath
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Per-sensor cache reuse: if region_name is a fingerprint of the exact
        # AOI/dates/tiling config (see ingestion_fingerprint()), a prior identical
        # request already left a valid collection manifest at this exact path --
        # skip the STAC search + tiling entirely and reuse it. Falls back to a
        # normal fresh ingest whenever the manifest is missing/empty/corrupt, same
        # as if this were the first time.
        s2_manifest_path = region_dir / "S2" / "collection_manifest_s2.json"
        if _collection_manifest_is_valid(s2_manifest_path):
            print(
                f"[ingest_region] S2: reusing existing ingestion for region='{region_name}' "
                f"-> {s2_manifest_path} (skipped search_s2_items)"
            )
            s2_coll = s2_manifest_path
        else:
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
                aoi_id=aoi_id,
                job_id=job_id,
                registry_path=registry_path,
            )

        s1_manifest_path = region_dir / "S1" / "collection_manifest_s1.json"
        if _collection_manifest_is_valid(s1_manifest_path):
            print(
                f"[ingest_region] S1: reusing existing ingestion for region='{region_name}' "
                f"-> {s1_manifest_path} (skipped search_s1_items)"
            )
            s1_coll = s1_manifest_path
        else:
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
                aoi_id=aoi_id,
                job_id=job_id,
                registry_path=registry_path,
            )

        return s1_coll, s2_coll, registry_path

    def _build_group_splits_from_manifest(
        self,
        landcover_manifest: Path,
        splits_path: Path,
        seed: int = 42,
        fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> None:
        records = _load_tile_records(landcover_manifest)
        if not records:
            raise RuntimeError(f"No tile records found in manifest: {landcover_manifest}")

        split_records = []
        for r in records:
            split_records.append(
                {
                    "scene_id": str(r.get("scene_id") or r.get("tile_id")),
                    "group_id": _record_group_id(r),
                }
            )

        splits = grouped_split(
            split_records,
            group_key="group_id",
            seed=seed,
            fractions=fractions,
        )
        save_splits(splits_path, splits, group_key="group_id")

    
    def _load_split_indices(
        self,
        landcover_manifest: Path,
        splits_path: Path,
    ) -> Tuple[List[int], List[int], List[int]]:
        if not splits_path.exists():
            self._build_group_splits_from_manifest(landcover_manifest, splits_path)

        records = _load_tile_records(landcover_manifest)
        split_ids = load_splits(splits_path)

        train_ids = split_ids.get("train", [])
        val_ids = split_ids.get("val", [])
        test_ids = split_ids.get("test", [])

        train_idx = _indices_from_group_ids(records, train_ids)
        val_idx = _indices_from_group_ids(records, val_ids)
        test_idx = _indices_from_group_ids(records, test_ids)

        if not train_idx:
            raise RuntimeError("Training split is empty.")
        if not val_idx:
            raise RuntimeError("Validation split is empty.")
        if not test_idx:
            print("[WARN] Test split is empty.")

        return train_idx, val_idx, test_idx

    def _ensure_head(self, dataset: LandCoverDataset) -> None:
        if self.head is not None:
            return

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

    def train(
        self,
        landcover_manifest: Path,
        splits_path: Path,
        ckpt_path: Path,
    ) -> TrainMetrics:
        print("Training land-cover head (WorldCover-supervised)...")

        dataset = LandCoverDataset(landcover_manifest)
        train_idx, val_idx, _test_idx = self._load_split_indices(landcover_manifest, splits_path)

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)

        self._ensure_head(dataset)
        return self.trainer.fit(train_set, val_set, self.fusion_kernel, self.head, ckpt_path)

    @torch.inference_mode()
    def infer_split(
        self,
        landcover_manifest: Path,
        splits_path: Path,
        ckpt_path: Path,
        out_dir: Path,
        split: str = "test",
        max_tiles: Optional[int] = None,
        stitch_scenes: bool = True,
    ) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)

        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint not found: {ckpt_path}")

        dataset = LandCoverDataset(landcover_manifest)
        self._ensure_head(dataset)

        train_idx, val_idx, test_idx = self._load_split_indices(landcover_manifest, splits_path)
        split_map = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        if split not in split_map:
            raise ValueError(f"Unsupported split: {split}")

        split_idx = split_map[split]
        if not split_idx:
            raise RuntimeError(f"Requested split is empty: {split}")

        subset = Subset(dataset, split_idx)

        state = torch.load(ckpt_path, map_location=self.device)
        self.fusion_kernel.load_state_dict(state["fusion_kernel"])
        self.head.load_state_dict(state["head"])

        self.fusion_kernel.eval()
        self.head.eval()

        loader = DataLoader(subset, batch_size=1, shuffle=False, collate_fn=collate_landcover)

        split_root = out_dir / split
        split_root.mkdir(parents=True, exist_ok=True)

        saved = 0
        for sample in loader:
            fb = sample["fusion_batch"]

            s2_path_str = fb.meta.get("s2_path")
            if s2_path_str is None:
                raise RuntimeError("FusionBatch.meta must contain 's2_path' for inference.")

            if isinstance(s2_path_str, list):
                if len(s2_path_str) != 1:
                    raise RuntimeError(f"Expected one s2_path for batch_size=1, got {len(s2_path_str)}")
                s2_path_str = s2_path_str[0]

            s2_path = Path(s2_path_str)

            scene_id = fb.meta.get("scene_id")
            if isinstance(scene_id, list):
                scene_id = scene_id[0]
            # manifest tiles built by build_landcover_multisensor_manifest carry no
            # "scene_id" field, so fall back to the actual per-scene directory:
            # out_dir/<scene_id>/tiles_s2/<tile>.tif, i.e. two levels above the tile
            # file. s2_path.parent.name is just the literal "tiles_s2" subdir shared
            # by every scene -- using it collapses all scenes into one output dir and
            # silently overwrites same-row/col predictions from different dates.
            scene_id = str(scene_id or s2_path.parent.parent.name)

            group_id = fb.meta.get("group_id")
            if isinstance(group_id, list):
                group_id = group_id[0]

            aoi_id = fb.meta.get("aoi_id")
            if isinstance(aoi_id, list):
                aoi_id = aoi_id[0]

            stitch_namespace = str(group_id or aoi_id or scene_id)

            scene_out_dir = split_root / stitch_namespace / scene_id
            scene_out_dir.mkdir(parents=True, exist_ok=True)

            with rasterio.open(s2_path) as src:
                H, W = src.height, src.width

            for k in fb.imagery:
                fb.imagery[k] = fb.imagery[k].to(self.device)
                fb.masks[k] = fb.masks[k].to(self.device)

            with autocast(enabled=(self.device.type == "cuda")):
                fusion_out = self.fusion_kernel(fb)
                logits = self.head(fusion_out, out_size=(H, W))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_classes = probs.argmax(axis=0).astype(np.uint8)

            out_mask = scene_out_dir / f"{s2_path.stem}_landcover_pred.tif"
            _save_mask_like(s2_path, pred_classes, out_mask)

            png_dir = scene_out_dir / "quicklooks"
            png_dir.mkdir(parents=True, exist_ok=True)

            pred_png_path = png_dir / f"{s2_path.stem}_landcover_pred.png"
            _save_quicklook_png_landcover(pred_classes, pred_png_path)

            gt = sample.get("labels")
            if gt is not None:
                gt_mask = gt.cpu().numpy()[0].astype(np.uint8)
                compare_png_path = png_dir / f"{s2_path.stem}_rgb_gt_pred.png"
                _save_side_by_side_quicklook_landcover(
                    s2_path=s2_path,
                    gt_mask=gt_mask,
                    pred_mask=pred_classes,
                    out_png=compare_png_path,
                )

            saved += 1
            if max_tiles is not None and saved >= max_tiles:
                break

        print(f"wrote {saved} land-cover tiles for split='{split}' -> {split_root}")

        if stitch_scenes:
            stitched_root = out_dir / f"{split}_stitched"
            stitched_manifest = stitch_prediction_tree_by_scene(
                pred_root=split_root,
                out_dir=stitched_root,
                color_map=LANDCOVER_COLORS,
                manifest_path=landcover_manifest,
            )
            print(f"stitched scene mosaics -> {stitched_manifest}")

        return split_root

    @torch.inference_mode()
    def infer_region(
        self,
        aoi_geojson: Dict[str, Any],
        start: str,
        end: str,
        ckpt_path: Path,
        region_name: str,
        out_dir: Path,
        aoi_id: Optional[str] = None,
        job_id: Optional[str] = None,
        registry_relpath: str = "data/corpus/registry/scenes.jsonl",
        iou_min: float = 0.8,
        worldcover_version: str = "v200",
        worldcover_year: str = "2021",
        max_tiles: Optional[int] = None,
        stitch_scenes: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest a fresh AOI/date range end-to-end and run inference over every tile
        produced, against an already-trained checkpoint for a specific sensor mode.

        For a canvas "run inference on this AOI right now" request: no train/val/test
        split logic (that's infer_split(), for evaluating a pre-existing manifest
        split). Never trains -- raises if `ckpt_path` doesn't exist.
        """
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise RuntimeError(
                f"Checkpoint not found for infer_region: {ckpt_path}. "
                "infer_region never trains -- train a checkpoint first "
                "(LandCoverWorkflowMS.run(mode='train') or mode='train_and_infer')."
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- ingest + build this request's own tile manifest, same sequence the
        # notebook uses (regional manifest -> merge into a pooled manifest). Scoped
        # to out_dir rather than the shared training corpus manifest, so an ad hoc
        # canvas AOI never gets mixed into the training pool. ---
        s1_coll, s2_coll, _registry_path = self.ingest_region(
            aoi_geojson=aoi_geojson,
            start=start,
            end=end,
            region_name=region_name,
            aoi_id=aoi_id,
            job_id=job_id,
            registry_relpath=registry_relpath,
        )

        regional_manifest = build_landcover_multisensor_manifest(
            s2_collection_manifest_path=s2_coll,
            s1_collection_manifest_path=s1_coll,
            iou_min=iou_min,
            worldcover_version=worldcover_version,
            worldcover_year=worldcover_year,
        )

        request_manifest = merge_record_manifests(
            manifest_paths=[regional_manifest],
            out_path=out_dir / f"{region_name}_manifest.json",
            record_key="tiles",
            task_name="landcover_multisensor",
            namespaces=[region_name],
            fields_to_prefix=("group_id", "tile_id", "scene_id"),
            set_default_aoi_id=True,
            deduplicate_on="tile_id",
            sort_by=("aoi_id", "group_id", "scene_id", "datetime", "row", "col", "tile_id"),
        )

        # --- inference only, no split ---
        dataset = LandCoverDataset(request_manifest)
        self._ensure_head(dataset)

        state = torch.load(ckpt_path, map_location=self.device)
        self.fusion_kernel.load_state_dict(state["fusion_kernel"])
        self.head.load_state_dict(state["head"])

        self.fusion_kernel.eval()
        self.head.eval()

        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_landcover)

        tiles_root = out_dir / "tiles"
        tiles_root.mkdir(parents=True, exist_ok=True)

        tiles: List[TileInferenceResult] = []
        saved = 0

        for sample in loader:
            fb = sample["fusion_batch"]

            s2_path_str = fb.meta.get("s2_path")
            if s2_path_str is None:
                raise RuntimeError("FusionBatch.meta must contain 's2_path' for inference.")
            if isinstance(s2_path_str, list):
                if len(s2_path_str) != 1:
                    raise RuntimeError(f"Expected one s2_path for batch_size=1, got {len(s2_path_str)}")
                s2_path_str = s2_path_str[0]
            s2_path = Path(s2_path_str)

            scene_id = fb.meta.get("scene_id")
            if isinstance(scene_id, list):
                scene_id = scene_id[0]
            # see matching comment in infer_split(): fall back to the real per-scene
            # directory (two levels above the tile file), not the shared "tiles_s2"
            # subdir name -- otherwise predictions from different S2 scenes at the
            # same row/col silently overwrite each other.
            scene_id = str(scene_id or s2_path.parent.parent.name)

            tile_id = fb.meta.get("tile_id")
            if isinstance(tile_id, list):
                tile_id = tile_id[0]
            tile_id = str(tile_id or s2_path.stem)

            group_id = fb.meta.get("group_id")
            if isinstance(group_id, list):
                group_id = group_id[0]

            aoi_id_meta = fb.meta.get("aoi_id")
            if isinstance(aoi_id_meta, list):
                aoi_id_meta = aoi_id_meta[0]

            stitch_namespace = str(group_id or aoi_id_meta or scene_id)

            scene_out_dir = tiles_root / stitch_namespace / scene_id
            scene_out_dir.mkdir(parents=True, exist_ok=True)

            with rasterio.open(s2_path) as src:
                H, W = src.height, src.width

            for k in fb.imagery:
                fb.imagery[k] = fb.imagery[k].to(self.device)
                fb.masks[k] = fb.masks[k].to(self.device)

            with autocast(enabled=(self.device.type == "cuda")):
                fusion_out = self.fusion_kernel(fb)
                logits = self.head(fusion_out, out_size=(H, W))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_classes = probs.argmax(axis=0).astype(np.uint8)

            # Confidence/xAI panel inputs -- reduced to per-tile summary stats rather
            # than raw feature maps. uncertainty is None today: LateFusionKernel.forward
            # always sets FusionOutput.uncertainty=None, so this passes that through
            # faithfully rather than fabricating a value.
            uncertainty_mean = (
                float(fusion_out.uncertainty.float().mean().item())
                if fusion_out.uncertainty is not None
                else None
            )
            per_modality_mean = {
                k: float(v.float().mean().item()) for k, v in fusion_out.per_modality.items()
            }

            out_mask = scene_out_dir / f"{s2_path.stem}_landcover_pred.tif"
            _save_mask_like(s2_path, pred_classes, out_mask)

            png_dir = scene_out_dir / "quicklooks"
            png_dir.mkdir(parents=True, exist_ok=True)

            pred_png_path = png_dir / f"{s2_path.stem}_landcover_pred.png"
            _save_quicklook_png_landcover(pred_classes, pred_png_path)

            compare_png_path = None
            gt = sample.get("labels")
            if gt is not None:
                gt_mask = gt.cpu().numpy()[0].astype(np.uint8)
                compare_png_path = png_dir / f"{s2_path.stem}_rgb_gt_pred.png"
                _save_side_by_side_quicklook_landcover(
                    s2_path=s2_path,
                    gt_mask=gt_mask,
                    pred_mask=pred_classes,
                    out_png=compare_png_path,
                )

            tiles.append(
                TileInferenceResult(
                    tile_id=tile_id,
                    scene_id=scene_id,
                    pred_mask_path=out_mask,
                    quicklook_path=pred_png_path,
                    compare_quicklook_path=compare_png_path,
                    uncertainty=uncertainty_mean,
                    per_modality=per_modality_mean,
                )
            )

            saved += 1
            if max_tiles is not None and saved >= max_tiles:
                break

        print(f"[infer_region] wrote {saved} land-cover tiles for region='{region_name}' -> {tiles_root}")

        stitched_manifest_path: Optional[Path] = None
        if stitch_scenes:
            stitched_root = out_dir / "stitched"
            stitched_manifest_path = stitch_prediction_tree_by_scene(
                pred_root=tiles_root,
                out_dir=stitched_root,
                color_map=LANDCOVER_COLORS,
                manifest_path=request_manifest,
            )
            print(f"[infer_region] stitched scene mosaics -> {stitched_manifest_path}")

        return {
            "region_name": region_name,
            "manifest_path": request_manifest,
            "tiles": tiles,
            "stitched_manifest_path": stitched_manifest_path,
        }

    def run(
        self,
        landcover_manifest: Path,
        splits_path: Path,
        ckpt_path: Path,
        out_dir: Path,
        mode: RunMode = "train_and_infer",
        retrain: bool = False,
        infer_split_name: str = "test",
        stitch_scenes: bool = True,
        **infer_kwargs,
    ) -> None:
        if mode == "train":
            if retrain or not ckpt_path.exists():
                self.train(landcover_manifest, splits_path, ckpt_path)
            else:
                print(f"Checkpoint already exists, skipping training: {ckpt_path}")
            return

        if mode == "infer":
            if retrain or not ckpt_path.exists():
                if retrain:
                    self.train(landcover_manifest, splits_path, ckpt_path)
                else:
                    raise RuntimeError(
                        "mode='infer' but checkpoint does not exist. "
                        "Set retrain=True or use mode='train_and_infer'."
                    )

            self.infer_split(
                landcover_manifest=landcover_manifest,
                splits_path=splits_path,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                split=infer_split_name,
                stitch_scenes=stitch_scenes,
                **infer_kwargs,
            )
            return

        if retrain or not ckpt_path.exists():
            self.train(landcover_manifest, splits_path, ckpt_path)
        else:
            print(f"Using existing land-cover checkpoint: {ckpt_path}")

        self.infer_split(
            landcover_manifest=landcover_manifest,
            splits_path=splits_path,
            ckpt_path=ckpt_path,
            out_dir=out_dir,
            split=infer_split_name,
            stitch_scenes=stitch_scenes,
            **infer_kwargs,
        )