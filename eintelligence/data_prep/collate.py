# eintelligence/data_prep/collate.py
import torch
from eintelligence.fusion.kernel_base import FusionBatch

def collate_change(batch):
    # batch: list of dicts from MultiSensorChangeDataset
    import torch
    keys = ("s1","s2")
    out = {"t0":{}, "t1":{}, "meta":[]}
    for k in keys:
        out["t0"][k] = torch.stack([b["t0"][k] for b in batch], dim=0)
        out["t1"][k] = torch.stack([b["t1"][k] for b in batch], dim=0)
    out["target"] = torch.stack([b["target"] for b in batch], dim=0)
    out["meta"]   = [b["meta"] for b in batch]
    return out

def collate_landcover(batch):
    fb_list = [b["fusion_batch"] for b in batch]

    imagery_keys = fb_list[0].imagery.keys()
    mask_keys = fb_list[0].masks.keys()

    imagery = {
        k: torch.stack([fb.imagery[k] for fb in fb_list], dim=0)   # [B, C, H, W]
        for k in imagery_keys
    }
    masks = {
        k: torch.stack([fb.masks[k] for fb in fb_list], dim=0)     # [B, 1, H, W]
        for k in mask_keys
    }

    meta = {
        "tile_id": [fb.meta["tile_id"] for fb in fb_list],
        "datetime": [fb.meta["datetime"] for fb in fb_list],
        "row": [fb.meta["row"] for fb in fb_list],
        "col": [fb.meta["col"] for fb in fb_list],
        "height": [fb.meta["height"] for fb in fb_list],
        "width": [fb.meta["width"] for fb in fb_list],
        "s2_path": [fb.meta["s2_path"] for fb in fb_list],
        "s1_path": [fb.meta["s1_path"] for fb in fb_list],
        "worldcover_path": [fb.meta["worldcover_path"] for fb in fb_list],
    }

    fusion_batch = FusionBatch(
        imagery=imagery,
        masks=masks,
        meta=meta,
    )

    return {
        "fusion_batch": fusion_batch,
        "labels": torch.stack([b["labels"] for b in batch], dim=0),          # [B, H, W]
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),  # [B, H, W]
    }