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
        k: torch.stack([fb.imagery[k] for fb in fb_list], dim=0)
        for k in imagery_keys
    }
    masks = {
        k: torch.stack([fb.masks[k] for fb in fb_list], dim=0)
        for k in mask_keys
    }

    meta_keys = [
        "tile_id",
        "scene_id",
        "datetime",
        "row",
        "col",
        "height",
        "width",
        "aoi_id",
        "job_id",
        "group_id",
        "s1_path",
        "s2_path",
        "worldcover_path",
    ]
    meta = {
        k: [fb.meta.get(k) for fb in fb_list]
        for k in meta_keys
    }

    fusion_batch = FusionBatch(
        imagery=imagery,
        masks=masks,
        meta=meta,
    )

    return {
        "fusion_batch": fusion_batch,
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
    }