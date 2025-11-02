# eintelligence/data_prep/collate.py
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
