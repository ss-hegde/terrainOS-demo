from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

import torch

@dataclass
class SegmentationMetrics:
    per_class_iou: torch.Tensor        # [C]
    mean_iou: float
    per_class_f1: torch.Tensor         # [C]
    macro_f1: float
    overall_accuracy: float
    support: torch.Tensor              # [C], number of gt pixels per class

def _confusion_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute confusion matrix [C, C] from logits and labels.
    Rows: ground truth classes, Columns: predicted classes.
    """
    # logits: [B, C, H, W], labels: [B, H, W]
    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)  # [B, H, W]

        # Flatten
        preds = preds.view(-1)
        labels = labels.view(-1)

        if ignore_index is not None:
            mask = labels != ignore_index
            preds = preds[mask]
            labels = labels[mask]

        # Filter out-of-range labels if any
        valid = (labels >= 0) & (labels < num_classes)
        preds = preds[valid]
        labels = labels[valid]

        # Map (gt, pred) pairs to indices in [0, C*C-1]
        idx = labels * num_classes + preds
        conf = torch.bincount(idx, minlength=num_classes * num_classes)
        conf = conf.view(num_classes, num_classes)

    return conf

def compute_segmentation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int] = None,
) -> SegmentationMetrics:
    """
    Compute per-class IoU, mean IoU, per-class F1, macro F1, and overall accuracy.
    Ignores pixels with labels == ignore_index.
    """
    conf = _confusion_from_logits(logits, labels, num_classes, ignore_index=ignore_index)
    conf = conf.float()

    # True positives, false positives, false negatives per class
    tp = torch.diag(conf)                         # [C]
    fp = conf.sum(dim=0) - tp                     # predicted as c but gt != c
    fn = conf.sum(dim=1) - tp                     # gt is c but predicted != c
    tn = conf.sum() - (tp + fp + fn)

    # IoU per class: tp / (tp + fp + fn)
    denom_iou = tp + fp + fn
    per_class_iou = torch.where(
        denom_iou > 0,
        tp / torch.clamp(denom_iou, min=1e-6),
        torch.zeros_like(tp),
    )
    mean_iou = float(per_class_iou.mean().item())

    # F1 per class: 2 * tp / (2*tp + fp + fn)
    denom_f1 = 2 * tp + fp + fn
    per_class_f1 = torch.where(
        denom_f1 > 0,
        2 * tp / torch.clamp(denom_f1, min=1e-6),
        torch.zeros_like(tp),
    )
    macro_f1 = float(per_class_f1.mean().item())

    # Overall accuracy: (sum tp) / (sum all non-ignored pixels)
    total_correct = tp.sum()
    total_pixels = conf.sum()
    overall_accuracy = float((total_correct / torch.clamp(total_pixels, min=1e-6)).item())

    support = conf.sum(dim=1)  # number of gt pixels per class

    return SegmentationMetrics(
        per_class_iou=per_class_iou,
        mean_iou=mean_iou,
        per_class_f1=per_class_f1,
        macro_f1=macro_f1,
        overall_accuracy=overall_accuracy,
        support=support,
    )