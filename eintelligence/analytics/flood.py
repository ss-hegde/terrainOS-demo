from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import rasterio
from shapely.geometry import shape, mapping
from rasterio.features import shapes
from shapely.ops import unary_union

def _pixel_area_m2(transform) -> float:
    # assumes square pixels; fallback if rotated/skewed
    return abs(transform.a * transform.e)

def flood_summary(prob: np.ndarray,
                  transform,
                  thr: float = 0.5,
                  slope_deg: Optional[np.ndarray] = None,
                  max_slope_deg: float = 5.0) -> Dict[str, Any]:
    """
    turn a prob map into actionable stats
    """
    mask = (prob >= thr).astype(np.uint8)
    if slope_deg is not None:
        mask = np.where(slope_deg <= max_slope_deg, mask, 0).astype(np.uint8)

    # area
    pa = _pixel_area_m2(transform)  # m² per pixel
    area_m2 = (mask > 0).sum() * pa
    area_ha = area_m2 / 1e4

    # vectorize (simple)
    results = []
    for geom, val in shapes(mask, mask=mask.astype(bool), transform=transform):
        if val != 1: continue
        results.append(shape(geom))
    merged = unary_union(results) if results else None

    return {
        "threshold": thr,
        "area_ha": float(area_ha),
        "patch_count": int(len(results)),
        "vector": mapping(merged) if merged else None
    }