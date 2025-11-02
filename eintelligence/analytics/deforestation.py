from __future__ import annotations
import numpy as np  
from typing import Dict, Any, Optional, Tuple

def _pixel_area_from_affine(A) -> float:
    # affine.a * affine.e for north-up rasters
    return abs(A.a * A.e)

def _binarize(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(np.uint8)

def _morph_clean(mask: np.ndarray, open_sz: int = 3) -> np.ndarray:
    try:
        import skimage.morphology as morph
        se = morph.square(open_sz)
        return morph.opening(mask.astype(bool), se).astype(np.uint8)
    except Exception:
        return mask  # skip if skimage not available
    
def _vectorize(mask: np.ndarray, affine, min_patch_ha: float, pix_area_m2: float):
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union
    from skimage.measure import label, regionprops
    labeled = label(mask)
    polys = []

    for r in regionprops(labeled):
        area_m2 = r.area * pix_area_m2
        if area_m2 >= (min_patch_ha * 10_000.0):
            # simple convex hull around pixels
            coords = [(c[1], c[0]) for c in r.coords]
            poly = MultiPoint(coords).convex_hull
            polys.append(poly)
    if len(polys) == 0:
        return None, 0.0, []
    
    footprint = unary_union(polys)
    total_area_ha = float(footprint.area * pix_area_m2 / 10_000.0)
    return footprint, total_area_ha, polys

def deforestation_summary(
    change_prob: np.ndarray, # [H,W] float32 0..1
    meta: Dict[str, Any],   # includes 'transform' (affine), optional 'pixel_area_m2'
    *,
    threshold: float = 0.5,
    min_patch_ha: float = 1.0,
    ndvi_pre: Optional[np.ndarray] = None, # [H,W] float32
    landcover: Optional[np.ndarray] = None, # [H,W] uint8
    forest_ids: Tuple[int,...] = (20, 30, 40, 50), # landcover IDs considered forest
    s1_coh_drop: Optional[np.ndarray] = None, # [H,W] float32
    coh_threshold: Optional[float] = 0.2,
    ) -> Dict[str, Any]:
    """
    Outputs an auditable summary independent of the model training.
    """
    # base mask
    m = _binarize(change_prob, threshold)
    m = _morph_clean(m, open_sz=3)

    # gates
    forest_gate = np.ones_like(m, dtype=bool)
    if landcover is not None:
        forest_gate &= np.isin(landcover, forest_ids)
    if ndvi_pre is not None:
        forest_gate &= (ndvi_pre >= 0.6)  # conservative forest
    if s1_coh_drop is not None:
        forest_gate &= (s1_coh_drop >= coh_threshold)

    valid = m.astype(bool) & forest_gate
    # geo+area
    A = meta.get("transform")
    pix_area_m2 = meta.get("pixel_area_m2", _pixel_area_from_affine(A))
    footprint, area_ha, polys = _vectorize(valid, A, min_patch_ha, pix_area_m2)

    return {
        "deforestation_area_ha": float(area_ha),
        "num_patches": int(len(polys)),
        "quality_flags": {
            "ndvi_gate": ndvi_pre is not None,
            "landcover_gate": landcover is not None,
            "coherence_gate": s1_coh_drop is not None,
        },
        "threshold_used": float(threshold),
        "footprint": footprint,  # shapely geometry or None
    }

# Optional - calibrate threshold on validation set to match target area

def calibrate_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    probs: (N,H,W) float
    targets: (N,H,W) {0,1}
    Returns F1-optimal threshold in [0,1].
    """

    from sklearn.metrics import f1_score
    best_threshold, best_f1 = 0.5, -1
    for threshold in np.linspace(0.1, 0.9, 17):
        preds = (probs >= threshold).astype(np.uint8).reshape(probs.shape[0], -1)
        gts = targets.astype(np.uint8).reshape(targets.shape[0], -1)
        f1 = f1_score(gts.flatten(), preds.flatten(), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)        
    return best_threshold