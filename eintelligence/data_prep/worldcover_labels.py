import numpy as np

# WorldCover 2021 class codes (Map band)
# 0: No data
# 10: Tree cover 
# 20: Shrubland 
# 30: Grassland 
# 40: Cropland
# 50: Built-up 
# 60: Bare/sparse 
# 70: Snow/Ice 
# 80: Water
# 90: Herbaceous wetland 
# 95: Mangroves 
# 100: Moss & lichen
# [10..100 codes from ESA WorldCover docs] [web:85][web:87]

# Reduced legend: 6 classes + ignore
REDUCED_LC_IGNORE = 255  # ignore_index

WORLD_COVER_TO_REDUCED = {
    10: 0,  # Forest / tree
    95: 0,  # Mangroves -> Forest
    20: 1,  # Other vegetated
    30: 1,  # Other vegetated
    90: 1,  # Other vegetated (wetland veg)
    100: 1, # Other vegetated (moss/lichen)
    40: 2,  # Cropland
    50: 3,  # Built-up
    60: 4,  # Bare
    80: 5,  # Water
    # 0 (no data) and 70 (snow/ice) will be ignored
}

def worldcover_to_reduced(labels: np.ndarray) -> np.ndarray:
    """
    Convert ESA WorldCover 'Map' codes to reduced classes [0..5] + ignore (255).
    labels: np.ndarray of int (e.g. read from WorldCover COG)
    """
    out = np.full_like(labels, REDUCED_LC_IGNORE, dtype=np.uint8)
    for src, dst in WORLD_COVER_TO_REDUCED.items():
        out[labels == src] = dst
    # ignore 0 (no data) and 70 (snow/ice) by leaving them as REDUCED_LC_IGNORE
    return out

