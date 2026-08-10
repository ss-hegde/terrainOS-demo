# earth-api

## Overview

The Earth Intelligence Platform (earth-api) is a modular, plug-and-play framework that brings together Earth observation (EO) data, Artificial Intelligence (AI), and API orchestration to transform raw satellite imagery into actionable insights.

It is designed as an open, scalable system with five conceptual layers:

1. Common Data & Preprocessing Layer – Fetches, cleans, and tiles satellite data (Sentinel-1/2, Landsat, ERA5).
2. Shared Geospatial Backbone – A sensor-agnostic encoder that extracts spatial and temporal features.
3. Task-Specific Adapters/Heads – Lightweight neural modules for domain tasks (e.g., deforestation, floods, crop stress).
4. Rules & Analytics Layer – Hybrid rule-based or statistical analysis modules for simple insights.
5. Orchestrator & API Layer – A unified interface (earth.query()) that enables end-to-end workflow execution via a FastAPI service.

---

## Current Capabilities

-  STAC-based Sentinel-1 + Sentinel-2 ingestion using the Microsoft Planetary Computer API
- Scene tiling and management as Cloud-Optimized GeoTIFFs (COGs)
- Temporal pairing of multi-scene satellite imagery
- Change detection adapter combining a ResNet backbone with a U-Net head
- End-to-end orchestration through the Workflow class
- Visualization outputs (NDVI checks, GeoTIFF masks, quicklook PNG overlays)
- Fusion of multi-sensor data (Sentinel-1 SAR + Sentinel-2 optical) for improved change detection and land cover monitoring

``` 
earth.query(task="deforestation", lat=..., lon=..., start=..., end=...)
```
---
## Architecture

```
eintelligence/
├── data_prep/          # Layer 1: Fetching, tiling, manifest, and pairing logic
├── backbone/           # Layer 2: Shared CNN encoder (ResNet)
├── adapters/           # Layer 3: Task-specific heads (e.g., U-Net for change detection)
├── analytics/          # Layer 4: Rule-based or hybrid analytics modules
├── orchestrator/       # Layer 5: Workflows and FastAPI service
│   ├── workflow_manager.py   # End-to-end orchestration logic
│   ├── api_server.py         # FastAPI implementation (earth.query)
│   └── ...
├── fusion/             # Multi-sensor fusion kernel (e.g., S1+S2)
├── utils/              # Logging, debugging, and helper tools
├── models/             # Trained model checkpoints
├── data/               # Downloaded tiles, manifests, predictions
└── notebooks/          # Jupyter notebooks for exploration and prototyping

```
---
## Core Components

| Layer | Description | Key Modules |
|-------|-------------|-------------|
| Data & Preprocessing | Fetches and tiles EO data via STAC |`fetch_multi_data.py`, `tiler_streaming.py` |
| Backbone | Shared encoder (ResNet-based) | `backbone/resnet_encoder.py` |
| Adapters | Task-specific heads (U-Net) | `adapters/change_head.py` |
| Analytics | NDVI and change metrics | `analytics/ndvi.py` |
| Orchestrator | Manages workflow execution | `orchestrator/workflow_manager.py` |
| Fusion | Multi-sensor fusion logic | `fusion/fusion_kernel.py` |
| API Layer | FastAPI serving earth.query() | `orchestrator/api_server.py` |

---
## Example Usage (Notebook or Script)

```
from pathlib import Path
import os, sys

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from eintelligence.data_prep.aoi import square_aoi
from orchestrator.workflow_manager_multisensor_v1 import (
    DeforestationWorkflowMS, TilingConfigMS, TilingConfigS1,
    TrainingConfig, FloodWorkflowS1
)

# 1. Define a 10×10 km AOI (example: Munich, Germany)
aoi = square_aoi(48.1351, 11.5820)

# 2. Select task + sensor mode
CASE = 1                   # 0=S2 deforestation, 1=multi-sensor deforestation, 2=flood S1
sensor_mode = "s1s2"         # "s2", "s1", or "s1s2"
case_name = "deforestation" if CASE == 1 else "flood"

# 3. Configure workflow
if CASE == 1:
    tiling_cfg = TilingConfigMS(
        bands_s2=("B02","B03","B04","B08"),
        bands_s1=("vv","vh"),
        tile_size=256, stride=256,
        max_cloud=50,
        sensor_mode=sensor_mode
    )
    wf = DeforestationWorkflowMS(project_root, tiling_cfg, TrainingConfig(), skip_to_pairing=True)
else:
    tiling_cfg = TilingConfigS1(tile_size=256, stride=256)
    wf = FloodWorkflowS1(project_root, tiling_cfg, TrainingConfig(), skip_to_pairing=False)

# 4. Build multi-scene dataset + temporal pairs
region = f"location_{case_name}_{sensor_mode}"
pairs_manifest = wf.build_data(
    aoi_geojson=aoi,
    start="2023-06-01",
    end="2023-08-01",
    region_name=region
)

# 5. Run model inference (or set retrain=True to fine-tune)
ckpt_path = Path(project_root) / "models" / f"{case_name}_{sensor_mode}_adapter.pt"
out_dir   = Path(project_root) / "data" / region / f"pred_{case_name}_{sensor_mode}"

wf.run(pairs_manifest, ckpt_path, out_dir, retrain=False, prob_thresh=0.5)


```

### Output

After running the workflow, you will find:
```
data/
  └── location_deforestation_s1s2/
        ├── S1/scene_1/tiles_s1/
        ├── S2/scene_2/tiles_s2/
        ├── collection_manifest.json
        ├── pairs_manifest.json
        └── pred_deforestation_s1s2/
              ├── *.tif         # georeferenced mask tiles
              └── quicklooks/
                    ├── *_overlay.png
                    ├── *_mask.png

```

---
## Future Work

### Implemented
- Multi-scene Sentinnel-1 and Sentinel-2 ingestion via STAC
- Automated tiling manifesting, matching S1 and S2 scenes temporally, and pairing for change detection
- Deforestation adapter (ResNet + U-Net)
- Land cover classification adapter (ResNet + U-Net)
- FastAPI endpoint with logging and visualization

### Next Steps
- Integrate additional sensors (e.g., Landsat, MODIS)
- Introduce self-supervised pretraining for the shared backbone 
- Add explainability layers
- Extend earth.query() to handle multi-task inference

---
## Vision
To democratize access to Earth observation intelligence by providing a unified, extensible, and developer-friendly platform that turns satellite data into real-time environmental insights.