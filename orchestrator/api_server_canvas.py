"""
Canvas-facing HTTP API for the land-cover workflow (LandCoverWorkflowMS).

Separate from orchestrator/api_server.py (deforestation, workflow_manager.py) --
this is the newer surface the block-canvas demo frontend talks to. See CLAUDE.md's
"Current focus: demo-canvas support" section before changing this file:

  - Only ever reaches inference against an already-trained checkpoint -- never
    triggers training (LandCoverWorkflowMS.infer_region() enforces this itself,
    raising if the checkpoint doesn't exist).
  - Every response includes FusionOutput.uncertainty / .per_modality (reduced to
    per-tile mean summary stats by infer_region()) for the canvas's confidence/xAI
    panel, not just the predicted raster.
  - "Removing a sensor" on the canvas means picking a different, separately
    trained sensor_mode checkpoint (see LANDCOVER_MODEL_REGISTRY below), not
    dropping a modality from a live request -- there is no single model that
    tolerates that today.

Run:
    EARTH_PROJECT_ROOT=$(pwd) uvicorn orchestrator.api_server_canvas:app --reload

Unlike api_server.py, PROJECT_ROOT here resolves correctly by default (this file
lives one level under the repo root, not two) -- EARTH_PROJECT_ROOT is still
honored so both servers can be pointed at the same root the same way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from orchestrator.workflow_manager_landcover import (
    LandCoverWorkflowMS,
    TilingConfigLandCover,
    TrainingConfigLandCover,
)
from eintelligence.data_prep.aoi import square_aoi

# ------- Config & Model Registry -------

env_root = os.environ.get("EARTH_PROJECT_ROOT")
PROJECT_ROOT = Path(env_root) if env_root else Path(__file__).resolve().parents[1]
print(f"[api_server_canvas] Using PROJECT_ROOT: {PROJECT_ROOT}")

DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# sensor_mode -> checkpoint + the TilingConfigLandCover it was trained with.
# bands/tile_size/stride must match what the checkpoint was actually trained on
# (mismatched band counts fail loudly via a state_dict shape error at load time,
# but a silent max_cloud/tile_size drift would just quietly degrade quality) --
# these mirror the notebook's landcover cell-2 config, which is what produced the
# current models/landcover_{s2,s1s2}.pt checkpoints.
#
# Note: sensor_mode does not currently change ingestion or model architecture --
# ingest_region() always fetches both S1 and S2 (CLAUDE.md item 4, not yet fixed),
# and LateFusionKernel always fuses both modalities regardless of this tag. Two
# sensor_mode checkpoints today differ only in trained weights, not in what data
# infer_region() pulls at request time.
LANDCOVER_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "s2": {
        "ckpt": "landcover_s2.pt",
        "tiling_cfg": TilingConfigLandCover(
            bands_s2=("B02", "B03", "B04", "B08"),
            bands_s1=("vv", "vh"),
            tile_size=256,
            stride=256,
            max_cloud=50,
            sensor_mode="s2",
        ),
    },
    "s1s2": {
        "ckpt": "landcover_s1s2.pt",
        "tiling_cfg": TilingConfigLandCover(
            bands_s2=("B02", "B03", "B04", "B08"),
            bands_s1=("vv", "vh"),
            tile_size=256,
            stride=256,
            max_cloud=50,
            sensor_mode="s1s2",
        ),
    },
}

# ------- request / response schemas -------


class AOIInput(BaseModel):
    kind: Literal["point10km", "geojson"] = Field(
        default="point10km",
        description="How to define AOI: 10 km square around a point, or a full GeoJSON geometry.",
    )
    lat: Optional[float] = None
    lon: Optional[float] = None
    geometry: Optional[Dict[str, Any]] = None  # GeoJSON geometry

    @model_validator(mode="after")
    def _check_aoi(self) -> "AOIInput":
        if self.kind == "point10km" and (self.lat is None or self.lon is None):
            raise ValueError("lat and lon must be provided when kind is 'point10km'")
        if self.kind == "geojson" and self.geometry is None:
            raise ValueError("geometry must be provided when kind is 'geojson'")
        return self


class LandCoverInferRequest(BaseModel):
    sensor_mode: str = Field(
        ..., description=f"Which trained checkpoint to use. One of {sorted(LANDCOVER_MODEL_REGISTRY)}."
    )
    start: str = Field(..., description="ISO date, e.g. '2023-06-01'.")
    end: str = Field(..., description="ISO date, e.g. '2023-08-01'.")
    aoi: AOIInput
    region_name: Optional[str] = Field(
        default=None,
        description="Folder name under data/corpus/canvas/ for this request's tiles/outputs. "
        "A request id is always appended, so re-using a name across requests never collides.",
    )
    max_tiles: Optional[int] = Field(default=None, description="Cap tiles processed, for quick debugging.")
    stitch_scenes: bool = True


class TileResult(BaseModel):
    tile_id: str
    scene_id: str
    pred_mask_url: str
    quicklook_url: str
    compare_quicklook_url: Optional[str] = None
    uncertainty: Optional[float] = None
    per_modality: Dict[str, float] = Field(default_factory=dict)


class LandCoverInferResponse(BaseModel):
    region_name: str
    sensor_mode: str
    manifest_url: str
    stitched_manifest_url: Optional[str] = None
    tile_count: int
    tiles: List[TileResult]


# ------- app -------

app = FastAPI(title="Earth Intelligence API - Canvas (Land Cover)", version="0.1.0")

# Permissive CORS: this API exists specifically for a separate browser-based canvas
# frontend to call. Fine for the demo; revisit before this is ever internet-facing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve data/ as static so clients can open predicted rasters/quicklooks/manifests
app.mount("/data", StaticFiles(directory=str(DATA_ROOT), html=False), name="data")

# ------- utility functions -------


def _build_aoi(aoi: AOIInput) -> Dict[str, Any]:
    if aoi.kind == "point10km":
        return square_aoi(aoi.lat, aoi.lon)
    return {"type": "Feature", "geometry": aoi.geometry, "properties": {}}


def _to_data_url(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(DATA_ROOT.resolve())
    except ValueError as e:
        raise RuntimeError(f"Output path is not under DATA_ROOT, can't build a /data URL for it: {resolved}") from e
    return f"/data/{rel.as_posix()}"


# ------- API endpoints -------


@app.get("/canvas/landcover/sensor_modes")
def landcover_sensor_modes() -> Dict[str, Any]:
    """What sensor-mode checkpoints are configured, and whether each is actually on disk."""
    return {
        mode: {
            "ckpt": entry["ckpt"],
            "available": (MODELS_ROOT / entry["ckpt"]).exists(),
        }
        for mode, entry in LANDCOVER_MODEL_REGISTRY.items()
    }


@app.post("/canvas/landcover/infer_region", response_model=LandCoverInferResponse)
def landcover_infer_region(req: LandCoverInferRequest) -> LandCoverInferResponse:
    """
    Canvas's "run inference on this AOI right now": ingest a fresh AOI/date range
    end-to-end and run inference over every tile, against an already-trained
    checkpoint for req.sensor_mode. Never trains.
    """
    if req.sensor_mode not in LANDCOVER_MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sensor_mode '{req.sensor_mode}'. Available: {sorted(LANDCOVER_MODEL_REGISTRY)}",
        )
    entry = LANDCOVER_MODEL_REGISTRY[req.sensor_mode]
    ckpt_path = MODELS_ROOT / entry["ckpt"]
    if not ckpt_path.exists():
        raise HTTPException(status_code=500, detail=f"Model checkpoint not found: {ckpt_path}")

    request_id = uuid.uuid4().hex[:10]
    region_name = f"{req.region_name or 'canvas'}_{request_id}"
    out_dir = DATA_ROOT / "corpus" / "canvas" / region_name

    aoi_geojson = _build_aoi(req.aoi)

    wf = LandCoverWorkflowMS(PROJECT_ROOT, entry["tiling_cfg"], TrainingConfigLandCover())

    try:
        result = wf.infer_region(
            aoi_geojson=aoi_geojson,
            start=req.start,
            end=req.end,
            ckpt_path=ckpt_path,
            region_name=region_name,
            out_dir=out_dir,
            max_tiles=req.max_tiles,
            stitch_scenes=req.stitch_scenes,
        )
    except RuntimeError as e:
        # infer_region()/ingest_region() raise RuntimeError for expected upstream
        # failures (no STAC items for the AOI/date range, empty splits, etc.).
        raise HTTPException(status_code=502, detail=str(e)) from e

    tiles = [
        TileResult(
            tile_id=t.tile_id,
            scene_id=t.scene_id,
            pred_mask_url=_to_data_url(t.pred_mask_path),
            quicklook_url=_to_data_url(t.quicklook_path),
            compare_quicklook_url=_to_data_url(t.compare_quicklook_path) if t.compare_quicklook_path else None,
            uncertainty=t.uncertainty,
            per_modality=t.per_modality,
        )
        for t in result["tiles"]
    ]

    return LandCoverInferResponse(
        region_name=result["region_name"],
        sensor_mode=req.sensor_mode,
        manifest_url=_to_data_url(result["manifest_path"]),
        stitched_manifest_url=_to_data_url(result["stitched_manifest_path"])
        if result["stitched_manifest_path"]
        else None,
        tile_count=len(tiles),
        tiles=tiles,
    )
