"""
Canvas-facing HTTP API for the land-cover workflow (LandCoverWorkflowMS).

Separate from orchestrator/api_server.py (deforestation, workflow_manager.py) --
this is the newer surface the block-canvas demo frontend talks to. See CLAUDE.md's
"Current focus: demo-canvas support" section before changing this file:

  - /canvas/landcover/infer_region only ever reaches inference against an
    already-trained checkpoint -- never triggers training
    (LandCoverWorkflowMS.infer_region() enforces this itself, raising if the
    checkpoint doesn't exist).
  - Every response includes FusionOutput.uncertainty / .per_modality (reduced to
    per-tile mean summary stats by infer_region()) for the canvas's confidence/xAI
    panel, not just the predicted raster.
  - "Removing a sensor" on the canvas means picking a different, separately
    trained sensor_mode checkpoint (see LANDCOVER_MODEL_REGISTRY below), not
    dropping a modality from a live request -- there is no single model that
    tolerates that today.
  - /canvas/landcover/train (added later, see root CLAUDE.md) is the one path that
    *does* train -- against the existing pooled manifest/splits under data/corpus/,
    never against a canvas request's own ingested tiles. It writes a fresh,
    uniquely-named checkpoint under models/ and never touches landcover_s2.pt /
    landcover_s1s2.pt or LANDCOVER_MODEL_REGISTRY -- closing the
    train-then-immediately-infer loop is a separate, later step.

Run (same pattern CLAUDE.md's "Setup & commands" documents for api_server.py):
    EARTH_PROJECT_ROOT=$(pwd) uvicorn orchestrator.api_server_canvas:app --reload

Host/port are NOT hardcoded here -- there's no `uvicorn.run(...)` call in this file,
only `app = FastAPI(...)`, so the command above binds to uvicorn's own CLI defaults:
    host 127.0.0.1, port 8000  ->  http://127.0.0.1:8000
Override with --host/--port if needed, e.g.:
    EARTH_PROJECT_ROOT=$(pwd) uvicorn orchestrator.api_server_canvas:app --host 0.0.0.0 --port 8010 --reload

Unlike api_server.py, PROJECT_ROOT here resolves correctly by default (this file
lives one level under the repo root, not two) -- EARTH_PROJECT_ROOT is still
honored so both servers can be pointed at the same root the same way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from orchestrator.workflow_manager_landcover import (
    LandCoverWorkflowMS,
    TilingConfigLandCover,
    TrainingConfigLandCover,
    ingestion_fingerprint,
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

# The existing pooled training corpus (built by the notebook -- see root CLAUDE.md
# and notebooks/05_application_workflow_landcover.ipynb's corpus_dir) -- /train
# trains against this, it never builds a manifest from a canvas request's own
# ingested tiles the way infer_region() does.
POOLED_MANIFEST_PATH = DATA_ROOT / "corpus" / "landcover_manifest_multisensor.json"
POOLED_SPLITS_PATH = DATA_ROOT / "corpus" / "landcover_splits.json"

# This is a live demo -- nobody should be able to kick off a 100-epoch run by
# accident (or by a future frontend passing through an unvalidated field). 10 is
# generous enough to see a real mIoU trend on a small canvas-triggered run without
# tying up the one shared server for an unbounded amount of time.
MIN_TRAIN_EPOCHS = 1
MAX_TRAIN_EPOCHS = 10

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


class LandCoverTrainRequest(BaseModel):
    sensor_mode: str = Field(
        ..., description=f"Which sensor_mode's tiling_cfg to train with. One of {sorted(LANDCOVER_MODEL_REGISTRY)}."
    )
    epochs: int = Field(
        ...,
        ge=MIN_TRAIN_EPOCHS,
        le=MAX_TRAIN_EPOCHS,
        description=f"Number of training epochs, bounded to [{MIN_TRAIN_EPOCHS}, {MAX_TRAIN_EPOCHS}] server-side "
        "-- this is a live shared demo, not a trusted-frontend-only guard.",
    )
    lr: float = Field(..., gt=0, description="Learning rate, passed straight to TrainingConfigLandCover.")


class LandCoverTrainResponse(BaseModel):
    checkpoint: str  # filename under models/, e.g. "canvas_train_s2_ab12cd34ef.pt"
    sensor_mode: str
    epochs: int
    lr: float
    train_loss: float
    val_loss: float
    mean_iou: float
    elapsed_seconds: float


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

    # This request's OUTPUT naming (out_dir, and the region_name echoed in the
    # response) -- stays UUID-suffixed and unique per request, unchanged from
    # before. This is what actually guarantees concurrent/repeated calls never
    # collide on disk; it's completely independent of the ingestion cache below.
    request_id = uuid.uuid4().hex[:10]
    output_region_name = f"{req.region_name or 'canvas'}_{request_id}"
    out_dir = DATA_ROOT / "corpus" / "canvas" / output_region_name

    aoi_geojson = _build_aoi(req.aoi)

    # Ingestion (S1/S2 STAC fetch + tiling) is a separate, shared cache keyed by
    # what actually determines its output -- see ingestion_fingerprint()/
    # ingest_region()'s per-sensor manifest reuse. Passing this (not
    # output_region_name) as infer_region()'s `region_name` is what lets two
    # requests for the same AOI/dates reuse the same ingested data on disk,
    # without touching out_dir's per-request uniqueness above.
    ingest_region_name = ingestion_fingerprint(aoi_geojson, req.start, req.end, entry["tiling_cfg"])

    wf = LandCoverWorkflowMS(PROJECT_ROOT, entry["tiling_cfg"], TrainingConfigLandCover())

    try:
        result = wf.infer_region(
            aoi_geojson=aoi_geojson,
            start=req.start,
            end=req.end,
            ckpt_path=ckpt_path,
            region_name=ingest_region_name,
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
        # This request's own unique name, not result["region_name"] -- that now
        # holds the shared ingestion fingerprint (see ingest_region_name above),
        # which is an internal cache key, not this request's identity.
        region_name=output_region_name,
        sensor_mode=req.sensor_mode,
        manifest_url=_to_data_url(result["manifest_path"]),
        stitched_manifest_url=_to_data_url(result["stitched_manifest_path"])
        if result["stitched_manifest_path"]
        else None,
        tile_count=len(tiles),
        tiles=tiles,
    )


@app.post("/canvas/landcover/train", response_model=LandCoverTrainResponse)
def landcover_train(req: LandCoverTrainRequest) -> LandCoverTrainResponse:
    """
    Train a fresh land-cover checkpoint against the existing pooled training corpus
    (data/corpus/landcover_manifest_multisensor.json + landcover_splits.json --
    built by the notebook, not by this request). Deliberately separate from
    infer_region()'s canvas-request ingestion: this never touches a canvas request's
    own tiles, and infer_region() never touches the pooled corpus -- the two stay on
    opposite sides of that line.

    Writes to a brand-new, uniquely-named checkpoint under models/ -- never
    landcover_s2.pt/landcover_s1s2.pt, and this checkpoint is NOT added to
    LANDCOVER_MODEL_REGISTRY. Wiring a freshly trained checkpoint into inference is a
    separate, later step (see root CLAUDE.md).
    """
    if req.sensor_mode not in LANDCOVER_MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown sensor_mode '{req.sensor_mode}'. Available: {sorted(LANDCOVER_MODEL_REGISTRY)}",
        )
    entry = LANDCOVER_MODEL_REGISTRY[req.sensor_mode]

    if not POOLED_MANIFEST_PATH.exists() or not POOLED_SPLITS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pooled training corpus not found: {POOLED_MANIFEST_PATH} / {POOLED_SPLITS_PATH}",
        )

    # New, uniquely-named checkpoint -- see the isolation note in the docstring above.
    ckpt_path = MODELS_ROOT / f"canvas_train_{req.sensor_mode}_{uuid.uuid4().hex[:10]}.pt"

    # Reuse LANDCOVER_MODEL_REGISTRY's own tiling_cfg for this sensor_mode rather than
    # constructing a fresh TilingConfigLandCover -- that's exactly how the bands_s1
    # case mismatch happened before (root CLAUDE.md's sensor_mode gotcha): a
    # from-scratch TilingConfigLandCover silently reverts to the dataclass's
    # uppercase ("VV","VH") default instead of the lowercase the checkpoints/STAC
    # asset keys actually need.
    #
    # amp=False explicitly -- TrainingConfigLandCover's own dataclass default is
    # amp=True, which is the known live-NaN issue (root CLAUDE.md's "Known issue").
    # Leaving that implicit here would be the same kind of default-trusting mistake.
    train_cfg = TrainingConfigLandCover(lr=req.lr, num_epochs=req.epochs, amp=False)
    wf = LandCoverWorkflowMS(PROJECT_ROOT, entry["tiling_cfg"], train_cfg)

    start = time.monotonic()
    try:
        metrics = wf.train(POOLED_MANIFEST_PATH, POOLED_SPLITS_PATH, ckpt_path)
    except RuntimeError as e:
        # train()/_load_split_indices() raise RuntimeError for expected upstream
        # failures (empty train/val split, no tile records, etc.) -- same pattern as
        # infer_region()'s error handling above.
        raise HTTPException(status_code=502, detail=str(e)) from e
    elapsed_seconds = time.monotonic() - start

    return LandCoverTrainResponse(
        checkpoint=ckpt_path.name,
        sensor_mode=req.sensor_mode,
        epochs=req.epochs,
        lr=req.lr,
        train_loss=metrics.train_loss,
        val_loss=metrics.val_loss,
        mean_iou=metrics.mean_iou,
        elapsed_seconds=elapsed_seconds,
    )
