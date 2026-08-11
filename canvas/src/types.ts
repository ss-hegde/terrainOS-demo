// Shapes confirmed against the live /openapi.json, not the (stale) example in
// canvas/CLAUDE.md's "Backend integration" -- the real response nests per-tile
// results under `tiles`, it isn't a flat object with pred_mask_url etc. at top level.
// Shared by App.tsx and the custom node components (ModelNode/OutputNode) -- defined
// once here rather than re-derived per consumer.

export interface SensorModeEntry {
  ckpt: string
  available: boolean
}

export type SensorModesResponse = Record<string, SensorModeEntry>

export interface TileResult {
  tile_id: string
  scene_id: string
  pred_mask_url: string
  quicklook_url: string
  compare_quicklook_url: string | null
  uncertainty: number | null
  per_modality: Record<string, number>
}

export interface LandCoverInferResponse {
  region_name: string
  sensor_mode: string
  manifest_url: string
  stitched_manifest_url: string | null
  tile_count: number
  tiles: TileResult[]
}

// POST /canvas/landcover/train's response (see root CLAUDE.md's train entry) --
// `checkpoint` is a filename under models/, not a URL, and this checkpoint is
// deliberately not wired into sensor_modes/infer_region (see ModelNode.tsx).
export interface LandCoverTrainResponse {
  checkpoint: string
  sensor_mode: string
  epochs: number
  lr: number
  train_loss: number
  val_loss: number
  mean_iou: number
  elapsed_seconds: number
}
