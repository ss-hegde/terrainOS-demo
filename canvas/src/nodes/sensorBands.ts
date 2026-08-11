// Kept in sync by hand with orchestrator/api_server_canvas.py's
// LANDCOVER_MODEL_REGISTRY -- the actual per-checkpoint TilingConfigLandCover the
// canvas's inference requests run against, NOT TilingConfigLandCover's own
// dataclass default. That default is bands_s1=("VV","VH") uppercase, but
// LANDCOVER_MODEL_REGISTRY explicitly overrides it to lowercase ("vv","vh") because
// the notebook that trained the current checkpoints used lowercase and Sentinel-1
// STAC asset keys are case-sensitive (see root CLAUDE.md's sensor_mode gotcha) --
// copying the dataclass default here instead was exactly that mistake repeated one
// layer up. Case is display-only here (never sent to the backend), but the source
// of truth is still LANDCOVER_MODEL_REGISTRY, not the class default, on principle:
// if that registry's bands_s2 ever drifts from TilingConfigLandCover's default too,
// these should follow the registry.
export const S1_BANDS = ['vv', 'vh'] as const
export const S2_BANDS = ['B02', 'B03', 'B04', 'B08'] as const

// "VV, VH (2ch)" for a short band list, "B02-B08 (4ch)" for a longer one -- exactly
// the two current cases, but derived from the actual arrays above rather than
// hardcoded strings so it can't silently drift from them.
export function bandSubtitle(bands: readonly string[]): string {
  const list = bands.length <= 2 ? bands.join(', ') : `${bands[0]}-${bands[bands.length - 1]}`
  return `${list} (${bands.length}ch)`
}
