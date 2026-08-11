import { useCallback, useEffect, useMemo, useState } from 'react'
import { ReactFlow } from '@xyflow/react'
import type { Node, Edge, NodeTypes } from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import type { LandCoverInferResponse, LandCoverTrainResponse, SensorModesResponse } from './types'
import { DataNode } from './nodes/DataNode'
import type { DataNodeData } from './nodes/DataNode'
import { ModelNode } from './nodes/ModelNode'
import type { ModelNodeData, ModelPaneMode, SensorMode } from './nodes/ModelNode'
import { OutputNode } from './nodes/OutputNode'
import type { OutputNodeData } from './nodes/OutputNode'
import { SensorNode } from './nodes/SensorNode'
import type { SensorNodeData } from './nodes/SensorNode'
import { S1_BANDS, S2_BANDS, bandSubtitle } from './nodes/sensorBands'

// orchestrator/api_server_canvas.py, run per canvas/CLAUDE.md's "Backend integration".
const BACKEND_URL = 'http://127.0.0.1:8000'

// Munich demo AOI/dates the current checkpoints (models/landcover_{s2,s1s2}.pt) were
// trained on -- see scripts/smoke_test_infer_region.py. Only *defaults* now -- the
// Data node's inputs pre-fill with these so the demo still works untouched out of the
// box, but the live values live in state below and are editable.
const DEFAULT_LAT = 48.1351
const DEFAULT_LON = 11.582
const DEFAULT_LOCATION_NAME = 'Munich'
const DEFAULT_START = '2023-06-01'
const DEFAULT_END = '2023-08-01'

// Train pane defaults -- 2 epochs is the exact value already verified live against
// the real backend (root CLAUDE.md's train entry: ~38s for epochs=2 against "s2"),
// 0.0001 is POST /train's own already-verified lr. Mirrors POST /train's server-side
// Field(ge=1,le=10) bound client-side too -- see validateTrainInputs below.
const DEFAULT_EPOCHS = 2
const DEFAULT_LR = 0.0001
const MIN_EPOCHS = 1
const MAX_EPOCHS = 10

// Registered once, outside the component -- @xyflow/react warns (and re-renders more
// than necessary) if nodeTypes is a fresh object every render.
const nodeTypes: NodeTypes = { data: DataNode, sensor: SensorNode, model: ModelNode, output: OutputNode }

// Basic validation only, per canvas/CLAUDE.md's philosophy -- no geocoding, no clamping,
// just tell the user what's wrong and let ModelNode disable the Run button on it.
function validateAoi(lat: number, lon: number, start: string, end: string): string | null {
  if (Number.isNaN(lat) || lat < -90 || lat > 90) return 'lat must be between -90 and 90'
  if (Number.isNaN(lon) || lon < -180 || lon > 180) return 'lon must be between -180 and 180'
  if (!start || !end) return 'start and end dates are required'
  if (!(new Date(start) < new Date(end))) return 'start date must be before end date'
  return null
}

// Mirrors POST /train's own Field(ge=1,le=10)/Field(gt=0) bounds -- don't just rely
// on its 422, same philosophy as validateAoi above.
function validateTrainInputs(epochs: number, lr: number): string | null {
  if (!Number.isInteger(epochs) || epochs < MIN_EPOCHS || epochs > MAX_EPOCHS) {
    return `epochs must be an integer between ${MIN_EPOCHS} and ${MAX_EPOCHS}`
  }
  if (!(lr > 0)) return 'learning rate must be greater than 0'
  return null
}

function blockLabel(title: string, subtitle: string) {
  return (
    <div>
      <div style={{ fontWeight: 600 }}>{title}</div>
      <div style={{ fontSize: 12, opacity: 0.75, marginTop: 2 }}>{subtitle}</div>
    </div>
  )
}

// Node *shells* -- id/type/position/width, i.e. exactly the structural part that
// needs to survive a sensor deletion. `data` here is a placeholder ({}); the real,
// always-fresh data payload for data/model/output/s1/s2 is overlaid at render time
// by `overlayNodeData` below (see the `nodes` useMemo in App) so it never goes stale
// without needing an effect to re-sync it. internal-data/vendor-b/analytics are
// static enough that their `data.label` is just set once here and never touched
// again -- overlayNodeData leaves anything it doesn't recognize alone.
//
// Layout (y positions) verified live via Playwright bounding boxes per the
// node-repositioning gotcha above, not eyeballed -- Data -> S1/S2 side by side ->
// Model -> Analytics row -> Output. Everything from Analytics down is sized against
// Model's *tallest realistic state*, not just its Train pane's empty inputs: a
// finished training result (checkpoint filename + 4 metric lines + the
// not-available-for-inference line) grows Model well past its pre-run Train-pane
// height, and unlike OutputNode's own post-result growth, ModelNode has no
// fitView-on-growth reflow -- Model/Analytics share the same x, so undersizing here
// is a real, visible overlap the first time someone actually finishes a training
// run, not just a Train-mode-vs-Infer-mode height difference.
const initialNodeShells: Node[] = [
  {
    id: 'internal-data',
    position: { x: 50, y: 790 },
    data: { label: blockLabel('Internal Data', 'CSV/spreadsheet upload, joined by AOI/date') },
  },
  {
    id: 'vendor-b',
    position: { x: 750, y: 790 },
    data: { label: blockLabel('Vendor B', 'Simulated second feed (mocked)') },
  },
  {
    id: 'analytics',
    position: { x: 400, y: 900 },
    data: { label: blockLabel('Analytics', 'Calibration thresholds + reconciliation') },
  },
  { id: 'data', type: 'data', position: { x: 400, y: 0 }, width: 220, data: {} },
  { id: 's1', type: 'sensor', position: { x: 290, y: 320 }, width: 180, data: {} },
  { id: 's2', type: 'sensor', position: { x: 510, y: 320 }, width: 180, data: {} },
  { id: 'model', type: 'model', position: { x: 400, y: 420 }, width: 220, data: {} },
  { id: 'output', type: 'output', position: { x: 400, y: 1020 }, width: 220, data: {} },
]

const initialEdges: Edge[] = [
  { id: 'data-s1', source: 'data', target: 's1' },
  { id: 'data-s2', source: 'data', target: 's2' },
  { id: 's1-model', source: 's1', target: 'model' },
  { id: 's2-model', source: 's2', target: 'model' },
  { id: 'model-analytics', source: 'model', target: 'analytics' },
  { id: 'analytics-output', source: 'analytics', target: 'output' },
  { id: 'internal-data-analytics', source: 'internal-data', target: 'analytics' },
  { id: 'vendor-b-analytics', source: 'vendor-b', target: 'analytics' },
]

interface OverlayContext {
  dataNodeData: DataNodeData
  modelNodeData: ModelNodeData
  outputNodeData: OutputNodeData
  onDeleteS1: () => void
  onDeleteS2: () => void
}

function overlayNodeData(shell: Node, ctx: OverlayContext): Node {
  switch (shell.id) {
    case 'data':
      return { ...shell, data: ctx.dataNodeData }
    case 'model':
      return { ...shell, data: ctx.modelNodeData }
    case 'output':
      return { ...shell, data: ctx.outputNodeData }
    case 's1':
      return {
        ...shell,
        data: { title: 'Sentinel-1', subtitle: bandSubtitle(S1_BANDS), onDelete: ctx.onDeleteS1 } satisfies SensorNodeData,
      }
    case 's2':
      return {
        ...shell,
        data: { title: 'Sentinel-2', subtitle: bandSubtitle(S2_BANDS), onDelete: ctx.onDeleteS2 } satisfies SensorNodeData,
      }
    default:
      return shell
  }
}

function App() {
  const [sensorModes, setSensorModes] = useState<SensorModesResponse | null>(null)
  const [sensorModesError, setSensorModesError] = useState<string | null>(null)

  // AOI/date/location-name state -- lives here (not the Data node) same as
  // sensorModes above, and flows down into DataNode via its data prop.
  const [lat, setLat] = useState(DEFAULT_LAT)
  const [lon, setLon] = useState(DEFAULT_LON)
  const [locationName, setLocationName] = useState(DEFAULT_LOCATION_NAME)
  const [startDate, setStartDate] = useState(DEFAULT_START)
  const [endDate, setEndDate] = useState(DEFAULT_END)
  const aoiValidationError = useMemo(
    () => validateAoi(lat, lon, startDate, endDate),
    [lat, lon, startDate, endDate],
  )

  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState<string | null>(null)
  const [result, setResult] = useState<LandCoverInferResponse | null>(null)

  // Train pane -- separate from the infer state above (different endpoint,
  // different response shape), but same running/error/result pattern.
  const [paneMode, setPaneMode] = useState<ModelPaneMode>('infer')
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS)
  const [lr, setLr] = useState(DEFAULT_LR)
  const trainInputError = useMemo(() => validateTrainInputs(epochs, lr), [epochs, lr])
  const [trainRunning, setTrainRunning] = useState(false)
  const [trainError, setTrainError] = useState<string | null>(null)
  const [trainResult, setTrainResult] = useState<LandCoverTrainResponse | null>(null)

  // The real state: which nodes/edges currently exist. Structural only (id/type/
  // position) -- everything else (input values, run status, ...) lives in the state
  // above and gets overlaid onto these shells at render time by the `nodes` useMemo
  // below. This is what makes sensor deletion possible: before this pass `nodes`
  // was a useMemo computed fresh every render from scratch, so there was nothing to
  // actually remove a node *from*.
  const [nodeShells, setNodeShells] = useState<Node[]>(initialNodeShells)
  const [edges, setEdges] = useState<Edge[]>(initialEdges)

  const hasS1 = nodeShells.some((n) => n.id === 's1')
  const hasS2 = nodeShells.some((n) => n.id === 's2')

  // Deleting a sensor node removes it AND every edge that references it (both
  // data->sensor and sensor->model) -- a dangling edge (referencing a node id that
  // no longer exists) is a real @xyflow/react error, not just a visual glitch, so
  // both removals happen together here rather than trusting the edges to get
  // cleaned up some other way.
  const removeSensor = useCallback((id: 's1' | 's2') => {
    setNodeShells((prev) => prev.filter((n) => n.id !== id))
    setEdges((prev) => prev.filter((e) => e.source !== id && e.target !== id))
  }, [])
  const onDeleteS1 = useCallback(() => removeSensor('s1'), [removeSensor])
  const onDeleteS2 = useCallback(() => removeSensor('s2'), [removeSensor])

  // sensor_mode is derived from which sensor nodes are present, not chosen from a
  // dropdown anymore -- see root CLAUDE.md's sensor_mode section and ModelNode.tsx.
  const derivedMode: SensorMode | null = hasS1 && hasS2 ? 's1s2' : hasS2 ? 's2' : hasS1 ? 's1' : null
  const sensorCount = (hasS1 ? 1 : 0) + (hasS2 ? 1 : 0)
  // 512 channels per sensor's backbone output, concatenated pre-fusion -- the exact
  // shape that caused the proj-layer mismatch bug fixed earlier this session.
  const preFusionWidth = sensorCount > 0 ? sensorCount * 512 : null

  useEffect(() => {
    fetch(`${BACKEND_URL}/canvas/landcover/sensor_modes`)
      .then((res) => {
        if (!res.ok) throw new Error(`GET sensor_modes failed: ${res.status}`)
        return res.json() as Promise<SensorModesResponse>
      })
      .then(setSensorModes)
      .catch((err) => setSensorModesError(String(err)))
  }, [])

  // Still fetched on mount (above) -- just no longer used to populate a dropdown,
  // only to check whether the *derived* mode's checkpoint is actually available
  // before letting Run/Train fire a request we already know will fail (e.g. "s1"
  // today -- LANDCOVER_MODEL_REGISTRY, and so this response, has no "s1" entry at
  // all). Shared between Run and Train (factored out, not duplicated) since both
  // need the same "is this even a selectable sensor_mode" answer -- Run and Train
  // each layer their own additional, mode-specific check on top of it below.
  const sensorModeGuardReason: string | null = useMemo(() => {
    if (derivedMode === null) return 'No sensor selected -- add S1 and/or S2 to the canvas.'
    if (!sensorModes) return sensorModesError ? `sensor_modes unavailable: ${sensorModesError}` : 'Checking sensor_modes…'
    if (!sensorModes[derivedMode]?.available) return `sensor_mode "${derivedMode}" has no checkpoint available.`
    return null
  }, [derivedMode, sensorModes, sensorModesError])

  // Run additionally needs a valid AOI/date range (it ingests against them).
  const runDisabledReason: string | null = aoiValidationError ?? sensorModeGuardReason

  // Train doesn't touch the AOI/date fields at all -- POST /train takes no AOI,
  // it trains against the fixed pooled corpus -- so it deliberately does NOT gate
  // on aoiValidationError, only on its own epochs/lr bounds plus the shared guard.
  const trainDisabledReason: string | null = trainInputError ?? sensorModeGuardReason

  const runInference = useCallback(async () => {
    if (derivedMode === null) return // Run is disabled in this case; guard anyway
    setRunning(true)
    setRunError(null)
    setResult(null)
    try {
      const res = await fetch(`${BACKEND_URL}/canvas/landcover/infer_region`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sensor_mode: derivedMode,
          start: startDate,
          end: endDate,
          aoi: { kind: 'point10km', lat, lon },
          region_name: locationName,
          max_tiles: 4, // keep this demo run quick; field exists exactly for this
        }),
      })
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(`POST infer_region failed: ${res.status} ${detail}`)
      }
      const data = (await res.json()) as LandCoverInferResponse
      setResult(data)
    } catch (err) {
      setRunError(String(err))
    } finally {
      setRunning(false)
    }
  }, [derivedMode, lat, lon, startDate, endDate, locationName])

  const runTraining = useCallback(async () => {
    if (derivedMode === null) return // Train is disabled in this case; guard anyway
    setTrainRunning(true)
    setTrainError(null)
    setTrainResult(null)
    try {
      const res = await fetch(`${BACKEND_URL}/canvas/landcover/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // No AOI/dates/region_name -- POST /train doesn't ingest anything, it
        // trains against the fixed pooled corpus (see root CLAUDE.md).
        body: JSON.stringify({ sensor_mode: derivedMode, epochs, lr }),
      })
      if (!res.ok) {
        const detail = await res.text()
        throw new Error(`POST train failed: ${res.status} ${detail}`)
      }
      const data = (await res.json()) as LandCoverTrainResponse
      setTrainResult(data)
    } catch (err) {
      setTrainError(String(err))
    } finally {
      setTrainRunning(false)
    }
  }, [derivedMode, epochs, lr])

  const dataNodeData: DataNodeData = useMemo(
    () => ({
      title: 'Data',
      subtitle: 'S1, S2, AOI + date range',
      lat,
      lon,
      locationName,
      startDate,
      endDate,
      onChangeLat: setLat,
      onChangeLon: setLon,
      onChangeLocationName: setLocationName,
      onChangeStartDate: setStartDate,
      onChangeEndDate: setEndDate,
      validationError: aoiValidationError,
    }),
    [lat, lon, locationName, startDate, endDate, aoiValidationError],
  )

  const modelNodeData: ModelNodeData = useMemo(
    () => ({
      title: 'Model',
      subtitle: 'Sensor mode + task selection',
      derivedMode,
      preFusionWidth,
      sensorModesError,
      paneMode,
      onChangePaneMode: setPaneMode,
      runDisabledReason,
      onRun: runInference,
      running,
      epochs,
      onChangeEpochs: setEpochs,
      lr,
      onChangeLr: setLr,
      trainDisabledReason,
      onTrain: runTraining,
      trainRunning,
      trainError,
      trainResult,
    }),
    [
      derivedMode,
      preFusionWidth,
      sensorModesError,
      paneMode,
      runDisabledReason,
      runInference,
      running,
      epochs,
      lr,
      trainDisabledReason,
      runTraining,
      trainRunning,
      trainError,
      trainResult,
    ],
  )

  const outputNodeData: OutputNodeData = useMemo(
    () => ({
      title: 'Output',
      subtitle: 'Map, quicklook, confidence panel',
      running,
      runError,
      result,
      backendUrl: BACKEND_URL,
    }),
    [running, runError, result],
  )

  // The array actually passed to <ReactFlow>: nodeShells (the real state) with fresh
  // data overlaid every render. Recomputed from scratch each time, but cheap (8
  // nodes) -- no need for the effect-based sync this would otherwise call for.
  const nodes: Node[] = useMemo(
    () =>
      nodeShells.map((shell) =>
        overlayNodeData(shell, { dataNodeData, modelNodeData, outputNodeData, onDeleteS1, onDeleteS2 }),
      ),
    [nodeShells, dataNodeData, modelNodeData, outputNodeData, onDeleteS1, onDeleteS2],
  )

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView colorMode="system" />
    </div>
  )
}

export default App
