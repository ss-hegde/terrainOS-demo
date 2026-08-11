import { Handle, Position } from '@xyflow/react'
import type { Node, NodeProps } from '@xyflow/react'
import type { LandCoverTrainResponse } from '../types'
import { nodeBoxStyle, nodeErrorStyle, nodeSubtitleStyle, nodeTitleStyle } from './nodeStyles'

export type SensorMode = 's1' | 's2' | 's1s2'
export type ModelPaneMode = 'infer' | 'train'

const fieldLabelStyle = { display: 'block', marginBottom: 6 }
const fieldInputStyle = { display: 'block', width: '100%', marginTop: 2, boxSizing: 'border-box' as const }

// State lives in App and flows down through `data`, same as before -- this component
// stays a plain renderer. The sensor-mode <select> is gone: sensor_mode is now
// *derived* in App from which of the S1/S2 nodes are present in canvas state (see
// App.tsx's `derivedMode`), not chosen here. This node only displays the result and
// the shape it implies.
//
// Infer and Train are two panes toggled by `paneMode` -- sensor_mode/pre-fusion
// width stay visible in both (both requests are built from the same derived mode),
// only the mode-specific controls below them swap.
export interface ModelNodeData extends Record<string, unknown> {
  title: string
  subtitle: string
  derivedMode: SensorMode | null
  // 512 channels per present sensor's backbone output, concatenated pre-fusion --
  // the exact number that caused the proj-layer shape-mismatch bug fixed earlier
  // this session (see root CLAUDE.md's sensor_mode section). null when no sensor is
  // present, since there's nothing to concatenate.
  preFusionWidth: number | null
  sensorModesError: string | null

  paneMode: ModelPaneMode
  onChangePaneMode: (mode: ModelPaneMode) => void

  // Infer pane. Set by App whenever Run should be disabled for a reason the user
  // needs to see (invalid AOI, no sensor selected, or GET /sensor_modes says the
  // derived mode's checkpoint isn't available) -- null means Run is enabled.
  runDisabledReason: string | null
  onRun: () => void
  running: boolean

  // Train pane. epochs/lr are plain canvas state (like the Data node's fields) --
  // client-side bounded to mirror POST /train's own Field(ge=1,le=10) rather than
  // just relying on its 422. trainDisabledReason shares its sensor_mode/availability
  // logic with runDisabledReason (see App.tsx's `sensorModeGuardReason`) but never
  // includes the AOI validation error -- training doesn't touch the AOI/date fields
  // at all (no ingestion happens), so gating Train on an invalid AOI would block it
  // for a reason that has nothing to do with what it actually does.
  epochs: number
  onChangeEpochs: (epochs: number) => void
  lr: number
  onChangeLr: (lr: number) => void
  trainDisabledReason: string | null
  onTrain: () => void
  trainRunning: boolean
  trainError: string | null
  trainResult: LandCoverTrainResponse | null
}

export type ModelNodeType = Node<ModelNodeData, 'model'>

export function ModelNode({ data }: NodeProps<ModelNodeType>) {
  const {
    title,
    subtitle,
    derivedMode,
    preFusionWidth,
    sensorModesError,
    paneMode,
    onChangePaneMode,
    runDisabledReason,
    onRun,
    running,
    epochs,
    onChangeEpochs,
    lr,
    onChangeLr,
    trainDisabledReason,
    onTrain,
    trainRunning,
    trainError,
    trainResult,
  } = data

  return (
    <div style={nodeBoxStyle}>
      <Handle type="target" position={Position.Top} />

      <div style={nodeTitleStyle}>{title}</div>
      <div style={nodeSubtitleStyle}>{subtitle}</div>

      <div style={{ marginBottom: 4 }}>
        sensor_mode: <strong>{derivedMode ?? 'none'}</strong>
      </div>
      <div style={{ marginBottom: 8, opacity: 0.75 }}>
        pre-fusion width: {preFusionWidth !== null ? `${preFusionWidth} (512 × sensors)` : '—'}
      </div>

      {/* nodrag: same escape hatch as the buttons below -- without it, clicking
          the toggle starts a canvas drag-select instead of firing onClick. */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
        <button
          className="nodrag"
          onClick={() => onChangePaneMode('infer')}
          disabled={paneMode === 'infer'}
          style={{ flex: 1, fontWeight: paneMode === 'infer' ? 700 : 400 }}
        >
          Infer
        </button>
        <button
          className="nodrag"
          onClick={() => onChangePaneMode('train')}
          disabled={paneMode === 'train'}
          style={{ flex: 1, fontWeight: paneMode === 'train' ? 700 : 400 }}
        >
          Train
        </button>
      </div>

      {paneMode === 'infer' && (
        <>
          <button
            className="nodrag"
            onClick={onRun}
            disabled={running || runDisabledReason !== null}
            style={{ width: '100%' }}
          >
            {running ? 'Running…' : 'Run inference'}
          </button>
          {!running && runDisabledReason && <div style={nodeErrorStyle}>{runDisabledReason}</div>}
        </>
      )}

      {paneMode === 'train' && (
        <>
          <label style={fieldLabelStyle}>
            Epochs
            <input
              className="nodrag nowheel"
              type="number"
              min={1}
              max={10}
              step={1}
              value={epochs}
              onChange={(e) => onChangeEpochs(e.target.valueAsNumber)}
              style={fieldInputStyle}
            />
          </label>
          <label style={fieldLabelStyle}>
            Learning rate
            <input
              className="nodrag nowheel"
              type="number"
              step="any"
              value={lr}
              onChange={(e) => onChangeLr(e.target.valueAsNumber)}
              style={fieldInputStyle}
            />
          </label>
          <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8 }}>
            A few epochs demonstrates the training mechanism -- it doesn't produce a good
            model.
          </div>

          <button
            className="nodrag"
            onClick={onTrain}
            disabled={trainRunning || trainDisabledReason !== null}
            style={{ width: '100%' }}
          >
            {trainRunning ? 'Training…' : 'Train'}
          </button>
          {!trainRunning && trainDisabledReason && <div style={nodeErrorStyle}>{trainDisabledReason}</div>}
          {!trainRunning && trainError && <div style={nodeErrorStyle}>{trainError}</div>}

          {!trainRunning && trainResult && (
            <div style={{ marginTop: 8, fontSize: 12 }}>
              <div>checkpoint: {trainResult.checkpoint}</div>
              <div>train_loss: {trainResult.train_loss.toFixed(4)}</div>
              <div>val_loss: {trainResult.val_loss.toFixed(4)}</div>
              <div>mean_iou: {trainResult.mean_iou.toFixed(4)}</div>
              <div>elapsed: {trainResult.elapsed_seconds.toFixed(1)}s</div>
              <div style={{ marginTop: 4, opacity: 0.75 }}>
                Not available for inference yet — intentionally not added to the sensor-mode
                selector (see root CLAUDE.md).
              </div>
            </div>
          )}
        </>
      )}

      {sensorModesError && <div style={nodeErrorStyle}>sensor_modes failed: {sensorModesError}</div>}

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}
