import { Handle, Position } from '@xyflow/react'
import type { Node, NodeProps } from '@xyflow/react'
import { nodeBoxStyle, nodeErrorStyle, nodeSubtitleStyle, nodeTitleStyle } from './nodeStyles'

// AOI/date/location-name state lives in App and flows down through `data`, same
// pattern as ModelNode/OutputNode -- this component stays a plain renderer, it
// doesn't hold its own copy of the values or do its own validation math (App
// computes `validationError` once and passes the verdict down).
export interface DataNodeData extends Record<string, unknown> {
  title: string
  subtitle: string
  lat: number
  lon: number
  locationName: string
  startDate: string
  endDate: string
  onChangeLat: (lat: number) => void
  onChangeLon: (lon: number) => void
  onChangeLocationName: (name: string) => void
  onChangeStartDate: (date: string) => void
  onChangeEndDate: (date: string) => void
  validationError: string | null
}

export type DataNodeType = Node<DataNodeData, 'data'>

const fieldLabelStyle = { display: 'block', marginBottom: 6 }
const fieldInputStyle = { display: 'block', width: '100%', marginTop: 2, boxSizing: 'border-box' as const }

export function DataNode({ data }: NodeProps<DataNodeType>) {
  const {
    title,
    subtitle,
    lat,
    lon,
    locationName,
    startDate,
    endDate,
    onChangeLat,
    onChangeLon,
    onChangeLocationName,
    onChangeStartDate,
    onChangeEndDate,
    validationError,
  } = data

  return (
    <div style={nodeBoxStyle}>
      <div style={nodeTitleStyle}>{title}</div>
      <div style={nodeSubtitleStyle}>{subtitle}</div>

      <label style={fieldLabelStyle}>
        Location name
        {/* nodrag/nowheel: ReactFlow's documented escape hatch so interactive
            elements inside a node handle their own pointer/wheel events instead of
            the canvas treating them as a drag-to-pan/select gesture. */}
        <input
          className="nodrag nowheel"
          type="text"
          value={locationName}
          onChange={(e) => onChangeLocationName(e.target.value)}
          style={fieldInputStyle}
        />
      </label>

      <label style={fieldLabelStyle}>
        Lat
        <input
          className="nodrag nowheel"
          type="number"
          value={lat}
          onChange={(e) => onChangeLat(e.target.valueAsNumber)}
          style={fieldInputStyle}
        />
      </label>

      <label style={fieldLabelStyle}>
        Lon
        <input
          className="nodrag nowheel"
          type="number"
          value={lon}
          onChange={(e) => onChangeLon(e.target.valueAsNumber)}
          style={fieldInputStyle}
        />
      </label>

      <label style={fieldLabelStyle}>
        Start date
        <input
          className="nodrag nowheel"
          type="date"
          value={startDate}
          onChange={(e) => onChangeStartDate(e.target.value)}
          style={fieldInputStyle}
        />
      </label>

      <label style={fieldLabelStyle}>
        End date
        <input
          className="nodrag nowheel"
          type="date"
          value={endDate}
          onChange={(e) => onChangeEndDate(e.target.value)}
          style={fieldInputStyle}
        />
      </label>

      {validationError && <div style={nodeErrorStyle}>{validationError}</div>}

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}
