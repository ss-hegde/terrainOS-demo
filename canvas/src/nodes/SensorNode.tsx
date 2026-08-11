import { Handle, Position } from '@xyflow/react'
import type { Node, NodeProps } from '@xyflow/react'
import { nodeBoxStyle, nodeSubtitleStyle, nodeTitleStyle } from './nodeStyles'

// One component for both Sentinel-1 and Sentinel-2 blocks rather than two -- they're
// identical except for title/subtitle/id, so a second component would just be a copy.
// Presence/absence of this node's id in App's `nodes` state (not any field in here)
// is what drives the derived sensor_mode -- this component only renders + reports a
// delete click, App owns removing the node (and any edge referencing it).
export interface SensorNodeData extends Record<string, unknown> {
  title: string
  subtitle: string
  onDelete: () => void
}

export type SensorNodeType = Node<SensorNodeData, 'sensor'>

export function SensorNode({ data }: NodeProps<SensorNodeType>) {
  const { title, subtitle, onDelete } = data

  return (
    <div style={{ ...nodeBoxStyle, position: 'relative' }}>
      <Handle type="target" position={Position.Top} />

      {/* nodrag: same escape hatch as ModelNode's button -- without it, clicking
          this button starts a canvas drag-select instead of firing onClick. */}
      <button
        className="nodrag"
        onClick={onDelete}
        aria-label={`Remove ${title}`}
        title={`Remove ${title}`}
        style={{
          position: 'absolute',
          top: 6,
          right: 6,
          width: 18,
          height: 18,
          lineHeight: '16px',
          padding: 0,
          fontSize: 12,
          borderRadius: '50%',
          border: '1px solid currentColor',
          background: 'transparent',
          color: 'inherit',
          cursor: 'pointer',
        }}
      >
        ×
      </button>

      <div style={nodeTitleStyle}>{title}</div>
      <div style={{ ...nodeSubtitleStyle, marginBottom: 0 }}>{subtitle}</div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}
