import { useEffect } from 'react'
import { Handle, Position, useReactFlow } from '@xyflow/react'
import type { Node, NodeProps } from '@xyflow/react'
import type { LandCoverInferResponse } from '../types'
import { nodeBoxStyle, nodeErrorStyle, nodeSubtitleStyle, nodeTitleStyle } from './nodeStyles'

export interface OutputNodeData extends Record<string, unknown> {
  title: string
  subtitle: string
  running: boolean
  runError: string | null
  result: LandCoverInferResponse | null
  backendUrl: string
}

export type OutputNodeType = Node<OutputNodeData, 'output'>

export function OutputNode({ data }: NodeProps<OutputNodeType>) {
  const { title, subtitle, running, runError, result, backendUrl } = data
  const firstTile = result?.tiles[0] ?? null
  const { fitView } = useReactFlow()

  // <ReactFlow fitView> only auto-frames the canvas once, on initial mount --
  // before any run has happened. This node's rendered height changes twice after
  // a run finishes: once synchronously when the result text renders, and again
  // asynchronously once the quicklook <img> below finishes loading (it has no
  // intrinsic size until then). Without re-fitting, the node's own bottom border
  // can end up rendered below the visible viewport with no way to scroll to it --
  // ReactFlow positions nodes via CSS transform on an internal pane, not normal
  // document scroll, so nothing above (not even a fullscreen browser window) can
  // reveal it. Re-fit on both triggers so the growing node stays in view.
  //
  // The double-rAF delay matters: fitView() computes its bounding box from
  // ReactFlow's own ResizeObserver-tracked node measurements, not the live DOM --
  // and ResizeObserver notifications land in their own step after layout/paint,
  // not synchronously inside the event handler (img onLoad, or this effect) that
  // triggered the resize. Calling fitView() too early re-fits against the node's
  // stale (pre-growth) size. Two rAFs reliably land after that step in practice.
  useEffect(() => {
    if (!result && !runError) return
    let id2 = 0
    const id1 = requestAnimationFrame(() => {
      id2 = requestAnimationFrame(() => fitView({ duration: 300, padding: 0.15 }))
    })
    return () => {
      cancelAnimationFrame(id1)
      cancelAnimationFrame(id2)
    }
  }, [result, runError, fitView])

  return (
    <div style={nodeBoxStyle}>
      <Handle type="target" position={Position.Top} />

      <div style={nodeTitleStyle}>{title}</div>
      <div style={nodeSubtitleStyle}>{subtitle}</div>

      {running && <div>Running inference…</div>}

      {!running && runError && <div style={nodeErrorStyle}>{runError}</div>}

      {!running && !runError && !result && <div style={{ opacity: 0.6 }}>No run yet.</div>}

      {!running && result && (
        <div className="nowheel" style={{ maxHeight: 260, overflowY: 'auto' }}>
          <div>tile_count: {result.tile_count}</div>
          {firstTile ? (
            <>
              <div>uncertainty: {firstTile.uncertainty ?? 'null'}</div>
              <div>
                per_modality:{' '}
                {Object.entries(firstTile.per_modality)
                  .map(([k, v]) => `${k}=${v.toFixed(3)}`)
                  .join(', ') || '{}'}
              </div>
              <img
                src={backendUrl + firstTile.quicklook_url}
                alt={`quicklook for ${firstTile.tile_id}`}
                style={{ maxWidth: '100%', height: 'auto', display: 'block', marginTop: 6, borderRadius: 4 }}
                onLoad={() =>
                  requestAnimationFrame(() =>
                    requestAnimationFrame(() => fitView({ duration: 300, padding: 0.15 })),
                  )
                }
              />
            </>
          ) : (
            <div>No tiles in response.</div>
          )}
        </div>
      )}
    </div>
  )
}
