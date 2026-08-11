import type { CSSProperties } from 'react'

// @xyflow/react's own stylesheet only applies the familiar bordered/padded box look
// to its built-in node types (.react-flow__node-default etc, see
// node_modules/@xyflow/react/dist/style.css) -- a custom node type registered via
// `nodeTypes` gets none of that for free, just bare positioning. Reuse the same
// --xy-node-* CSS custom properties xyflow itself uses so these custom nodes match
// the still-default-styled Data/Analytics/Internal Data/Vendor B boxes exactly,
// including staying correct in dark mode (see canvas/CLAUDE.md's colorMode gotcha --
// these vars are what colorMode="system" actually drives).
export const nodeBoxStyle: CSSProperties = {
  padding: 10,
  borderRadius: 'var(--xy-node-border-radius, var(--xy-node-border-radius-default))',
  width: 220,
  fontSize: 12,
  color: 'var(--xy-node-color, var(--xy-node-color-default))',
  textAlign: 'left',
  border: 'var(--xy-node-border, var(--xy-node-border-default))',
  backgroundColor: 'var(--xy-node-background-color, var(--xy-node-background-color-default))',
}

export const nodeTitleStyle: CSSProperties = { fontWeight: 600 }
export const nodeSubtitleStyle: CSSProperties = { fontSize: 12, opacity: 0.75, marginTop: 2, marginBottom: 8 }
export const nodeErrorStyle: CSSProperties = { color: '#e5484d', marginTop: 6, fontSize: 11 }
