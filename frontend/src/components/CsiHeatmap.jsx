import { useRef, useEffect } from 'react'
import { useAppContext } from '../App'

const N_SUB = 64
const WIDTH = 240   // history columns

// amplitude -> color (viridis-ish): low=dark blue, high=yellow
function color(v) {
  const t = Math.max(0, Math.min(1, v / 70))
  const r = Math.round(255 * Math.min(1, t * 1.6))
  const g = Math.round(255 * Math.min(1, t * 1.1))
  const b = Math.round(120 * (1 - t) + 60)
  return [r, g, b]
}

export default function CsiHeatmap() {
  const canvasRef = useRef(null)
  const { frame } = useAppContext()
  const lastTs = useRef(null)

  useEffect(() => {
    if (!frame?.csi || frame.timestamp === lastTs.current) return
    lastTs.current = frame.timestamp
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    // scroll left by 1px
    const img = ctx.getImageData(1, 0, WIDTH - 1, N_SUB)
    ctx.putImageData(img, 0, 0)
    // draw newest column on the right
    const col = ctx.createImageData(1, N_SUB)
    for (let k = 0; k < N_SUB; k++) {
      const [r, g, b] = color(frame.csi[k] ?? 0)
      const i = k * 4
      col.data[i] = r; col.data[i + 1] = g; col.data[i + 2] = b; col.data[i + 3] = 255
    }
    ctx.putImageData(col, WIDTH - 1, 0)
  }, [frame])

  return (
    <div className="rounded-xl bg-stone-800 border border-stone-700 px-4 py-3">
      <p className="text-xs text-stone-500 mb-2 uppercase tracking-widest">Live CSI Signal</p>
      <canvas
        ref={canvasRef}
        width={WIDTH}
        height={N_SUB}
        className="w-full h-28 rounded-lg image-render-pixel"
        style={{ imageRendering: 'pixelated' }}
      />
      <p className="text-xs text-stone-600 mt-1">Brighter = more signal disturbance from motion.</p>
    </div>
  )
}
