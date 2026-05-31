import { useEffect, useRef, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts'
import { useAppContext } from '../App'

const MAX_SLOTS = 30
const DEMO_STALE_MS = 10_000

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-stone-700 border border-stone-600 rounded-lg px-3 py-2 text-xs shadow">
      <p className="text-stone-400 mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }} className="font-medium">
          {p.dataKey === 'hr' ? `HR: ${p.value} bpm` : `BR: ${p.value} /min`}
        </p>
      ))}
    </div>
  )
}

export default function VitalsGraph() {
  const { vitals } = useAppContext()
  const [demoVitals, setDemoVitals] = useState([])
  const lastRealTs = useRef(null)
  const tickRef = useRef(0)

  useEffect(() => {
    if (vitals.length > 0) lastRealTs.current = Date.now()
  }, [vitals])

  useEffect(() => {
    const id = setInterval(() => {
      const stale = !lastRealTs.current || Date.now() - lastRealTs.current > DEMO_STALE_MS
      if (!stale) {
        setDemoVitals([])
        return
      }
      tickRef.current += 1
      const t = tickRef.current
      const entry = {
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        hr: parseFloat((70 + 6 * Math.sin(t / 11) + (Math.random() - 0.5) * 2).toFixed(1)),
        br: parseFloat((14 + 2 * Math.sin(t / 7) + (Math.random() - 0.5) * 0.6).toFixed(1)),
      }
      setDemoVitals((prev) => {
        const next = [...prev, entry]
        return next.length > MAX_SLOTS ? next.slice(-MAX_SLOTS) : next
      })
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const source = vitals.length > 0 ? vitals : demoVitals
  const isDemo = vitals.length === 0 && demoVitals.length > 0

  const padded = [
    ...source,
    ...Array(Math.max(0, MAX_SLOTS - source.length)).fill({ time: '', hr: null, br: null }),
  ]

  return (
    <div className="rounded-xl bg-stone-800 border border-stone-700 px-4 py-3">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs text-stone-500 uppercase tracking-widest">Vitals Trend</p>
        {isDemo && (
          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-amber-900/60 text-amber-400 border border-amber-700/50">
            DEMO
          </span>
        )}
      </div>

      {source.length === 0 ? (
        <div className="flex items-center justify-center h-24 text-stone-600 text-xs">
          Waiting for data…
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={padded} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#292524" />
            <XAxis
              dataKey="time"
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 120]}
              tick={{ fontSize: 10, fill: '#78716c' }}
              tickLine={false}
              axisLine={false}
              tickCount={4}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="hr" stroke="#f87171" dot={false} strokeWidth={2} connectNulls={false} />
            <Line type="monotone" dataKey="br" stroke="#60a5fa" dot={false} strokeWidth={2} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      )}

      {source.length > 0 && (() => {
        const latest = source[source.length - 1]
        return (
          <div className="flex gap-4 mt-2 pt-2 border-t border-stone-700">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-red-400 inline-block" />
              <span className="text-xs text-stone-400">HR <span className="text-red-400 font-semibold">{Math.round(latest.hr)}</span></span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
              <span className="text-xs text-stone-400">BR <span className="text-blue-400 font-semibold">{latest.br?.toFixed(1)}</span></span>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
