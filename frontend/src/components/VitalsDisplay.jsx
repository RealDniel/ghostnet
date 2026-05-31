import { useState, useEffect, useRef } from 'react'
import { useAppContext } from '../App'

function VitalNumber({ label, value, unit, low }) {
  const [pulse, setPulse] = useState(false)

  useEffect(() => {
    if (value == null) return
    setPulse(true)
    const t = setTimeout(() => setPulse(false), 300)
    return () => clearTimeout(t)
  }, [value])

  const alert = low && value != null && value < low
  const color = alert ? 'text-red-400' : 'text-cyan-400'

  return (
    <div className="flex-1 flex flex-col gap-1">
      <span className="text-xs text-stone-500 uppercase tracking-widest">{label}</span>
      <div className={`flex items-end gap-1 transition-opacity duration-150 ${pulse ? 'opacity-60' : 'opacity-100'}`}>
        <span className={`text-4xl font-bold tabular-nums truncate ${color}`}>
          {value != null ? Math.round(value) : '--'}
        </span>
        <span className="text-stone-500 text-sm mb-1">{unit}</span>
      </div>
    </div>
  )
}

const DEMO_STALE_MS = 10_000

export default function VitalsDisplay() {
  const { vitals } = useAppContext()
  const [demoHr, setDemoHr] = useState(null)
  const [demoBr, setDemoBr] = useState(null)
  const lastRealTs = useRef(null)
  const tickRef = useRef(0)

  // Track when real vitals last arrived
  useEffect(() => {
    if (vitals.length > 0) lastRealTs.current = Date.now()
  }, [vitals])

  // Kick off demo oscillation when real data is stale
  useEffect(() => {
    const id = setInterval(() => {
      const stale = !lastRealTs.current || Date.now() - lastRealTs.current > DEMO_STALE_MS
      if (!stale) {
        setDemoHr(null)
        setDemoBr(null)
        return
      }
      tickRef.current += 1
      const t = tickRef.current
      setDemoHr(parseFloat((70 + 6 * Math.sin(t / 11) + (Math.random() - 0.5) * 2).toFixed(1)))
      setDemoBr(parseFloat((14 + 2 * Math.sin(t / 7) + (Math.random() - 0.5) * 0.6).toFixed(1)))
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const latest = vitals[vitals.length - 1] ?? null
  const isDemo = !latest && demoHr != null
  const hr = latest?.hr ?? demoHr
  const br = latest?.br ?? demoBr

  return (
    <div className="rounded-xl bg-stone-800 border border-stone-700 px-4 py-3">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs text-stone-500 uppercase tracking-widest">Live Vitals</p>
        {isDemo && (
          <span className="text-xs font-semibold px-2 py-0.5 rounded bg-amber-900/60 text-amber-400 border border-amber-700/50">
            DEMO
          </span>
        )}
      </div>
      <div className="flex gap-4">
        <VitalNumber label="Heart Rate" value={hr} unit="bpm" low={50} />
        <div className="w-px bg-stone-700" />
        <VitalNumber label="Breathing" value={br} unit="/min" low={8} />
      </div>
    </div>
  )
}
