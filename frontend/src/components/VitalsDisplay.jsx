import { useState, useEffect } from 'react'
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

export default function VitalsDisplay() {
  const { vitals } = useAppContext()
  const latest = vitals[vitals.length - 1] ?? null

  return (
    <div className="rounded-xl bg-stone-800 border border-stone-700 px-4 py-3">
      <p className="text-xs text-stone-500 mb-3 uppercase tracking-widest">Live Vitals</p>
      <div className="flex gap-4">
        <VitalNumber label="Heart Rate" value={latest?.hr ?? null} unit="bpm" low={50} />
        <div className="w-px bg-stone-700" />
        <VitalNumber label="Breathing" value={latest?.br ?? null} unit="/min" low={8} />
      </div>
    </div>
  )
}
