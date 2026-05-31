import { useEffect, useState } from 'react'
import { useAppContext } from '../App'

const EVENT_META = {
  fall_detected:      { label: 'Possible Fall',       color: 'bg-amber-100 text-amber-800' },
  low_heart_rate:     { label: 'Low Heart Rate',       color: 'bg-red-100 text-red-700' },
  low_breathing_rate: { label: 'Low Breathing Rate',   color: 'bg-orange-100 text-orange-700' },
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return iso
  }
}

function detail(event) {
  if (event.event === 'fall_detected')
    return 'Fall detected'
  if (event.event === 'low_heart_rate')
    return `${event.heart_rate_bpm} bpm — below safe range`
  if (event.event === 'low_breathing_rate')
    return `${event.breathing_rate_bpm} bpm — below safe range`
  return ''
}

function EventRow({ event }) {
  const meta = EVENT_META[event.event] ?? { label: event.event, color: 'bg-gray-100 text-gray-600' }
  return (
    <li className="flex items-start gap-3 py-3 border-b border-stone-700 last:border-0">
      <span className="text-xs text-stone-600 mt-0.5 w-12 shrink-0">{formatTime(event.timestamp)}</span>
      <div className="flex flex-col gap-0.5">
        <span className={`text-xs font-bold px-2 py-0.5 rounded-full w-fit ${meta.color}`}>
          {meta.label}
        </span>
        <span className="text-xs text-stone-500">{detail(event)}</span>
      </div>
    </li>
  )
}

export default function EventLog() {
  const { events } = useAppContext()
  const [history, setHistory] = useState([])

  useEffect(() => {
    fetch('http://localhost:8000/events')
      .then((r) => r.json())
      .then((data) => {
        const alerts = Array.isArray(data) ? data.filter((e) => EVENT_META[e.event]) : []
        setHistory(alerts)
      })
      .catch(() => {})
  }, [])

  const merged = [...events]
  for (const h of history) {
    if (!merged.some((e) => e.timestamp === h.timestamp)) merged.push(h)
  }
  merged.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))

  return (
    <div className="rounded-xl bg-stone-800 border border-stone-700">
      <div className="px-4 py-3 border-b border-stone-700">
        <h2 className="text-xs text-stone-500 uppercase tracking-widest">Alert Log</h2>
      </div>
      <div className="px-4 max-h-64 overflow-y-auto">
        {merged.length === 0 ? (
          <p className="text-xs text-stone-600 py-6 text-center">No alerts yet.</p>
        ) : (
          <ul>
            {merged.map((e, i) => (
              <EventRow key={`${e.timestamp}-${i}`} event={e} />
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
