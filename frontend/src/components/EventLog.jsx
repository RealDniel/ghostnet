import { useEffect, useState } from 'react'
import { useAppContext } from '../App'

const EVENT_LABELS = {
  fall_detected: { label: 'Fall Detected', color: 'bg-red-100 text-red-700' },
  presence_update: { label: 'Presence Update', color: 'bg-blue-100 text-blue-700' },
}

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  } catch {
    return iso
  }
}

function EventRow({ event }) {
  const meta = EVENT_LABELS[event.event] ?? { label: event.event, color: 'bg-gray-100 text-gray-600' }

  return (
    <li className="flex items-start gap-3 py-3 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-400 mt-0.5 w-20 shrink-0">{formatTime(event.timestamp)}</span>
      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full shrink-0 ${meta.color}`}>
        {meta.label}
      </span>
      <span className="text-sm text-gray-600">
        {event.event === 'fall_detected' && event.confidence !== undefined &&
          `${Math.round(event.confidence * 100)}% confidence`}
        {event.event === 'presence_update' &&
          (event.occupied ? 'Room occupied' : 'Room unoccupied')}
      </span>
    </li>
  )
}

export default function EventLog() {
  const { events } = useAppContext()
  const [history, setHistory] = useState([])

  useEffect(() => {
    fetch('http://localhost:8000/events')
      .then((r) => r.json())
      .then((data) => setHistory(Array.isArray(data) ? data : []))
      .catch(() => {})
  }, [])

  const merged = [...events]
  for (const h of history) {
    if (!merged.some((e) => e.timestamp === h.timestamp)) {
      merged.push(h)
    }
  }
  merged.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100">
      <div className="px-6 py-4 border-b border-gray-100">
        <h2 className="text-sm font-medium text-gray-500">Event Log</h2>
      </div>
      <div className="px-6 max-h-80 overflow-y-auto">
        {merged.length === 0 ? (
          <p className="text-sm text-gray-400 py-6 text-center">No events yet</p>
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
