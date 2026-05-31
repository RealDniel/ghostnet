import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer
} from 'recharts'
import { useAppContext } from '../App'

function formatTick(timeStr) {
  // Only show HH:MM, suppress seconds tick noise
  if (!timeStr) return ''
  const parts = timeStr.split(':')
  if (parts.length < 3) return timeStr
  return `${parts[0]}:${parts[1]}`
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-gray-200 rounded-xl px-4 py-3 shadow text-sm">
      <p className="text-gray-400 text-xs mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }} className="font-medium">
          {p.dataKey === 'hr' ? `Heart Rate: ${p.value} bpm` : `Breathing Rate: ${p.value} br/min`}
        </p>
      ))}
    </div>
  )
}

const MAX_SLOTS = 30

export default function VitalsGraph() {
  const { vitals } = useAppContext()

  // Pad with nulls on the right so data plots left-to-right and then scrolls
  const padded = [
    ...vitals,
    ...Array(Math.max(0, MAX_SLOTS - vitals.length)).fill({ time: '', hr: null, br: null }),
  ]

  // Only show a tick when the second is :00 (once per minute)
  const tickFormatter = (val) => {
    if (!val) return ''
    const parts = val.split(':')
    if (parts.length < 3) return ''
    return parts[2] === '00' ? `${parts[0]}:${parts[1]}` : ''
  }

  return (
    <div className="bg-stone-50 rounded-2xl shadow-sm border border-stone-200 p-6">
      <h2 className="text-base font-semibold text-gray-700 mb-1">Vitals Monitor</h2>
      <p className="text-xs text-gray-400 mb-5">Heart rate and breathing rate over time</p>

      {vitals.length > 0 && (() => {
        const latest = vitals[vitals.length - 1]
        return (
          <div className="flex gap-6 mb-5">
            <div className="flex flex-col">
              <span className="text-xs text-gray-400 uppercase tracking-wide">Heart Rate</span>
              <div className="flex items-baseline gap-1">
                <span style={{ fontFamily: 'Nunito, sans-serif' }} className="text-4xl font-extrabold text-red-500">{Math.round(latest.hr)}</span>
                <span className="text-sm text-gray-400">bpm</span>
              </div>
            </div>
            <div className="w-px bg-gray-100" />
            <div className="flex flex-col">
              <span className="text-xs text-gray-400 uppercase tracking-wide">Breathing Rate</span>
              <div className="flex items-baseline gap-1">
                <span style={{ fontFamily: 'Nunito, sans-serif' }} className="text-4xl font-extrabold text-blue-500">{latest.br.toFixed(1)}</span>
                <span className="text-sm text-gray-400">br/min</span>
              </div>
            </div>
          </div>
        )
      })()}

      {vitals.length === 0 ? (
        <div className="flex items-center justify-center h-64 text-gray-400 text-sm">
          Waiting for vitals data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={padded} margin={{ top: 8, right: 16, left: -8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
            <XAxis
              dataKey="time"
              tickFormatter={tickFormatter}
              tick={{ fontSize: 11, fill: '#9ca3af' }}
              tickLine={false}
              interval={0}
              minTickGap={60}
            />
            <YAxis
              tick={{ fontSize: 11, fill: '#9ca3af' }}
              domain={[0, 120]}
              tickCount={7}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              formatter={(v) => (
                <span className="text-sm text-gray-600">
                  {v === 'hr' ? 'Heart Rate (bpm)' : 'Breathing Rate (br/min)'}
                </span>
              )}
            />
            <ReferenceLine
              y={50}
              stroke="#ef4444"
              strokeDasharray="5 5"
              strokeOpacity={0.6}
              label={{ value: 'HR threshold', fontSize: 10, fill: '#ef4444', position: 'insideTopRight' }}
            />
            <ReferenceLine
              y={8}
              stroke="#f97316"
              strokeDasharray="5 5"
              strokeOpacity={0.6}
              label={{ value: 'BR threshold', fontSize: 10, fill: '#f97316', position: 'insideTopRight' }}
            />
            <Line type="monotone" dataKey="hr" stroke="#ef4444" dot={false} strokeWidth={4} connectNulls={false} name="hr" />
            <Line type="monotone" dataKey="br" stroke="#3b82f6" dot={false} strokeWidth={4} connectNulls={false} name="br" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
