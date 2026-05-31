import { useEffect, useState } from 'react'
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts'

const BACKEND = 'http://localhost:8000'
const DAYS = 60

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-stone-700 border border-stone-600 rounded-lg px-3 py-2 text-xs shadow">
      <p className="text-stone-400 mb-1">{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color }} className="font-medium">
          {p.dataKey === 'avg_hr' && `HR: ${p.value} bpm`}
          {p.dataKey === 'avg_br' && `BR: ${p.value} /min`}
          {p.dataKey === 'falls' && `Falls: ${p.value}`}
        </p>
      ))}
    </div>
  )
}

export default function HistoryPanel() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${BACKEND}/history?days=${DAYS}`)
      .then((r) => r.json())
      .then((d) => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  // Merge vitals + falls onto shared day keys
  const chartData = (() => {
    if (!data) return []
    const map = {}

    // Fill all days in range so gaps are visible
    for (let i = DAYS - 1; i >= 0; i--) {
      const d = new Date()
      d.setDate(d.getDate() - i)
      const key = d.toISOString().slice(0, 10)
      const label = d.toLocaleDateString([], { month: 'short', day: 'numeric' })
      map[key] = { day: key, label, avg_hr: null, avg_br: null, falls: 0 }
    }

    for (const v of data.vitals) {
      if (map[v.day]) { map[v.day].avg_hr = v.avg_hr; map[v.day].avg_br = v.avg_br }
    }
    for (const f of data.falls_by_day) {
      if (map[f.day]) map[f.day].falls = f.falls
    }

    return Object.values(map)
  })()

  // Show only every 4th x-label to avoid crowding
  const tickFormatter = (val, idx) => idx % 4 === 0 ? val : ''

  const totalFalls = data?.total_falls ?? 0
  const hasVitals = chartData.some((d) => d.avg_hr !== null)

  return (
    <div className="rounded-xl bg-stone-800 border border-stone-700 px-4 py-3">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs text-stone-500 uppercase tracking-widest">60-Day History</p>
        <span className="text-xs text-stone-600">{DAYS}d</span>
      </div>

      {/* Fall count summary */}
      <div className="flex items-center gap-3 mb-3 px-1">
        <div className="flex flex-col">
          <span className="text-2xl font-bold text-white leading-none">{totalFalls}</span>
          <span className="text-xs text-stone-500 mt-0.5">fall{totalFalls !== 1 ? 's' : ''} detected</span>
        </div>
        {totalFalls > 0 && (
          <div className="flex gap-1 items-end pb-0.5">
            {chartData.map((d, i) =>
              d.falls > 0 ? (
                <div
                  key={i}
                  title={`${d.label}: ${d.falls} fall${d.falls !== 1 ? 's' : ''}`}
                  className="w-1.5 rounded-sm bg-red-500"
                  style={{ height: `${Math.min(d.falls * 8 + 8, 28)}px` }}
                />
              ) : (
                <div key={i} className="w-1.5 rounded-sm bg-stone-700" style={{ height: '4px' }} />
              )
            )}
          </div>
        )}
        {totalFalls === 0 && (
          <span className="text-xs text-emerald-500 font-medium">None — great!</span>
        )}
      </div>

      {/* Vitals chart */}
      {loading ? (
        <div className="flex items-center justify-center h-24 text-stone-600 text-xs">
          Loading history…
        </div>
      ) : !hasVitals ? (
        <div className="flex items-center justify-center h-24 text-stone-600 text-xs">
          No vitals data yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={110}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#292524" />
            <XAxis
              dataKey="label"
              tickFormatter={tickFormatter}
              tick={{ fontSize: 9, fill: '#78716c' }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              yAxisId="vitals"
              domain={[0, 140]}
              tick={{ fontSize: 9, fill: '#78716c' }}
              tickLine={false}
              axisLine={false}
              tickCount={4}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              yAxisId="vitals"
              type="monotone"
              dataKey="avg_hr"
              stroke="#f87171"
              dot={false}
              strokeWidth={1.5}
              connectNulls={false}
            />
            <Line
              yAxisId="vitals"
              type="monotone"
              dataKey="avg_br"
              stroke="#60a5fa"
              dot={false}
              strokeWidth={1.5}
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}

      {hasVitals && (
        <div className="flex gap-4 mt-2 pt-2 border-t border-stone-700">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-red-400 inline-block" />
            <span className="text-xs text-stone-400">Avg HR</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
            <span className="text-xs text-stone-400">Avg BR</span>
          </div>
        </div>
      )}
    </div>
  )
}
