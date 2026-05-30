import { createContext, useContext, useEffect, useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import FallAlert from './components/FallAlert'
import StatusCard from './components/StatusCard'
import EventLog from './components/EventLog'
import ConnectionStatus from './components/ConnectionStatus'
import VitalsGraph from './components/VitalsGraph'

export const AppContext = createContext(null)

export function useAppContext() {
  return useContext(AppContext)
}

const ALERT_EVENTS = new Set(['fall_detected', 'low_heart_rate', 'low_breathing_rate'])
const MAX_VITALS = 30

export default function App() {
  const { message, connected } = useWebSocket()
  const [fallDetected, setFallDetected] = useState(false)
  const [fallConfidence, setFallConfidence] = useState(null)
  const [occupied, setOccupied] = useState(false)
  const [events, setEvents] = useState([])
  const [vitals, setVitals] = useState([])

  useEffect(() => {
    if (!message) return

    if (message.event === 'vital_signs') {
      setVitals((prev) => {
        const entry = {
          time: new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          hr: message.heart_rate_bpm,
          br: message.breathing_rate_bpm,
        }
        const next = [...prev, entry]
        return next.length > MAX_VITALS ? next.slice(-MAX_VITALS) : next
      })
      return
    }

    if (message.event === 'presence_update') {
      setOccupied(message.occupied)
      return
    }

    if (ALERT_EVENTS.has(message.event)) {
      setEvents((prev) => {
        if (prev.some((e) => e.timestamp === message.timestamp)) return prev
        return [message, ...prev]
      })

      if (message.event === 'fall_detected') {
        setFallDetected(true)
        setFallConfidence(message.confidence ?? null)
      }
    }
  }, [message])

  function dismissFall() {
    setFallDetected(false)
    setFallConfidence(null)
  }

  return (
    <AppContext.Provider value={{ fallDetected, fallConfidence, occupied, events, vitals, connected, dismissFall }}>
      <div className="min-h-screen bg-stone-100">
        <header className="bg-stone-50 border-b border-stone-200 px-4 py-3 flex items-center justify-between">
          <h1 className="text-lg font-bold text-gray-900">GhostNet</h1>
          <ConnectionStatus />
        </header>

        <main className="max-w-2xl mx-auto px-4 py-6 space-y-6">
          <FallAlert />
          <StatusCard />
          <VitalsGraph />
          <EventLog />
        </main>
      </div>
    </AppContext.Provider>
  )
}
