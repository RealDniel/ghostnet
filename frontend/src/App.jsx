import { createContext, useContext, useEffect, useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import FallAlert from './components/FallAlert'
import EventLog from './components/EventLog'
import ConnectionStatus from './components/ConnectionStatus'
import Scene3D from './components/Scene3D'
import CsiHeatmap from './components/CsiHeatmap'
import VitalsDisplay from './components/VitalsDisplay'
import VitalsGraph from './components/VitalsGraph'
import HistoryPanel from './components/HistoryPanel'

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
  const [fallGrace, setFallGrace] = useState(null)
  const [callPlaced, setCallPlaced] = useState(false)
  const [occupied, setOccupied] = useState(false)
  const [events, setEvents] = useState([])
  const [vitals, setVitals] = useState([])
  const [frame, setFrame] = useState(null)

  useEffect(() => {
    if (!connected) {
      setFrame(null)
      setOccupied(false)
    }
  }, [connected])

  useEffect(() => {
    if (!message) return

    if (message.event === 'frame') {
      setFrame(message)
      if (message.occupied !== undefined) setOccupied(message.occupied)
      return
    }

    if (message.event === 'session_end') return

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

    if (message.event === 'fall_cancelled') {
      setFallDetected(false)
      setFallConfidence(null)
      setFallGrace(null)
      setCallPlaced(false)
      return
    }

    if (message.event === 'call_placed') {
      setCallPlaced(true)
      setEvents((prev) => [message, ...prev])
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
        setFallGrace(message.grace_seconds ?? null)
        setCallPlaced(false)
      }
    }
  }, [message])

  function dismissFall() {
    setFallDetected(false)
    setFallConfidence(null)
    setFallGrace(null)
    setCallPlaced(false)
  }

  return (
    <AppContext.Provider value={{ fallDetected, fallConfidence, fallGrace, callPlaced, occupied, events, vitals, connected, dismissFall, frame }}>
      <div className="flex h-screen w-screen overflow-hidden bg-stone-950">

        {/* Left — 3D scene fills all remaining space */}
        <div className="flex-1 min-w-0">
          <Scene3D />
        </div>

        {/* Right panel */}
        <div className="w-72 flex flex-col bg-stone-900 border-l border-stone-800 overflow-hidden">

          {/* Logo + connection */}
          <div className="px-5 py-4 border-b border-stone-800 flex items-center justify-between shrink-0">
            {/* Drop your logo file at frontend/src/assets/logo.svg and swap this block */}
            <span className="text-white font-bold text-lg tracking-wide">GhostNet</span>
            <ConnectionStatus />
          </div>

          {/* Scrollable content */}
          <div className="flex-1 overflow-y-auto flex flex-col gap-4 px-4 py-4">
            <VitalsDisplay />
            <VitalsGraph />
            <HistoryPanel />
            <CsiHeatmap />
            <FallAlert />
            <EventLog />
          </div>

        </div>
      </div>
    </AppContext.Provider>
  )
}
