import { createContext, useContext, useEffect, useState } from 'react'
import { useWebSocket } from './hooks/useWebSocket'
import FallAlert from './components/FallAlert'
import StatusCard from './components/StatusCard'
import EventLog from './components/EventLog'
import ConnectionStatus from './components/ConnectionStatus'

export const AppContext = createContext(null)

export function useAppContext() {
  return useContext(AppContext)
}

export default function App() {
  const { message, connected } = useWebSocket()
  const [fallDetected, setFallDetected] = useState(false)
  const [fallConfidence, setFallConfidence] = useState(null)
  const [occupied, setOccupied] = useState(false)
  const [events, setEvents] = useState([])

  useEffect(() => {
    if (!message) return

    setEvents((prev) => {
      if (prev.some((e) => e.timestamp === message.timestamp)) return prev
      return [message, ...prev]
    })

    if (message.event === 'fall_detected') {
      setFallDetected(true)
      setFallConfidence(message.confidence ?? null)
    } else if (message.event === 'presence_update') {
      setOccupied(message.occupied)
    }
  }, [message])

  function dismissFall() {
    setFallDetected(false)
    setFallConfidence(null)
  }

  return (
    <AppContext.Provider value={{ fallDetected, fallConfidence, occupied, events, connected, dismissFall }}>
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
          <h1 className="text-lg font-bold text-gray-900">GhostNet</h1>
          <ConnectionStatus />
        </header>

        <main className="max-w-2xl mx-auto px-4 py-6 space-y-6">
          <FallAlert />
          <StatusCard />
          <EventLog />
        </main>
      </div>
    </AppContext.Provider>
  )
}
