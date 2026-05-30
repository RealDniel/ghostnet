import { useEffect, useRef, useState } from 'react'

const WS_URL = 'ws://localhost:8000/ws'
const RECONNECT_DELAY = 2000

export function useWebSocket() {
  const [message, setMessage] = useState(null)
  const [connected, setConnected] = useState(false)
  const socketRef = useRef(null)
  const reconnectTimer = useRef(null)
  const unmounted = useRef(false)

  useEffect(() => {
    function connect() {
      if (unmounted.current) return

      const ws = new WebSocket(WS_URL)
      socketRef.current = ws

      ws.onopen = () => {
        if (!unmounted.current) setConnected(true)
      }

      ws.onmessage = (e) => {
        if (unmounted.current) return
        try {
          setMessage(JSON.parse(e.data))
        } catch {
          // ignore malformed messages
        }
      }

      ws.onclose = () => {
        if (unmounted.current) return
        setConnected(false)
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY)
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      unmounted.current = true
      clearTimeout(reconnectTimer.current)
      socketRef.current?.close()
    }
  }, [])

  return { message, connected }
}
