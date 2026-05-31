import { useEffect, useState } from 'react'
import { useAppContext } from '../App'

export default function FallAlert() {
  const { fallDetected, fallConfidence, fallGrace, callPlaced, dismissFall } = useAppContext()
  const [remaining, setRemaining] = useState(null)

  // Count down the grace period after a fall is detected.
  useEffect(() => {
    if (!fallDetected || fallGrace == null) {
      setRemaining(null)
      return
    }
    setRemaining(fallGrace)
    const start = Date.now()
    const id = setInterval(() => {
      const left = Math.max(0, fallGrace - (Date.now() - start) / 1000)
      setRemaining(left)
      if (left <= 0) clearInterval(id)
    }, 200)
    return () => clearInterval(id)
  }, [fallDetected, fallGrace])

  if (!fallDetected) return null

  const counting = remaining != null && remaining > 0 && !callPlaced
  const danger = callPlaced || (!counting)

  return (
    <div
      role="alert"
      className={`rounded-2xl p-6 shadow-md flex flex-col gap-4 border-2 ${
        danger ? 'bg-red-50 border-red-400' : 'bg-amber-50 border-amber-400'
      }`}
    >
      <div className="flex items-start gap-4">
        <span className={`text-2xl font-bold mt-0.5 ${danger ? 'text-red-500' : 'text-amber-500'}`}>!</span>
        <div>
          <h2 className={`text-xl font-bold leading-tight ${danger ? 'text-red-800' : 'text-amber-800'}`}>
            {callPlaced ? 'Caregiver Called' : 'Possible Fall Detected'}
          </h2>
          {callPlaced ? (
            <p className="text-red-700 text-sm mt-1">No movement after the fall — caregiver has been called.</p>
          ) : counting ? (
            <p className="text-amber-700 text-sm mt-1">
              Calling caregiver in <span className="font-bold tabular-nums">{Math.ceil(remaining)}s</span> — dismiss if you're okay.
            </p>
          ) : (
            <p className="text-red-700 text-sm mt-1">No response — calling caregiver…</p>
          )}
          {fallConfidence !== null && !callPlaced && (
            <p className="text-amber-600 text-xs mt-1">Confidence: {Math.round(fallConfidence * 100)}%</p>
          )}
        </div>
      </div>

      <button
        onClick={dismissFall}
        className={`w-full text-white font-semibold text-base py-3 rounded-xl transition-colors ${
          danger
            ? 'bg-red-400 hover:bg-red-500 active:bg-red-600'
            : 'bg-amber-400 hover:bg-amber-500 active:bg-amber-600'
        }`}
      >
        {callPlaced ? 'Dismiss' : "I'm Okay — Cancel"}
      </button>
    </div>
  )
}
