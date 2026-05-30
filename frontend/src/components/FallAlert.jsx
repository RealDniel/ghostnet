import { useAppContext } from '../App'

export default function FallAlert() {
  const { fallDetected, fallConfidence, dismissFall } = useAppContext()

  if (!fallDetected) return null

  return (
    <div
      role="alert"
      className="bg-red-600 text-white rounded-2xl p-6 shadow-lg flex flex-col gap-4"
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-3xl">⚠️</span>
            <h2 className="text-2xl font-bold tracking-tight">Fall Detected</h2>
          </div>
          {fallConfidence !== null && (
            <p className="text-red-100 text-sm">
              Confidence: {Math.round(fallConfidence * 100)}%
            </p>
          )}
        </div>
      </div>

      <button
        onClick={dismissFall}
        className="w-full bg-white text-red-600 font-bold text-lg py-3 rounded-xl hover:bg-red-50 active:bg-red-100 transition-colors"
      >
        Dismiss Alert
      </button>
    </div>
  )
}
