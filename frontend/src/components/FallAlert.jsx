import { useAppContext } from '../App'

export default function FallAlert() {
  const { fallDetected, fallConfidence, dismissFall } = useAppContext()

  if (!fallDetected) return null

  return (
    <div
      role="alert"
      className="bg-amber-50 border-2 border-amber-400 rounded-2xl p-6 shadow-md flex flex-col gap-4"
    >
      <div className="flex items-start gap-4">
        <span className="text-4xl mt-0.5">⚠️</span>
        <div>
          <h2 className="text-xl font-bold text-amber-800 leading-tight">
            Possible Fall Detected
          </h2>
          <p className="text-amber-700 text-sm mt-1">
            Please check in on your loved one to make sure they're okay.
          </p>
          {fallConfidence !== null && (
            <p className="text-amber-600 text-xs mt-1">
              Confidence: {Math.round(fallConfidence * 100)}%
            </p>
          )}
        </div>
      </div>

      <button
        onClick={dismissFall}
        className="w-full bg-amber-400 hover:bg-amber-500 active:bg-amber-600 text-white font-semibold text-base py-3 rounded-xl transition-colors"
      >
        I've Checked In — Dismiss
      </button>
    </div>
  )
}
