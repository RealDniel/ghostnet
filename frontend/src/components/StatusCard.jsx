import { useAppContext } from '../App'

export default function StatusCard() {
  const { occupied } = useAppContext()

  return (
    <div className="bg-stone-50 rounded-2xl p-6 shadow-sm border border-stone-200 flex items-center gap-5">
      <span className="relative flex h-5 w-5 shrink-0">
        {occupied && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-60" />
        )}
        <span className={`relative inline-flex h-5 w-5 rounded-full shadow-sm ${occupied ? 'bg-green-500' : 'bg-orange-400'}`} />
      </span>
      <div>
        <p className="text-lg font-semibold text-gray-800">
          {occupied ? 'In the Room' : 'Not in the Room'}
        </p>
        <p className="text-sm text-gray-400">
          {occupied ? 'Your loved one is currently present.' : 'No one detected in the room right now.'}
        </p>
      </div>
    </div>
  )
}
