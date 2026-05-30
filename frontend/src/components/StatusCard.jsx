import { useAppContext } from '../App'

export default function StatusCard() {
  const { occupied } = useAppContext()

  return (
    <div className="bg-white rounded-2xl p-6 shadow-sm border border-gray-100">
      <p className="text-sm font-medium text-gray-500 mb-3">Room Status</p>
      <div className="flex items-center gap-3">
        <span className="text-4xl">{occupied ? '🧍' : '🪑'}</span>
        <div>
          <p className={`text-2xl font-bold ${occupied ? 'text-gray-900' : 'text-gray-400'}`}>
            {occupied ? 'Occupied' : 'Unoccupied'}
          </p>
          <p className="text-sm text-gray-400">
            {occupied ? 'Someone is in the room' : 'Room is empty'}
          </p>
        </div>
      </div>
    </div>
  )
}
