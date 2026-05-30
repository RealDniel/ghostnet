import { useAppContext } from '../App'

export default function ConnectionStatus() {
  const { connected } = useAppContext()

  return (
    <div className="flex items-center gap-2">
      <span
        className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`}
      />
      <span className="text-sm text-gray-500">
        {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  )
}
