import { useAppContext } from '../App'

export default function ConnectionStatus() {
  const { connected } = useAppContext()

  return (
    <div className="flex items-center gap-2">
      <span className="relative flex h-2.5 w-2.5">
        {connected && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-60" />
        )}
        <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${connected ? 'bg-green-500' : 'bg-red-500'}`} />
      </span>
      <span className="text-sm text-gray-500">
        {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  )
}
