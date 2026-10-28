import { AppHeader } from './components/AppHeader'
import { ChatPanel } from './components/ChatPanel'
import { CircuitCanvas } from './components/CircuitCanvas'
import { TelemetryMonitor } from './components/TelemetryMonitor'
import { useWebSocket } from './hooks/useWebSocket'

export default function App() {
  useWebSocket()

  return (
    <div className="flex h-full flex-col">
      <AppHeader />
      <main className="grid min-h-0 flex-1 gap-3 p-3 max-lg:auto-rows-[minmax(420px,auto)] max-lg:overflow-y-auto lg:grid-cols-[minmax(0,1fr)_minmax(380px,34%)]">
        <CircuitCanvas className="min-h-[420px] min-w-0" />
        <div className="grid min-h-0 min-w-0 gap-3 lg:grid-rows-[minmax(0,1fr)_minmax(240px,40%)]">
          <ChatPanel className="min-h-[320px] min-w-0" />
          <TelemetryMonitor className="min-h-[260px] min-w-0" />
        </div>
      </main>
    </div>
  )
}
