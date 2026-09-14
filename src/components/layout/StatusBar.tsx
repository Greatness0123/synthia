import { useConnectionStore } from '../../store/connectionStore';
import { useAgentStore } from '../../store/agentStore';
import { useAgentRuntimeStore } from '../../store/agentRuntimeStore';
import { useMemoryStore } from '../../store/memoryStore';
import { cn } from '../../utils/cn';
import { useMediaQuery } from '../../hooks/useMediaQuery';

const Metric = ({ label, value, colorClass = "text-text-secondary" }: { label: string, value: string | number, colorClass?: string }) => (
  <div className="flex items-center gap-1.5 px-2.5 h-full border-r border-white/10 last:border-r-0">
    <span className="text-xs text-text-tertiary">{label}</span>
    <span className={cn("text-xs font-mono", colorClass)}>{value}</span>
  </div>
);

/**
 * Floating glassmorphic metrics pill at bottom-center of viewport.
 * Shows only client-side metrics — RTT/Inference are removed (coordinator-era).
 */
export const StatusBar: React.FC = () => {
  const { frameSize } = useConnectionStore();
  const { heartbeat, activeAgentId } = useAgentStore();
  const loopState = useAgentRuntimeStore((s) => s.loopStates[activeAgentId] || 'not_started');
  const cycleMs = useAgentRuntimeStore((s) => s.configs[activeAgentId]?.cycleMs ?? useConnectionStore.getState().cycleMs ?? 2000);
  const snapshot = useMemoryStore((s) => s.snapshot);
  const isWide = useMediaQuery('(min-width: 768px)');

  const loopColor = loopState === 'running' ? 'text-text-primary font-medium'
    : loopState === 'error' ? 'text-text-primary opacity-60'
    : loopState === 'paused' ? 'text-text-secondary'
    : 'text-text-tertiary';

  const dotColor = loopState === 'running' ? 'bg-text-primary animate-pulse'
    : loopState === 'error' ? 'bg-text-primary opacity-60'
    : loopState === 'paused' ? 'bg-text-secondary'
    : 'bg-text-tertiary';

  // Platform memory: usedJSHeapSize is the best cross-browser approximation
  // of total tab memory consumption (includes JS + WASM + DOM + framework overhead)
  const usedBytes = snapshot?.jsHeapUsedBytes;
  const limitBytes = snapshot?.jsHeapLimitBytes;
  const memMB = usedBytes != null ? usedBytes / (1024 * 1024) : null;
  const memPct = usedBytes != null && limitBytes ? (usedBytes / limitBytes) * 100 : null;

  const memColor = memPct == null ? 'text-text-secondary'
    : memPct >= 85 ? 'text-red-400'
    : memPct >= 65 ? 'text-yellow-400'
    : 'text-text-secondary';

  const memBarColor = memPct == null ? 'bg-text-tertiary'
    : memPct >= 85 ? 'bg-red-400'
    : memPct >= 65 ? 'bg-yellow-400'
    : 'bg-text-secondary';

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 glassmorphism rounded-full flex items-center h-9 px-1 z-50">
      {/* Agent loop status dot */}
      <div className="flex items-center gap-1.5 px-2.5 border-r border-white/10 h-full">
        <div className={cn("w-1.5 h-1.5 rounded-full", dotColor)} />
        <span className={cn("text-xs font-mono", loopColor)}>{loopState.replace('_', ' ')}</span>
      </div>

      <Metric
        label="Cycle"
        value={`${(cycleMs / 1000).toFixed(1)}s`}
      />
      {isWide && (
        <>
          <Metric
            label="Frame"
            value={frameSize != null && frameSize > 0 ? `${(frameSize / 1024).toFixed(1)}KB` : '--'}
          />

          {/* Platform memory: single unified indicator */}
          <div className="flex items-center gap-1.5 px-2.5 h-full border-r border-white/10 last:border-r-0">
            <span className="text-xs text-text-tertiary">Mem</span>
            {memMB != null ? (
              <div className="flex items-center gap-1.5">
                <span className={cn("text-xs font-mono", memColor)}>
                  {memMB < 1024 ? `${memMB.toFixed(0)}m` : `${(memMB / 1024).toFixed(1)}g`}
                </span>
                {memPct != null && (
                  <div className="w-12 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={cn("h-full rounded-full transition-all duration-500", memBarColor)}
                      style={{ width: `${Math.min(memPct, 100)}%` }}
                    />
                  </div>
                )}
              </div>
            ) : (
              <span className="text-xs font-mono text-text-secondary">--</span>
            )}
          </div>

          <Metric label="HB" value={heartbeat} />
        </>
      )}
    </div>
  );
};
