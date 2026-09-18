import { useAgentStore } from '../../store/agentStore';
import { useAgentRuntimeStore } from '../../store/agentRuntimeStore';
import { Badge } from '../ui/Badge';
import { Brain } from '../ui/icons';
import { cn } from '../../utils/cn';

export const AgentStatus: React.FC = () => {
  const activeAgentId = useAgentStore((state) => state.activeAgentId);
  const loopState = useAgentRuntimeStore((state) => state.loopStates[activeAgentId] || 'not_started');

  const statusLabel = loopState === 'running' ? 'thinking'
    : loopState === 'paused' ? 'paused'
    : loopState === 'stopped' ? 'stopped'
    : loopState === 'error' ? 'error'
    : 'idle';

  return (
    <div className="flex items-center gap-3 shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <Brain size={16} className="text-text-secondary shrink-0" />
        <h2 className="text-xs font-semibold text-text-primary truncate">{activeAgentId}</h2>
      </div>
      <Badge
        variant={loopState === 'running' ? 'accent' : 'default'}
        className={cn(
          'text-[10px] px-1.5 py-0 shrink-0',
          loopState === 'running' && 'animate-pulse'
        )}
      >
        {statusLabel}
      </Badge>
    </div>
  );
};
