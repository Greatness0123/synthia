/**
 * Collapsible task input pill.
 * Collapsed: small pill below status bar, click to expand.
 * Expanded: full input bar. After submit, shows success indicator then auto-collapses.
 * Empty = free will. Occupied = task/training mode.
 */

import { useState, useEffect, useRef } from 'react';
import { useAgentStore } from '../../store/agentStore';
import { useAgentRuntimeStore } from '../../store/agentRuntimeStore';
import { useVideoTaskStore } from '../../store/videoTaskStore';
import { X, Check, VideoCamera, Target } from '../ui/icons';
import { cn } from '../../utils/cn';

export const TaskInput: React.FC = () => {
  const { directiveMode, setDirectiveMode, currentGoal, setCurrentGoal } = useAgentStore();
  const activeAgentId = useAgentStore((state) => state.activeAgentId);
  const config = useAgentRuntimeStore((s) => s.configs[activeAgentId]);
  const {
    activeTask: videoTask,
    setModalOpen: setVideoModalOpen,
    clearActiveTask: clearVideoTask,
    currentMilestoneIndex,
  } = useVideoTaskStore();

  const [localValue, setLocalValue] = useState(currentGoal || '');
  const [expanded, setExpanded] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  /* eslint-disable react-hooks/set-state-in-effect -- sync local state from store on agent/goal switch */
  useEffect(() => {
    setLocalValue(currentGoal || '');
    if (currentGoal) {
      setExpanded(true);
    } else {
      setExpanded(false);
    }
  }, [activeAgentId, currentGoal]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const hasProvider = !!config?.endpoint;
  const isActive = directiveMode === 'training' && currentGoal;
  const hasVideoTask = !!videoTask && isActive;

  const handleSubmit = () => {
    const text = localValue.trim();
    if (!text) return;
    setCurrentGoal(text);
    setDirectiveMode('training');
    setShowSuccess(true);
    setTimeout(() => {
      setShowSuccess(false);
      setExpanded(false);
    }, 1500);
  };

  const handleClear = () => {
    setLocalValue('');
    setCurrentGoal(null);
    setDirectiveMode('free_will');
    if (videoTask) {
      clearVideoTask();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
    if (e.key === 'Escape') {
      setExpanded(false);
    }
  };

  const containerRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (!isActive && containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setExpanded(false);
      }
    };
    if (expanded) {
      document.addEventListener('mousedown', handleOutsideClick);
      return () => document.removeEventListener('mousedown', handleOutsideClick);
    }
  }, [expanded, isActive]);

  // Collapsed pill
  if (!expanded && !showSuccess) {
    return (
      <div
        data-tour="task-input"
        className="fixed bottom-14 left-1/2 -translate-x-1/2 flex items-center gap-1.5 z-50"
      >
        <button
          onClick={() => { setExpanded(true); setTimeout(() => inputRef.current?.focus(), 50); }}
          className={cn(
            "rounded-full flex items-center gap-2 px-4 py-2 transition-all cursor-pointer",
            "bg-bg-panel border border-white/10 hover:border-white/20 shadow-lg backdrop-blur-md",
            isActive && "border-white/30"
          )}
        >
          {hasVideoTask ? (
            <div className="flex items-center gap-1.5 text-xs text-text-primary font-medium">
              <VideoCamera size={13} className="text-emerald-400" />
              <span>Video Task</span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/10 text-white/90 flex items-center gap-0.5">
                <Target size={9} />
                {currentMilestoneIndex + 1}/{videoTask?.keyframeIndices.length}
              </span>
            </div>
          ) : (
            <span className="text-xs text-text-tertiary">Give the agent a task...</span>
          )}
          {isActive && (
            <span className="w-1.5 h-1.5 rounded-full bg-text-primary animate-pulse" />
          )}
        </button>

        {/* Dedicated Quick Video Import Trigger Button */}
        <button
          type="button"
          onClick={() => setVideoModalOpen(true)}
          className={cn(
            "p-2 rounded-full border transition-all cursor-pointer shadow-lg backdrop-blur-md flex items-center justify-center",
            hasVideoTask
              ? "bg-emerald-500/20 border-emerald-500/40 text-emerald-300"
              : "bg-bg-panel border-white/10 text-text-tertiary hover:text-text-primary hover:border-white/25"
          )}
          title={hasVideoTask ? `Video task: ${videoTask?.name}` : "Import Video Demonstration"}
        >
          <VideoCamera size={15} />
        </button>
      </div>
    );
  }

  // Success indicator
  if (showSuccess) {
    return (
      <div className="fixed bottom-14 left-1/2 -translate-x-1/2 rounded-full flex items-center gap-2 px-4 py-2 bg-bg-panel border border-white/20 z-50">
        <Check size={14} className="text-text-primary" />
        <span className="text-xs text-text-primary font-medium">
          {hasVideoTask ? 'Video demonstration registered' : 'Task registered'}
        </span>
      </div>
    );
  }

  // Expanded input
  return (
    <div
      ref={containerRef}
      data-tour="task-input"
      className={cn(
        "fixed bottom-14 left-1/2 -translate-x-1/2 rounded-2xl flex items-center z-50",
        "w-[min(680px,calc(100vw-2rem))] h-12 px-4 gap-2.5",
        "bg-bg-panel border border-white/10 shadow-xl backdrop-blur-md",
        isActive && "border-white/30"
      )}
    >
      {/* Video Demonstration Action Button */}
      <button
        type="button"
        onMouseDown={(e) => {
          e.preventDefault();
          setVideoModalOpen(true);
        }}
        onClick={(e) => {
          e.stopPropagation();
          setVideoModalOpen(true);
        }}
        className={cn(
          "p-2 rounded-xl border transition-all flex items-center gap-1.5 cursor-pointer shrink-0",
          hasVideoTask
            ? "bg-white/15 border-white/30 text-text-primary"
            : "bg-white/5 border-white/10 text-text-tertiary hover:text-text-primary hover:bg-white/10 hover:border-white/20"
        )}
        title={hasVideoTask ? `Video demonstration: ${videoTask?.name}` : "Import Video Demonstration"}
      >
        <VideoCamera size={16} />
        {hasVideoTask && (
          <span className="text-[10px] font-mono px-1 rounded bg-white/15">
            {currentMilestoneIndex + 1}/{videoTask?.keyframeIndices.length}
          </span>
        )}
      </button>

      <input
        ref={inputRef}
        type="text"
        value={localValue}
        onChange={(e) => setLocalValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={hasProvider ? (hasVideoTask ? "Video demonstration goal active..." : "Give the agent a task or import video...") : "Set up an inference provider first"}
        disabled={!hasProvider}
        className={cn(
          "flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-tertiary",
          "outline-none ring-0 focus:outline-none focus:ring-0 focus:ring-transparent focus:border-transparent disabled:opacity-40 disabled:cursor-not-allowed"
        )}
        style={{ outline: 'none', WebkitTapHighlightColor: 'transparent' }}
        aria-label="Task input"
      />

      {localValue && (
        <button
          type="button"
          onMouseDown={(e) => e.preventDefault()}
          onClick={handleClear}
          className="p-1 text-text-tertiary hover:text-text-primary transition-colors cursor-pointer"
          aria-label="Clear task"
          title="Terminate task and return to free will"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
};
