/**
 * Scrolling stream of agent thoughts with clean process separation,
 * Markdown/JSON tag sanitization, and text copy support.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useAgentStore } from '../../store/agentStore';
import { motion, AnimatePresence } from 'framer-motion';
import { Syringe, ArrowDown, Copy, Check } from '../ui/icons';
import { STRINGS } from '../../constants/strings';
import { cleanThoughtText } from '../../utils/thoughtUtils';

function isNearBottom(el: HTMLElement, threshold = 80): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

export const ThoughtBank: React.FC = () => {
  const { thoughts } = useAgentStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = useCallback((text: string, id: string) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => {
      setCopiedId((curr) => (curr === id ? null : curr));
    }, 2000);
  }, []);

  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return;
    const near = isNearBottom(scrollRef.current);
    pinnedRef.current = near;
    setShowJump(!near);
  }, []);

  const scrollToBottom = useCallback(() => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    pinnedRef.current = true;
    setShowJump(false);
  }, []);

  useEffect(() => {
    if (!scrollRef.current || !pinnedRef.current) return;
    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [thoughts]);

  return (
    <div className="relative flex-1 flex flex-col min-h-0 select-text">
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-3.5 select-text"
      >
        <AnimatePresence initial={false}>
          {thoughts.map((thought, index) => {
            const cleaned = cleanThoughtText(thought.text);
            if (!cleaned) return null;

            return (
              <motion.div
                key={thought.id || `thought_${index}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.15 }}
                className="group p-3.5 rounded-xl bg-bg-elevated/40 hover:bg-bg-elevated/70 border border-border-subtle/80 hover:border-border transition-all shadow-sm"
              >
                <div className="flex items-center justify-between gap-2 mb-2 select-none">
                  <div className="flex items-center gap-2">
                    <span className="text-[11px] font-mono text-text-tertiary font-medium">
                      [{new Date(thought.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}]
                    </span>
                    {thought.isInjected && (
                      <span className="inline-flex items-center gap-1 text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                        <Syringe size={10} /> DIRECTIVE
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => handleCopy(cleaned, thought.id)}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-white/10 text-text-tertiary hover:text-text-primary transition-all cursor-pointer"
                    title="Copy thought"
                    aria-label="Copy thought"
                  >
                    {copiedId === thought.id ? <Check size={13} className="text-accent-primary" /> : <Copy size={13} />}
                  </button>
                </div>

                <p className="text-[13px] font-sans leading-relaxed text-text-primary whitespace-pre-wrap select-text cursor-text">
                  {cleaned}
                </p>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {thoughts.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center opacity-30 py-12 select-none">
            <p className="text-xs font-serif italic text-text-tertiary">{STRINGS.AGENT.VOID_SILENT}</p>
            <p className="text-xs text-text-tertiary mt-2">Configure a provider in Settings to begin cognition.</p>
          </div>
        )}
      </div>

      {/* Jump-to-latest button */}
      <button
        onClick={scrollToBottom}
        aria-label="Jump to latest thought"
        style={{
          opacity: showJump ? 1 : 0,
          pointerEvents: showJump ? 'auto' : 'none',
          transform: showJump ? 'translateY(0) scale(1)' : 'translateY(6px) scale(0.9)',
          transition: 'opacity 0.2s ease, transform 0.2s ease',
        }}
        className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-bg-elevated border border-border shadow-lg text-xs font-medium text-text-secondary hover:text-text-primary hover:border-white/20 hover:bg-bg-elevated/90 z-10 select-none cursor-pointer"
      >
        <ArrowDown size={12} />
        Latest
      </button>
    </div>
  );
};
