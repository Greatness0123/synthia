/**
 * Startup modal showing rehydration summary from the AI agent.
 */

import React, { useEffect, useState } from 'react';
import { useAgentStore } from '../../store/agentStore';
import { motion } from 'framer-motion';

export const RehydrationModal: React.FC = () => {
  const { rehydrationSummary, hasRehydrated } = useAgentStore();
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (hasRehydrated && !dismissed) {
      const timer = setTimeout(() => setDismissed(true), 2000);
      return () => clearTimeout(timer);
    }
  }, [hasRehydrated, dismissed]);

  const isVisible = !dismissed && !hasRehydrated && !!rehydrationSummary;

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-[200] flex items-center justify-center bg-black/70">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 1.05 }}
        className="max-w-lg w-full p-8"
      >
        <div className="flex flex-col items-center text-center space-y-6">
          <motion.div
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-12 h-12 rounded-full border-2 border-white/20 flex items-center justify-center"
          >
            <div className="w-2 h-2 bg-white/10 rounded-full" />
          </motion.div>

          <h2 className="text-xl font-serif text-text-primary tracking-tight">
            Agent is waking up...
          </h2>

          <div className="w-full min-h-[100px] p-4 bg-bg-panel border border-white/10 rounded-panel text-left">
            <p className="text-sm font-serif leading-relaxed text-text-secondary italic">
              {rehydrationSummary || "Consulting long-term memory archives..."}
              {!hasRehydrated && (
                <motion.span
                  animate={{ opacity: [0, 1, 0] }}
                  transition={{ duration: 0.8, repeat: Infinity }}
                  className="inline-block w-1 h-3 ml-1 bg-white/10"
                />
              )}
            </p>
          </div>

          {hasRehydrated && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-xs font-mono text-text-primary uppercase tracking-widest"
            >
              Rehydration Complete
            </motion.p>
          )}
        </div>
      </motion.div>
    </div>
  );
};
