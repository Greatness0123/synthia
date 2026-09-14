import { useEffect, useRef } from 'react';
import { useWorldStore } from '../../store/worldStore';
import { STRINGS } from '../../constants/strings';
import { Eye } from '../ui/icons';
import { motion } from 'framer-motion';
import { useMediaQuery } from '../../hooks/useMediaQuery';
import { cn } from '../../utils/cn';

/**
 * Draggable picture-in-picture overlay showing what the AI sees (same 448x448 frame as inference).
 */
export const ModelInputPiP: React.FC = () => {
  const lastFrame = useWorldStore(state => state.lastAIFrameForDisplay);
  const imgRef = useRef<HTMLImageElement>(null);
  const isWide = useMediaQuery('(min-width: 768px)');

  useEffect(() => {
    if (lastFrame && imgRef.current) {
      imgRef.current.src = `data:image/webp;base64,${lastFrame}`;
    }
  }, [lastFrame]);

  const hasContent = !!lastFrame;

  return (
    <motion.div
      drag
      dragMomentum={false}
      dragElastic={0}
      className="absolute bottom-4 right-4 z-40 cursor-grab active:cursor-grabbing"
    >
      {isWide && (
        <div className="text-xs font-mono text-text-tertiary mb-1 uppercase tracking-tighter select-none">
          {STRINGS.AGENT.SENDING_TO_AI}
        </div>
      )}
      <div className={cn(
        "bg-bg-elevated border border-border rounded-[8px] relative overflow-hidden",
        isWide ? "w-[200px] h-[113px]" : "w-[80px] h-[80px]"
      )}>
        <img
          ref={imgRef}
          className="w-full h-full object-cover"
          style={{ display: hasContent ? 'block' : 'none' }}
          alt="AI perception view"
        />
        {!hasContent && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-bg-elevated">
            <Eye size={isWide ? 24 : 16} className="text-text-tertiary/20" />
          </div>
        )}
        <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent pointer-events-none" />
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className={cn("relative", isWide ? "w-5 h-5" : "w-3 h-3")}>
            <div className="absolute top-1/2 left-0 w-full h-px bg-white/40 -translate-y-1/2" />
            <div className="absolute left-1/2 top-0 h-full w-px bg-white/40 -translate-x-1/2" />
            <div className="absolute top-1/2 left-1/2 w-1 h-1 rounded-full bg-white/60 -translate-x-1/2 -translate-y-1/2" />
          </div>
        </div>
        {hasContent && (
          <div className="absolute top-1 right-1 flex items-center gap-1 pointer-events-none">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            {isWide && (
              <span className="text-xs font-mono text-green-400 uppercase">live</span>
            )}
          </div>
        )}
      </div>
    </motion.div>
  );
};

