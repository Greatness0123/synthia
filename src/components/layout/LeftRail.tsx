/**
 * Left rail - vertical toolbar consolidating all floating trigger buttons.
 * Uses flex column with gap instead of hardcoded top-[Npx] positioning.
 */

import { useUIStore } from '../../store/uiStore';
import { useWorldStore } from '../../store/worldStore';
import { GearSix, TreeStructure, Export, Cube } from '../ui/icons';
import { Logo } from '../ui/Logo';
import { cn } from '../../utils/cn';
import { useMediaQuery } from '../../hooks/useMediaQuery';

export const LeftRail: React.FC = () => {
  const {
    setExportModalOpen,
    setObjectSpawnerOpen,
    rightPanelOpen,
    setRightPanelOpen,
  } = useUIStore();
  const { godModeOpen, setGodModeOpen } = useWorldStore();
  const isWide = useMediaQuery('(min-width: 768px)');

  return (
    <div className={cn(
      "fixed left-4 top-4 flex flex-col z-50",
      isWide ? "gap-2" : "gap-1"
    )}>
      {/* Logo */}
      <div className={cn(
        "glassmorphism rounded-full flex items-center justify-center",
        isWide ? "w-10 h-10" : "w-8 h-8"
      )}>
        <Logo size={isWide ? 28 : 20} />
      </div>

      <div className="w-full h-px bg-white/10 my-1" />

      {/* World Controls */}
      <button
        onClick={() => setGodModeOpen(!godModeOpen)}
        data-tour="world-controls-trigger"
        className={cn(
          "glassmorphism rounded-full flex items-center justify-center hover:bg-white/10 transition-all group",
          isWide ? "w-10 h-10" : "w-8 h-8",
          godModeOpen && "bg-white/10"
        )}
        aria-label="World Controls"
        title="World Controls"
      >
        <GearSix size={isWide ? 20 : 16} className={cn("text-text-secondary group-hover:text-text-primary transition-colors", godModeOpen && "text-text-primary")} />
      </button>

      {/* Agent Inspector */}
      <button
        onClick={() => setRightPanelOpen(!rightPanelOpen)}
        className={cn(
          "glassmorphism rounded-full flex items-center justify-center hover:bg-white/10 transition-all group",
          isWide ? "w-10 h-10" : "w-8 h-8",
          rightPanelOpen && "bg-white/10"
        )}
        aria-label="Agent Inspector"
        title="Agent Inspector"
      >
        <TreeStructure size={isWide ? 20 : 16} className={cn("text-text-secondary group-hover:text-text-primary transition-colors", rightPanelOpen && "text-text-primary")} />
      </button>

      <div className="w-full h-px bg-white/10 my-1" />

      {/* Export */}
      <button
        onClick={() => setExportModalOpen(true)}
        className={cn(
          "glassmorphism rounded-full flex items-center justify-center hover:bg-white/10 transition-all group",
          isWide ? "w-10 h-10" : "w-8 h-8"
        )}
        aria-label="Export Data"
        title="Export Data"
      >
        <Export size={isWide ? 20 : 16} className="text-text-secondary group-hover:text-text-primary transition-colors" />
      </button>

      {/* Object Spawner */}
      <button
        onClick={() => setObjectSpawnerOpen(true)}
        className={cn(
          "glassmorphism rounded-full flex items-center justify-center hover:bg-white/10 transition-all group",
          isWide ? "w-10 h-10" : "w-8 h-8"
        )}
        aria-label="Spawn Objects"
        title="Spawn Objects"
      >
        <Cube size={isWide ? 20 : 16} className="text-text-secondary group-hover:text-text-primary transition-colors" />
      </button>
    </div>
  );
};
