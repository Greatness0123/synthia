import { AppShell } from './components/layout/AppShell';
import { WorldViewport } from './components/world/WorldViewport';
import { RehydrationModal } from './components/ui/RehydrationModal';
import { StatusBar } from './components/layout/StatusBar';
import { AgentStatus } from './components/agent/AgentStatus';
import { ThoughtBank } from './components/agent/ThoughtBank';
import { InjectionInput } from './components/agent/InjectionInput';
import { MemoryViewer } from './components/agent/MemoryViewer';
import { StructureViewer } from './components/agent/StructureViewer';
import { GodModePanel } from './components/godmode/GodModePanel';
import { ObjectSpawner } from './components/godmode/ObjectSpawner';
import { useUIStore } from './store/uiStore';
import { useWorldStore } from './store/worldStore';
import { useAgentStore } from './store/agentStore';
import { Dropdown } from './components/ui/Dropdown';
import {
  Brain,
  Database,
  Cube,
  ListChecks,
  Camera,
  VideoCamera,
  Monitor,
  X,
  Sun,
  Moon,
  SpeakerHigh,
  Gear,
} from './components/ui/icons';
import { ExportModal } from './components/export/ExportModal';
import { AgentSettingsModal } from './components/agent/AgentSettingsModal';
import { MotorCodexModal } from './components/agent/MotorCodexModal';
import { LogViewer } from './components/agent/LogViewer';
import { motion, AnimatePresence, useDragControls } from 'framer-motion';
import { useEffect } from 'react';
import * as Tone from 'tone';
import { cn } from './utils/cn';
import { TaskInput } from './components/layout/TaskInput';
import { VideoTaskModal } from './components/videoTask/VideoTaskModal';
import { VideoTaskPip } from './components/videoTask/VideoTaskPip';
import { LeftRail } from './components/layout/LeftRail';
import { OnboardingProvider } from './components/onboarding/OnboardingProvider';
import { CrashOverlay } from './components/ui/CrashOverlay';
import type { CameraMode } from './types/world';
import { useSpeechStore } from './store/speechStore';
import { initSpeech } from './utils/speech';
import { synthiaToast } from './utils/synthiaToast';

function App() {
  const {
    activeRightPanelTab,
    setActiveRightPanelTab,
    rightPanelOpen,
    setRightPanelOpen,
    theme,
    toggleTheme,
    settingsModalOpen,
    setSettingsModalOpen,
    spawning,
    setSpawning,
  } = useUIStore();
  const { cameraMode, setCameraMode } = useWorldStore();
  const { globalTtsEnabled, setGlobalTtsEnabled } = useSpeechStore();
  const activeAgentId = useAgentStore((state) => state.activeAgentId);
  const inspectorDragControls = useDragControls();

  // Initialize browser-native Web Speech synthesis voices
  useEffect(() => {
    initSpeech();
  }, []);

  useEffect(() => {
    const resumeAudio = async () => {
      await Tone.start();
      if ((window as any)._synthia_audio_engine) {
        await (window as any)._synthia_audio_engine.initialize();
      }
      document.removeEventListener('click', resumeAudio);
    };
    document.addEventListener('click', resumeAudio);
    return () => document.removeEventListener('click', resumeAudio);
  }, []);

  // Apply theme class to root element
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'light') {
      root.classList.add('light');
    } else {
      root.classList.remove('light');
    }
  }, [theme]);

  const agentItems = Object.keys(useAgentStore.getState().agents).map((id) => ({
    value: id,
    label: id,
  }));

  return (
    <AppShell>
      {/* 3D Viewport - Full Screen Canvas */}
      <WorldViewport />

      {/* === Floating UI Layer === */}

      {/* Left Rail - consolidated trigger buttons */}
      <LeftRail />

      {/* Dev Multi-Agent Controller - Top Center */}
      <div className="fixed top-4 left-1/2 -translate-x-1/2 glassmorphism rounded-full flex items-center gap-3 p-2 z-50">
        <button
          onClick={async () => {
            if (!(window as any).synthia?.spawnAgent || spawning) return;
            setSpawning(true);
            try {
              const newBinder = await (window as any).synthia.spawnAgent();
              if (newBinder) {
                synthiaToast.success('Agent spawned successfully.');
              } else {
                synthiaToast.error('Spawning failed - check the log for details.');
              }
            } catch {
              synthiaToast.error('Spawning failed unexpectedly.');
            } finally {
              setSpawning(false);
            }
          }}
          disabled={spawning}
          data-tour="spawn-agent"
          className="px-4 py-1.5 bg-white/10 hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed text-text-primary text-xs font-medium rounded-full transition-all flex items-center gap-2"
        >
          {spawning ? 'Spawning…' : '+ Spawn Agent'}
        </button>

        {/* Agent Selector Custom Dropdown */}
        <Dropdown
          value={activeAgentId}
          onChange={(nextAgentId) => useAgentStore.getState().setActiveAgentId(nextAgentId)}
          items={agentItems}
          className="w-28"
          triggerClassName="h-7 text-xs font-medium bg-transparent border-0 text-text-secondary hover:text-text-primary px-2"
        />

        {/* Agent Settings Button (Gear) */}
        <button
          onClick={() => setSettingsModalOpen(!settingsModalOpen)}
          className={cn(
            "w-7 h-7 rounded-full flex items-center justify-center hover:bg-white/10 transition-all text-text-secondary group",
            settingsModalOpen && "bg-white/10 text-text-primary"
          )}
          aria-label="Agent Settings"
          title="Agent Settings"
        >
          <Gear size={15} className="group-hover:rotate-45 transition-transform" />
        </button>

        {/* Global TTS Speaker Toggle Button (visible only when speech is enabled) */}
        {globalTtsEnabled && (
          <button
            onClick={() => setGlobalTtsEnabled(false)}
            className="w-7 h-7 rounded-full flex items-center justify-center hover:bg-white/10 transition-all text-text-secondary group"
            aria-label="Mute Global TTS Voice"
            title="Mute Global TTS Voice"
          >
            <SpeakerHigh size={15} className="group-hover:text-text-primary transition-colors" />
          </button>
        )}

        {/* Theme Toggle Button */}
        <button
          onClick={toggleTheme}
          className="w-7 h-7 rounded-full flex items-center justify-center hover:bg-white/10 transition-all text-text-secondary group"
          aria-label={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
          title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
        >
          {theme === 'dark' ? (
            <Sun size={15} className="group-hover:text-text-primary transition-colors" />
          ) : (
            <Moon size={15} className="group-hover:text-text-primary transition-colors" />
          )}
        </button>
      </div>

      {/* Camera Controls Pill - Top Right */}
      <div data-tour="camera-modes" className="fixed top-4 right-4 glassmorphism rounded-full flex items-center p-1 z-50">
        {[
          { mode: 'third_person', icon: Camera, label: '3RD' },
          { mode: 'first_person', icon: VideoCamera, label: '1ST' },
          { mode: 'model_input', icon: Monitor, label: '2ND' },
        ].map(({ mode, icon: Icon, label }) => (
          <button
            key={mode}
            onClick={() => setCameraMode(mode as CameraMode)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 rounded-full transition-all",
              cameraMode === mode
                ? "bg-white/10 text-text-primary"
                : "text-text-tertiary hover:text-text-secondary hover:bg-white/5"
            )}
            aria-label={`Camera Mode ${label}`}
          >
            <Icon size={14} />
            <span className="text-xs font-medium">{label}</span>
          </button>
        ))}
      </div>

      {/* Task Input Pill - Bottom Center */}
      <TaskInput />

      {/* Metrics Pill - Bottom Center */}
      <StatusBar />

      {/* GodMode Panel */}
      <GodModePanel />

      {/* Right Panel - Agent Inspector */}
      <AnimatePresence>
        {rightPanelOpen && (
          <motion.div
            initial={{ opacity: 0, x: 8 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 8 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            drag
            dragListener={false}
            dragControls={inspectorDragControls}
            dragMomentum={false}
            dragElastic={0}
            dragConstraints={{ top: -200, left: -600, right: 200, bottom: 400 }}
            data-tour="agent-inspector-panel"
            className="fixed top-[10vh] right-[5vw] w-[380px] max-w-[calc(100vw-5rem)] h-[80vh] max-h-[calc(100vh-2rem)] bg-bg-panel border border-white/10 rounded-modal z-[60] flex flex-col overflow-hidden select-text"
          >
            {/* Header - drag handle */}
            <div 
              onPointerDown={(e) => inspectorDragControls.start(e)}
              className="flex items-center justify-between px-4 py-2 border-b border-white/10 shrink-0 cursor-grab active:cursor-grabbing select-none"
            >
              <AgentStatus />
              <button
                onClick={() => setRightPanelOpen(false)}
                className="w-6 h-6 rounded-full flex items-center justify-center hover:bg-white/10 transition-colors cursor-pointer"
                aria-label="Close Agent Inspector"
              >
                <X size={16} className="text-text-tertiary" />
              </button>
            </div>

            {/* Tab Bar */}
            <div className="flex border-b border-white/10 p-1 gap-1 shrink-0">
              {[
                { id: 'thoughts', icon: Brain, label: 'Thoughts' },
                { id: 'memories', icon: Database, label: 'Memories' },
                { id: 'structure', icon: Cube, label: 'Body' },
                { id: 'logs', icon: ListChecks, label: 'Logs' },
              ].map(({ id, icon: Icon, label }) => (
                <button
                  key={id}
                  onClick={() => setActiveRightPanelTab(id as any)}
                  className={`flex-1 flex items-center justify-center gap-2 py-1.5 rounded transition-all ${
                    activeRightPanelTab === id
                      ? 'bg-white/10 text-text-primary font-bold'
                      : 'text-text-tertiary hover:text-text-secondary hover:bg-white/5'
                  }`}
                  aria-label={`View ${label}`}
                >
                  <Icon size={14} />
                  <span className="text-xs font-medium">{label}</span>
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="flex-1 overflow-hidden flex flex-col">
              {activeRightPanelTab === 'thoughts' && (
                <>
                  <ThoughtBank />
                  <InjectionInput />
                </>
              )}
              {activeRightPanelTab === 'structure' && <StructureViewer />}
              {activeRightPanelTab === 'memories' && <MemoryViewer />}
              {activeRightPanelTab === 'logs' && <LogViewer />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Standalone Object Spawner */}
      <ObjectSpawner />

      {/* Existing Modals */}
      <ExportModal />
      <AgentSettingsModal />
      <MotorCodexModal />
      <RehydrationModal />
      <VideoTaskModal />

      {/* Video Demonstration Floating PiP HUD */}
      <VideoTaskPip />

      {/* Onboarding */}
      <OnboardingProvider />

      {/* WASM Crash Overlay */}
      <CrashOverlay />
    </AppShell>
  );
}

export default App;
