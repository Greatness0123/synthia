/**
 * Modal for exporting simulation data with side-panel preview.
 * Cleaned up in Phase 7 to run client-side with optional local fallback, LeRobot support, and Task Filter.
 */

import React, { useState, useEffect, useMemo } from 'react';
import { useUIStore } from '../../store/uiStore';
import { useAgentStore } from '../../store/agentStore';
import { useConnectionStore } from '../../store/connectionStore';
import {
  Export,
  X,
  Database,
  Notebook,
  Archive,
  Robot,
  FileCode,
  Document as FileCsv,
  User,
  People as Users,
  ArrowsClockwise,
  Info,
  Eye,
  DownloadSimple,
} from '../ui/icons';
import { STRINGS } from '../../constants/strings';
import type { ExportFormat, ExportConfig, ExportType } from '../../types/export';
import { synthiaToast } from '../../utils/synthiaToast';
import { runClientSideExport } from '../../utils/clientDatasetExporter';
import { getSupabaseClient } from '../../utils/supabaseClient';
import { Panel } from '../ui/Panel';
import { motion, AnimatePresence } from 'framer-motion';

interface SessionRow {
  id: string;
  started_at: string;
  ended_at: string | null;
  total_heartbeats: number;
  body_type: string | null;
  memory_count: number;
  estimated_size_bytes: number;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export const ExportModal: React.FC = () => {
  const { exportModalOpen, setExportModalOpen, exportProgress, setExportProgress } = useUIStore();
  const { supabaseUrl, supabaseKey } = useConnectionStore();
  const { memories, heartbeat, activeAgentId } = useAgentStore();

  const [availableSessions, setAvailableSessions] = useState<SessionRow[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [sessionsError, setSessionsError] = useState<string | null>(null);

  // Scoping states
  const [exportScope, setExportScope] = useState<'single' | 'all'>('single');
  const [zipPerAgent, setZipPerAgent] = useState(false);

  const [exportType, setExportType] = useState<ExportType>('dataset');
  const [format, setFormat] = useState<ExportFormat>('JSONL');
  const [scope, setScope] = useState<ExportConfig['scope']>('all');

  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [hbFrom, setHbFrom] = useState(0);
  const [hbTo, setHbTo] = useState(heartbeat);
  const [selectedSessions, setSelectedSessions] = useState<string[]>([]);
  const [selectedTasks, setSelectedTasks] = useState<string[]>([]);

  const [includeTiers, setIncludeTiers] = useState<number[]>([1, 2, 3]);
  const [includeThoughts, setIncludeThoughts] = useState(true);
  const [includeSkills, setIncludeSkills] = useState(true);
  const [includeMotorPrograms, setIncludeMotorPrograms] = useState(true);
  const [excludeInjected, setExcludeInjected] = useState(false);
  const [successfulOnly, setSuccessfulOnly] = useState(false);
  const [tierInfoOpen, setTierInfoOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // Keep hbTo updated as simulation goes forward
  const effectiveHbTo = isExporting ? hbTo : heartbeat;

  // Fetch sessions directly from Supabase if configured
  useEffect(() => {
    if (!exportModalOpen || !supabaseUrl || !supabaseKey) return;

    const supabase = getSupabaseClient(supabaseUrl, supabaseKey);
    if (!supabase) {
      (async () => { setSessionsLoading(false); setSessionsError('Supabase not configured'); })();
      return;
    }
    const targetAgentIds = exportScope === 'single'
      ? [activeAgentId]
      : Object.keys(useAgentStore.getState().agents);

    (async () => {
      setSessionsLoading(true);
      setSessionsError(null);
      try {
        const { data, error } = await supabase.from('sessions')
          .select('*')
          .in('agent_id', targetAgentIds)
          .order('started_at', { ascending: false });

        if (error) {
          console.error('[ExportModal] Fetch sessions error:', error.message);
          setSessionsError(error.message || 'Failed to fetch sessions');
        } else if (data) {
          setAvailableSessions(data);
        }
      } catch (err) {
        console.error('[ExportModal] Fetch sessions exception:', err);
        setSessionsError('Network error fetching sessions');
      } finally {
        setSessionsLoading(false);
      }
    })();
  }, [exportModalOpen, supabaseUrl, supabaseKey, activeAgentId, exportScope]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExportModalOpen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [setExportModalOpen]);

  const handleExportScopeChange = (newScope: 'single' | 'all') => {
    setExportScope(newScope);
    if (newScope === 'all' && scope === 'session') {
      setScope('all');
    }
  };

  // Dynamically resolve memories across targeted agents for real-time preview
  const targetedMemories = useMemo(() => {
    if (exportScope === 'single') {
      return memories;
    }
    const allAgents = Object.values(useAgentStore.getState().agents);
    return allAgents.flatMap(a => a.memories || []);
  }, [memories, exportScope]);

  // Extract unique tasks across targeted memories
  const availableTasks = useMemo(() => {
    const tasks = new Set<string>();
    targetedMemories.forEach(m => {
      if (m.goalAtTime) tasks.add(m.goalAtTime);
    });
    return Array.from(tasks);
  }, [targetedMemories]);

  const filteredCount = useMemo(() => {
    let filtered = targetedMemories.filter(memory => includeTiers.includes(memory.tier));
    if (excludeInjected) filtered = filtered.filter(memory => !memory.isInjected);
    if (successfulOnly) filtered = filtered.filter(memory => (memory.rewardSignal || 0) > 0.5);
    if (selectedTasks.length > 0) {
      filtered = filtered.filter(memory => memory.goalAtTime && selectedTasks.includes(memory.goalAtTime));
    }
    if (scope === 'heartbeat_range') {
      filtered = filtered.filter(memory => memory.heartbeat >= hbFrom && memory.heartbeat <= effectiveHbTo);
    } else if (scope === 'session' && selectedSessions.length > 0) {
      filtered = filtered.filter((memory) => selectedSessions.includes(memory.sessionId || ''));
    }
    return filtered.length;
  }, [targetedMemories, includeTiers, excludeInjected, successfulOnly, selectedTasks, scope, hbFrom, effectiveHbTo, selectedSessions]);

  const estimatedSize = useMemo(() => {
    const bytesPerRow = 2 * 1024;
    const totalBytes = filteredCount * bytesPerRow;
    if (totalBytes > 1024 * 1024) return `~${(totalBytes / (1024 * 1024)).toFixed(1)}MB`;
    return `~${(totalBytes / 1024).toFixed(1)}KB`;
  }, [filteredCount]);

  const selectedSessionData = useMemo(() => {
    return availableSessions.filter(s => selectedSessions.includes(s.id));
  }, [availableSessions, selectedSessions]);

  const totalMemoryCount = useMemo(() => {
    if (scope === 'session' && selectedSessionData.length > 0) {
      return selectedSessionData.reduce((sum, s) => sum + (s.memory_count || 0), 0);
    }
    return filteredCount;
  }, [scope, selectedSessionData, filteredCount]);

  const handleExport = async () => {
    setIsExporting(true);
    setExportProgress(0);
    synthiaToast.info(STRINGS.TOASTS.EXPORT_STARTED);

    const config: ExportConfig = {
      exportType,
      format: exportType === 'dataset' ? format : undefined,
      agentIds: exportScope === 'single' ? [activeAgentId] : Object.keys(useAgentStore.getState().agents),
      zipPerAgent: exportScope === 'all' ? zipPerAgent : undefined,
      scope,
      includeTiers: includeTiers as any,
      includeFrames: false,
      includeThoughts: exportType === 'session_full' ? includeThoughts : undefined,
      includeSkills: exportType === 'session_full' ? includeSkills : undefined,
      includeMotorPrograms: exportType === 'session_full' ? includeMotorPrograms : undefined,
      excludeInjected,
      successfulOnly,
      taskFilter: selectedTasks.length > 0 ? selectedTasks : undefined,
      dateFrom: scope === 'date_range' ? dateFrom : undefined,
      dateTo: scope === 'date_range' ? dateTo : undefined,
      heartbeatFrom: scope === 'heartbeat_range' ? hbFrom : undefined,
      heartbeatTo: scope === 'heartbeat_range' ? effectiveHbTo : undefined,
      sessionIds: scope === 'session' ? selectedSessions : undefined,
    };

    try {
      await runClientSideExport(config, supabaseUrl || '', supabaseKey || '', setExportProgress);
      synthiaToast.success('Dataset export downloaded successfully!');
      setTimeout(() => setExportModalOpen(false), 800);
    } catch (err: any) {
      console.error('[ClientExport] Error during export:', err);
      synthiaToast.error(err.message || 'Export failed.');
    } finally {
      setIsExporting(false);
    }
  };

  if (!exportModalOpen) return null;

  const exportTypes: { id: ExportType; name: string; icon: React.ComponentType<{ size?: string | number }>; desc: string }[] = [
    { id: 'dataset', name: 'Dataset', icon: Database, desc: 'ML training datasets (HDF5, LeRobot, JSONL, CSV, Parquet)' },
    { id: 'thoughts_report', name: 'Thoughts Report', icon: Notebook, desc: STRINGS.EXPORT.EXPORT_TYPE_THOUGHTS_DESC },
    { id: 'session_full', name: 'Session Full', icon: Archive, desc: STRINGS.EXPORT.EXPORT_TYPE_SESSION_FULL_DESC },
  ];

  const formats: { id: ExportFormat; name: string; icon: React.ComponentType<{ size?: string | number }>; tag?: string }[] = [
    { id: 'HDF5', name: 'HDF5 (.h5)', icon: Database, tag: 'RoboMimic' },
    { id: 'LeRobot', name: 'LeRobot', icon: Robot, tag: 'Hugging Face' },
    { id: 'JSONL', name: 'JSONL', icon: FileCode },
    { id: 'CSV', name: 'CSV', icon: FileCsv },
    { id: 'Parquet', name: 'Parquet', icon: Database, tag: 'Apache' },
  ];

  const scopeOptions = [
    { id: 'all', label: 'All Heartbeats' },
    { id: 'date_range', label: 'Date Range' },
    { id: 'session', label: 'Session Picker', disabled: !supabaseUrl || !supabaseKey || exportScope === 'all' },
    { id: 'heartbeat_range', label: 'Heartbeat Range' },
  ];

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70"
      onClick={() => setExportModalOpen(false)}
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        onClick={(e) => e.stopPropagation()}
        className="w-[820px] max-w-[calc(100vw-2rem)] max-h-[90vh] max-h-[calc(100vh-2rem)] flex flex-col"
      >
        <Panel className="border-border-subtle shadow-2xl overflow-hidden flex flex-col bg-bg-panel">
          {/* Header */}
          <div className="p-4 border-b border-border flex items-center justify-between bg-bg-panel shrink-0">
            <div className="flex items-center gap-2.5">
              <Export size={18} className="text-text-primary" />
              <h2 className="text-sm font-bold uppercase tracking-widest text-text-secondary">{STRINGS.EXPORT.TITLE}</h2>
            </div>
            <button
              onClick={() => setExportModalOpen(false)}
              className="text-text-tertiary hover:text-text-primary w-7 h-7 rounded-full flex items-center justify-center hover:bg-white/5 transition-colors"
              aria-label="Close Export Modal"
            >
              <X size={18} />
            </button>
          </div>

          <div className="flex overflow-hidden" style={{ minHeight: '480px' }}>
            {/* LEFT PANEL — Configuration */}
            <div className="flex-1 p-6 space-y-6 overflow-y-auto border-r border-border-subtle custom-scrollbar">
              {/* Scoping Selector - Single Agent vs All Active Agents */}
              <div className="space-y-2">
                <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">Export Scoping</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      handleExportScopeChange('single');
                      setZipPerAgent(false);
                    }}
                    className={`flex-1 h-9 flex items-center justify-center gap-2 border rounded-btn text-xs font-semibold transition-all ${
                      exportScope === 'single'
                        ? "border-white/20 bg-white/10 text-text-primary font-bold"
                        : "border-border text-text-tertiary hover:border-text-secondary"
                    }`}
                  >
                    <User size={14} />
                    <span>Agent ({activeAgentId})</span>
                  </button>
                  <button
                    onClick={() => {
                      handleExportScopeChange('all');
                    }}
                    className={`flex-1 h-9 flex items-center justify-center gap-2 border rounded-btn text-xs font-semibold transition-all ${
                      exportScope === 'all'
                        ? "border-white/20 bg-white/10 text-text-primary font-bold"
                        : "border-border text-text-tertiary hover:border-text-secondary"
                    }`}
                  >
                    <Users size={14} />
                    <span>All Active Agents</span>
                  </button>
                </div>
              </div>

              {/* ZIP archive isolation - Multi-Agent only */}
              <AnimatePresence>
                {exportScope === 'all' && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden space-y-1 border-b border-border-subtle pb-3"
                  >
                    <label className="flex items-center gap-2.5 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={zipPerAgent}
                        onChange={() => setZipPerAgent(!zipPerAgent)}
                        className="rounded border-border text-secondary cursor-pointer"
                      />
                      <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors font-medium">
                        Separate folder per agent inside ZIP archive
                      </span>
                    </label>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Export Type Selection */}
              <div className="space-y-2">
                <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">{STRINGS.EXPORT.EXPORT_TYPE}</label>
                <div className="grid grid-cols-3 gap-2">
                  {exportTypes.map((t) => {
                    const Icon = t.icon;
                    return (
                      <button
                        key={t.id}
                        onClick={() => setExportType(t.id)}
                        className={`p-3 flex flex-col items-center gap-1.5 text-center border rounded-btn transition-all ${
                          exportType === t.id
                            ? 'border-white/20 bg-white/10 text-text-primary font-bold'
                            : 'border-border text-text-tertiary hover:border-text-secondary hover:bg-white/[0.01]'
                        }`}
                      >
                        <Icon size={18} />
                        <span className="text-xs font-bold uppercase tracking-wide">{t.name}</span>
                        <span className="text-xs text-text-tertiary leading-snug font-normal line-clamp-2 mt-0.5">{t.desc}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Format Selection (Dataset only) */}
              <AnimatePresence>
                {exportType === 'dataset' && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="space-y-2 overflow-hidden border-b border-border-subtle pb-3"
                  >
                    <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">{STRINGS.EXPORT.SELECT_FORMAT}</label>
                    <div className="grid grid-cols-2 gap-2">
                      {formats.map((f) => (
                        <button
                          key={f.id}
                          onClick={() => setFormat(f.id)}
                          className={`h-10 flex items-center justify-center gap-2 border rounded-btn transition-all relative ${
                            format === f.id
                              ? "border-white/20 bg-white/10 text-text-primary font-bold"
                              : "border-border text-text-tertiary hover:border-text-secondary"
                          }`}
                        >
                          <f.icon size={15} />
                          <span className="text-xs font-bold">{f.name}</span>
                          {f.tag && (
                            <span className="text-xs bg-white/10 text-text-primary px-1 py-0.5 rounded font-mono font-normal">
                              {f.tag}
                            </span>
                          )}
                        </button>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Session Full sub-options */}
              <AnimatePresence>
                {exportType === 'session_full' && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="space-y-2 overflow-hidden border-b border-border-subtle pb-3"
                  >
                    <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">Include In Archive</label>
                    <div className="grid grid-cols-3 gap-2">
                      {[
                        { label: 'Thoughts', checked: includeThoughts, onChange: () => setIncludeThoughts(!includeThoughts) },
                        { label: 'Skills', checked: includeSkills, onChange: () => setIncludeSkills(!includeSkills) },
                        { label: 'Motors', checked: includeMotorPrograms, onChange: () => setIncludeMotorPrograms(!includeMotorPrograms) },
                      ].map((opt, i) => (
                        <label key={i} className="flex items-center gap-2 cursor-pointer group">
                          <input
                            type="checkbox"
                            checked={opt.checked}
                            onChange={opt.onChange}
                            className="rounded border-border text-secondary cursor-pointer"
                          />
                          <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">{opt.label}</span>
                        </label>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Task Tracker / Task Filter */}
              {availableTasks.length > 0 && (
                <div className="space-y-2 border-b border-border-subtle pb-3">
                  <div className="flex justify-between items-center">
                    <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">Training Task Filter</label>
                    {selectedTasks.length > 0 && (
                      <button
                        onClick={() => setSelectedTasks([])}
                        className="text-xs text-text-primary hover:underline"
                      >
                        Clear Task Filter
                      </button>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5 max-h-[80px] overflow-y-auto custom-scrollbar">
                    {availableTasks.map((t) => {
                      const isSelected = selectedTasks.includes(t);
                      return (
                        <button
                          key={t}
                          onClick={() => {
                            if (isSelected) {
                              setSelectedTasks(selectedTasks.filter(x => x !== t));
                            } else {
                              setSelectedTasks([...selectedTasks, t]);
                            }
                          }}
                          className={`px-2.5 py-1 rounded text-xs font-mono border transition-all ${
                            isSelected
                              ? 'border-white/20 bg-white/10 text-text-primary font-bold'
                              : 'border-border text-text-tertiary hover:border-text-secondary'
                          }`}
                        >
                          {t}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Scope Selector */}
              <div className="space-y-2">
                <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">{STRINGS.EXPORT.SCOPE_LABEL}</label>
                <div className="grid grid-cols-2 gap-2">
                  {scopeOptions.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => !s.disabled && setScope(s.id as ExportConfig['scope'])}
                      disabled={s.disabled}
                      className={`h-9 flex items-center justify-center border rounded-btn text-xs font-semibold transition-all ${
                        s.disabled
                          ? 'opacity-40 cursor-not-allowed border-dashed border-border text-text-tertiary'
                          : scope === s.id
                            ? 'border-white/20 bg-white/10 text-text-primary font-bold'
                            : 'border-border text-text-tertiary hover:border-text-secondary'
                      }`}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Conditional scope inputs */}
              <AnimatePresence mode="wait">
                {scope === 'date_range' && (
                  <motion.div
                    key="date"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="grid grid-cols-2 gap-3 overflow-hidden"
                  >
                    <div className="space-y-1.5">
                      <label className="text-xs uppercase tracking-wider text-text-tertiary">From Date</label>
                      <input
                        type="datetime-local"
                        value={dateFrom}
                        onChange={e => setDateFrom(e.target.value)}
                        className="w-full h-8 px-2.5 bg-bg-elevated border border-border rounded-btn text-xs text-text-primary focus:outline-none"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs uppercase tracking-wider text-text-tertiary">To Date</label>
                      <input
                        type="datetime-local"
                        value={dateTo}
                        onChange={e => setDateTo(e.target.value)}
                        className="w-full h-8 px-2.5 bg-bg-elevated border border-border rounded-btn text-xs text-text-primary focus:outline-none"
                      />
                    </div>
                  </motion.div>
                )}

                {scope === 'heartbeat_range' && (
                  <motion.div
                    key="heartbeat"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="grid grid-cols-2 gap-3 overflow-hidden"
                  >
                    <div className="space-y-1.5">
                      <label className="text-xs uppercase tracking-wider text-text-tertiary">Min Heartbeat</label>
                      <input
                        type="number"
                        min="0"
                        value={hbFrom}
                        onChange={e => setHbFrom(parseInt(e.target.value) || 0)}
                        className="w-full h-8 px-2.5 bg-bg-elevated border border-border rounded-btn text-xs text-text-primary focus:outline-none"
                      />
                    </div>
                    <div className="space-y-1.5">
                      <label className="text-xs uppercase tracking-wider text-text-tertiary">Max Heartbeat</label>
                      <input
                        type="number"
                        min="0"
                        value={hbTo}
                        onChange={e => setHbTo(parseInt(e.target.value) || 0)}
                        className="w-full h-8 px-2.5 bg-bg-elevated border border-border rounded-btn text-xs text-text-primary focus:outline-none"
                      />
                    </div>
                  </motion.div>
                )}

                {scope === 'session' && (
                  <motion.div
                    key="sessions"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="space-y-2 overflow-hidden"
                  >
                    <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">Select Sessions</label>
                    <div className="max-h-[140px] overflow-y-auto border border-border-subtle rounded-btn bg-black/10 divide-y divide-border-subtle custom-scrollbar">
                      {sessionsLoading ? (
                        <div className="p-3 text-center text-xs font-mono text-text-tertiary flex items-center justify-center gap-2">
                          <ArrowsClockwise size={14} className="animate-spin text-text-primary" />
                          Fetching sessions...
                        </div>
                      ) : sessionsError ? (
                        <div className="p-3 text-center text-xs font-mono text-red-400">{sessionsError}</div>
                      ) : availableSessions.length === 0 ? (
                        <div className="p-3 text-center text-xs text-text-tertiary uppercase">No database sessions found.</div>
                      ) : (
                        availableSessions.map((s) => {
                          const isSelected = selectedSessions.includes(s.id);
                          const toggleSession = () => {
                            if (isSelected) {
                              setSelectedSessions(selectedSessions.filter(id => id !== s.id));
                            } else {
                              setSelectedSessions([...selectedSessions, s.id]);
                            }
                          };
                          return (
                            <label
                              key={s.id}
                              className={`flex items-center justify-between px-3 py-2 cursor-pointer transition-colors ${
                                isSelected ? 'bg-white/10 text-text-primary' : 'hover:bg-white/[0.02]'
                              }`}
                            >
                              <div className="flex items-center gap-2.5">
                                <input
                                  type="checkbox"
                                  checked={isSelected}
                                  onChange={toggleSession}
                                  className="rounded border-border text-secondary cursor-pointer"
                                />
                                <span className="text-xs font-mono font-bold">{s.id.slice(0, 16)}</span>
                              </div>
                              <span className="text-xs font-mono text-text-tertiary">
                                {formatBytes(s.estimated_size_bytes || 0)}
                              </span>
                            </label>
                          );
                        })
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Memory Tiers Filter */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">{STRINGS.EXPORT.TIER_LABEL}</label>
                  <button onClick={() => setTierInfoOpen(!tierInfoOpen)} className="text-text-tertiary hover:text-text-primary" aria-label="Tier info">
                    <Info size={14} />
                  </button>
                </div>

                <AnimatePresence>
                  {tierInfoOpen && (
                    <motion.p
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="text-xs text-text-tertiary bg-white/5 p-2.5 rounded border border-border-subtle leading-normal"
                    >
                      The agent saves thoughts and context to specific memory tiers: Tier 1 is Working Memory (short-term), Tier 2 is Episodic Memory (medium-term), and Tier 3 is Long-term/Archival Memory. Select which tiers are included in the export.
                    </motion.p>
                  )}
                </AnimatePresence>

                <div className="flex gap-2">
                  {[1, 2, 3].map((tier) => (
                    <button
                      key={tier}
                      onClick={() => {
                        if (includeTiers.includes(tier)) {
                          setIncludeTiers(includeTiers.filter(t => t !== tier));
                        } else {
                          setIncludeTiers([...includeTiers, tier]);
                        }
                      }}
                      className={`flex-1 h-8 flex items-center justify-center border rounded-btn text-xs font-bold transition-all ${
                        includeTiers.includes(tier)
                          ? 'border-white/20 bg-white/10 text-text-primary font-bold'
                          : 'border-border text-text-tertiary hover:border-text-secondary'
                      }`}
                    >
                      Tier {tier}
                    </button>
                  ))}
                </div>
              </div>

              {/* Additional Filters */}
              <div className="space-y-2">
                <label className="text-xs font-bold uppercase tracking-widest text-text-tertiary">{STRINGS.EXPORT.FILTERS}</label>
                <div className="grid grid-cols-2 gap-y-2">
                  {[
                    { label: 'Exclude Injected', checked: excludeInjected, onChange: () => setExcludeInjected(!excludeInjected) },
                    { label: 'Successful Only', checked: successfulOnly, onChange: () => setSuccessfulOnly(!successfulOnly) },
                  ].map((opt, i) => (
                    <label key={i} className="flex items-center gap-2.5 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={opt.checked}
                        onChange={opt.onChange}
                        className="rounded border-border text-secondary cursor-pointer"
                      />
                      <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors">{opt.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* RIGHT PANEL — Preview */}
            <div className="w-[300px] p-5 bg-bg-elevated/20 flex flex-col shrink-0">
              <div className="flex items-center gap-2 mb-4">
                <Eye size={14} className="text-text-primary" />
                <span className="text-xs font-bold uppercase tracking-widest text-text-secondary">{STRINGS.EXPORT.PREVIEW_PANEL}</span>
              </div>

              <div className="flex-1 space-y-4">
                <div className="p-4 bg-black/10 rounded-panel border border-border-subtle space-y-4 text-left text-xs">
                  {/* Target Scoping Indicator */}
                  <div className="space-y-1">
                    <span className="text-xs uppercase text-text-tertiary">Target Scoping</span>
                    <span className="text-xs font-bold text-text-primary flex items-center gap-1.5 mt-0.5">
                      {exportScope === 'single' ? (
                        <>
                          <User size={14} className="text-text-primary" />
                          {activeAgentId}
                        </>
                      ) : (
                        <>
                          <Users size={14} className="text-text-primary" />
                          All Active Agents ({Object.keys(useAgentStore.getState().agents).length})
                        </>
                      )}
                    </span>
                  </div>

                  {/* Export Type */}
                  <div className="space-y-1">
                    <span className="text-xs uppercase text-text-tertiary">Type</span>
                    <span className="text-xs font-bold text-text-primary block">
                      {exportTypes.find(t => t.id === exportType)?.name}
                    </span>
                  </div>

                  {/* Format (dataset only) */}
                  {exportType === 'dataset' && (
                    <div className="space-y-1">
                      <span className="text-xs uppercase text-text-tertiary">Format</span>
                      <span className="text-xs font-bold text-text-primary block">{format}</span>
                    </div>
                  )}

                  {/* Task Filter indicator if active */}
                  {selectedTasks.length > 0 && (
                    <div className="space-y-1">
                      <span className="text-xs uppercase text-text-tertiary">Tasks Filtered</span>
                      <span className="text-xs font-mono text-text-primary font-bold block">
                        {selectedTasks.length} task(s) selected
                      </span>
                    </div>
                  )}

                  {/* Scope */}
                  <div className="space-y-1">
                    <span className="text-xs uppercase text-text-tertiary">Scope</span>
                    <span className="text-xs font-bold text-text-primary capitalize block">{scope.replace('_', ' ')}</span>
                  </div>

                  {/* Breakdown */}
                  <div className="space-y-1">
                    <span className="text-xs uppercase text-text-tertiary">Memories Match</span>
                    <span className="text-xs font-mono font-bold text-text-primary block">
                      {totalMemoryCount.toLocaleString()} rows
                    </span>
                  </div>

                  <div className="h-px bg-border-subtle" />

                  {/* Size Estimate */}
                  <div className="space-y-1">
                    <span className="text-xs uppercase text-text-tertiary">{STRINGS.EXPORT.ESTIMATED_SIZE}</span>
                    <span className="text-sm font-bold text-text-primary block">
                      {estimatedSize}
                    </span>
                  </div>
                </div>
              </div>

              {/* Progress and Action Button */}
              <div className="mt-auto pt-4 space-y-3 shrink-0">
                {isExporting && (
                  <div className="space-y-1 text-left">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-text-secondary uppercase">Progress</span>
                      <span className="text-text-primary font-bold">{exportProgress}%</span>
                    </div>
                    <div className="w-full h-1 bg-bg-elevated rounded-full overflow-hidden">
                      <div className="h-full bg-white/10 transition-all duration-300" style={{ width: `${exportProgress}%` }} />
                    </div>
                  </div>
                )}

                <button
                  onClick={handleExport}
                  disabled={isExporting || totalMemoryCount === 0}
                  className={`w-full h-10 rounded-btn text-xs font-bold uppercase tracking-widest flex items-center justify-center gap-2 transition-all cursor-pointer
                    ${isExporting
                      ? 'bg-white/10 border border-white/20 text-text-primary cursor-not-allowed'
                      : totalMemoryCount === 0
                        ? 'bg-border text-text-tertiary border-transparent cursor-not-allowed opacity-50'
                        : 'bg-white/10 hover:bg-white/10-hover text-white border-transparent shadow-md'
                    }`}
                >
                  {isExporting ? (
                    <>
                      <ArrowsClockwise size={14} className="animate-spin" />
                      <span>{STRINGS.EXPORT.EXPORTING(exportProgress)}</span>
                    </>
                  ) : (
                    <>
                      <DownloadSimple size={14} />
                      <span>{STRINGS.EXPORT.START_EXPORT}</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </Panel>
      </motion.div>
    </div>
  );
};
