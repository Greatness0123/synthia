import JSZip from 'jszip';
import { getSupabaseClient } from './supabaseClient';
import { ExportConfig } from '../types/export';
import { useAgentStore } from '../store/agentStore';
import { writeParquet } from './parquetWriter';
import { formatMemoriesToHDF5 } from './hdf5Writer';

const PAGE_SIZE = 500;
const MAX_ROWS = 50_000;

// Simple function to trigger browser download of a Blob
function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export async function runClientSideExport(
  config: ExportConfig,
  supabaseUrl: string,
  supabaseKey: string,
  onProgress: (percent: number) => void
): Promise<void> {
  onProgress(5);
  let memories: any[] = [];

  // 1. Gather memories from Supabase if configured, otherwise fallback to local store
  if (supabaseUrl && supabaseKey) {
    try {
      const supabase = getSupabaseClient(supabaseUrl, supabaseKey);
      if (!supabase) throw new Error('Supabase not configured');

      // Build the base query with filters
      let baseQuery = supabase.from('memories').select('*', { count: 'exact' }).in('agent_id', config.agentIds);

      if (config.scope === 'date_range') {
        if (config.dateFrom) baseQuery = baseQuery.gte('created_at', config.dateFrom);
        if (config.dateTo) baseQuery = baseQuery.lte('created_at', config.dateTo);
      } else if (config.scope === 'session') {
        if (config.sessionIds && config.sessionIds.length > 0) {
          baseQuery = baseQuery.in('session_id', config.sessionIds);
        }
      } else if (config.scope === 'heartbeat_range') {
        if (config.heartbeatFrom !== undefined) baseQuery = baseQuery.gte('heartbeat', config.heartbeatFrom);
        if (config.heartbeatTo !== undefined) baseQuery = baseQuery.lte('heartbeat', config.heartbeatTo);
      }

      if (config.includeTiers && config.includeTiers.length > 0) {
        baseQuery = baseQuery.in('tier', config.includeTiers);
      }
      if (config.excludeInjected) {
        baseQuery = baseQuery.not('injected', 'eq', true);
      }
      if (config.successfulOnly) {
        baseQuery = baseQuery.neq('outcome', 'failure');
      }

      // Paginated fetch: 500 rows per page, max 50k rows
      let offset = 0;
      let totalFetched = 0;
      let hitCap = false;

      while (!hitCap) {
        const remaining = MAX_ROWS - totalFetched;
        const limit = Math.min(PAGE_SIZE, remaining);
        if (limit <= 0) { hitCap = true; break; }

        const { data, error, count } = await baseQuery
          .order('created_at', { ascending: true })
          .range(offset, offset + limit - 1);

        if (error) {
          console.error('[ClientExport] Paginated query error:', error.message);
          break;
        }

        if (data && data.length > 0) {
          memories = memories.concat(data);
          totalFetched += data.length;
          offset += data.length;

          // Update progress (5% to 40% during fetch)
          const estimatedTotal = count || totalFetched;
          const fetchPct = Math.min(40, 5 + Math.round((totalFetched / estimatedTotal) * 35));
          onProgress(fetchPct);

          // Stop if we got fewer rows than requested (end of data)
          if (data.length < limit) break;
        } else {
          break;
        }

        // Safety: stop if we hit the cap
        if (totalFetched >= MAX_ROWS) {
          hitCap = true;
          console.warn(`[ClientExport] Export capped at ${MAX_ROWS} rows`);
        }
      }
    } catch (err) {
      console.warn('[ClientExport] Supabase query failed, falling back to local memory store:', err);
    }
  }

  // Local fallback if Supabase unavailable or yielded no records
  if (memories.length === 0) {
    const agents = useAgentStore.getState().agents;
    const targetAgents = config.agentIds.flatMap(id => agents[id] ? [agents[id]] : []);
    const rawMemories = targetAgents.flatMap(a => (a.memories || []).map(m => ({
      agent_id: m.agentId || 'agent_0',
      id: m.id,
      heartbeat: m.heartbeat,
      tier: m.tier,
      thought: m.thought,
      summary: m.summary,
      visual_description: m.summary || '',
      action_taken: m.actionTaken || '',
      reward_signal: m.rewardSignal,
      injected: m.isInjected,
      goal_at_time: m.goalAtTime,
      session_id: m.sessionId || 'local_session',
      created_at: new Date().toISOString(),
    })));

    memories = rawMemories.filter(m => {
      if (config.includeTiers && !config.includeTiers.includes(m.tier as any)) return false;
      if (config.excludeInjected && m.injected) return false;
      if (config.successfulOnly && (m.reward_signal || 0) < 0.5) return false;
      if (config.scope === 'heartbeat_range') {
        if (config.heartbeatFrom !== undefined && m.heartbeat < config.heartbeatFrom) return false;
        if (config.heartbeatTo !== undefined && m.heartbeat > config.heartbeatTo) return false;
      }
      return true;
    });
  }

  // Apply taskFilter if specified
  if (config.taskFilter && config.taskFilter.length > 0) {
    memories = memories.filter(m => m.goal_at_time && config.taskFilter!.includes(m.goal_at_time));
  }

  if (!memories || memories.length === 0) {
    throw new Error('No memories found matching the export criteria');
  }

  onProgress(40);

  // Fetch session meta if session_full is chosen and Supabase is configured
  let sessionsMeta: any[] = [];
  let skillsMeta: any[] = [];
  if (config.exportType === 'session_full' && supabaseUrl && supabaseKey) {
    try {
      const supabase = getSupabaseClient(supabaseUrl, supabaseKey);
      if (!supabase) throw new Error('Supabase not configured');
      const sessionIds = [...new Set(memories.map((m) => m.session_id))].filter(Boolean);
      if (sessionIds.length > 0) {
        const { data: sessions } = await supabase.from('sessions').select('*').in('id', sessionIds);
        if (sessions) sessionsMeta = sessions;
      }

      const { data: skills } = await supabase.from('skills').select('*').in('agent_id', config.agentIds);
      if (skills) skillsMeta = skills;
    } catch (e) {
      console.warn('[ClientExport] Fetch session meta warning:', e);
    }
  }

  onProgress(60);

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const baseFilename = `synthia_export_${config.exportType}_${timestamp}`;

  // 2. Perform formatting based on ExportType
  if (config.exportType === 'dataset') {
    const format = config.format || 'JSONL';

    if (config.zipPerAgent && config.agentIds.length > 1) {
      const zip = new JSZip();

      const agentGroups: Record<string, any[]> = {};
      config.agentIds.forEach((id) => {
        agentGroups[id] = memories.filter((m) => m.agent_id === id);
      });

      Object.entries(agentGroups).forEach(([agentId, agentMemories]) => {
        if (agentMemories.length === 0) return;

        if (format === 'CSV') {
          zip.file(`${agentId}/export.csv`, formatCSV(agentMemories));
        } else if (format === 'LeRobot') {
          zip.file(`${agentId}/lerobot_dataset.jsonl`, formatLeRobot(agentMemories));
        } else if (format === 'Parquet') {
          zip.file(`${agentId}/data.parquet`, formatParquet(agentMemories));
        } else if (format === 'HDF5') {
          zip.file(`${agentId}/dataset.hdf5`, formatMemoriesToHDF5(agentMemories));
        } else {
          zip.file(`${agentId}/data.jsonl`, formatJSONL(agentMemories));
        }
      });

      onProgress(85);
      const content = await zip.generateAsync({ type: 'blob' });
      triggerDownload(content, `${baseFilename}.zip`);
    } else {
      if (format === 'CSV') {
        const csvContent = formatCSV(memories);
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        triggerDownload(blob, `${baseFilename}.csv`);
      } else if (format === 'LeRobot') {
        const leRobotContent = formatLeRobot(memories);
        const blob = new Blob([leRobotContent], { type: 'application/x-jsonlines;charset=utf-8;' });
        triggerDownload(blob, `${baseFilename}_lerobot.jsonl`);
      } else if (format === 'Parquet') {
        const parquetBytes = formatParquet(memories);
        const blob = new Blob([parquetBytes.buffer as ArrayBuffer], { type: 'application/octet-stream' });
        triggerDownload(blob, `${baseFilename}.parquet`);
      } else if (format === 'HDF5') {
        const hdf5Bytes = formatMemoriesToHDF5(memories);
        const blob = new Blob([hdf5Bytes.buffer as ArrayBuffer], { type: 'application/x-hdf5' });
        triggerDownload(blob, `${baseFilename}.hdf5`);
      } else {
        const jsonlContent = formatJSONL(memories);
        const blob = new Blob([jsonlContent], { type: 'application/x-jsonlines;charset=utf-8;' });
        triggerDownload(blob, `${baseFilename}.jsonl`);
      }
    }
  } else if (config.exportType === 'thoughts_report') {
    const reportContent = formatThoughtsReport(memories);

    if (config.zipPerAgent && config.agentIds.length > 1) {
      const zip = new JSZip();
      const agentGroups: Record<string, any[]> = {};
      config.agentIds.forEach((id) => {
        agentGroups[id] = memories.filter((m) => m.agent_id === id);
      });

      Object.entries(agentGroups).forEach(([agentId, agentMemories]) => {
        if (agentMemories.length === 0) return;
        zip.file(`${agentId}/thoughts_report.md`, formatThoughtsReport(agentMemories));
      });

      onProgress(85);
      const content = await zip.generateAsync({ type: 'blob' });
      triggerDownload(content, `${baseFilename}.zip`);
    } else {
      const blob = new Blob([reportContent], { type: 'text/markdown;charset=utf-8;' });
      triggerDownload(blob, `${baseFilename}.md`);
    }
  } else if (config.exportType === 'session_full') {
    const sessionFullData = {
      export_metadata: {
        exported_at: new Date().toISOString(),
        scoping: config.scope,
        agent_ids: config.agentIds,
      },
      sessions: sessionsMeta,
      memories: memories.map(m => ({
        ...m,
        frame_buffer: undefined
      })),
      skills: skillsMeta,
    };

    if (config.zipPerAgent && config.agentIds.length > 1) {
      const zip = new JSZip();
      const agentGroups: Record<string, any[]> = {};
      config.agentIds.forEach((id) => {
        agentGroups[id] = memories.filter((m) => m.agent_id === id);
      });

      Object.entries(agentGroups).forEach(([agentId, agentMemories]) => {
        const agentSessions = sessionsMeta.filter((s) => s.agent_id === agentId);
        const agentSkills = skillsMeta.filter((sk) => sk.agent_id === agentId);

        const agentData = {
          export_metadata: {
            exported_at: new Date().toISOString(),
            agent_id: agentId,
          },
          sessions: agentSessions,
          memories: agentMemories.map(m => ({ ...m, frame_buffer: undefined })),
          skills: agentSkills,
        };

        zip.file(`${agentId}/session_full.json`, JSON.stringify(agentData, null, 2));
      });

      onProgress(85);
      const content = await zip.generateAsync({ type: 'blob' });
      triggerDownload(content, `${baseFilename}.zip`);
    } else {
      const blob = new Blob([JSON.stringify(sessionFullData, null, 2)], { type: 'application/json;charset=utf-8;' });
      triggerDownload(blob, `${baseFilename}.json`);
    }
  }

  onProgress(100);
}

// Helper: Parquet Format
function formatParquet(memories: any[]): Uint8Array {
  const rows = memories.map((m) => ({
    agent_id: String(m.agent_id || 'agent_0'),
    session_id: String(m.session_id || ''),
    heartbeat: typeof m.heartbeat === 'number' ? m.heartbeat : 0,
    tier: typeof m.tier === 'number' ? m.tier : 1,
    thought: String(m.thought || ''),
    visual_description: String(m.visual_description || ''),
    audio_state: String(m.audio_state || ''),
    observation_before: JSON.stringify(getTelemetry(m).observation_before || null),
    observation_after: JSON.stringify(getTelemetry(m).observation_after || null),
    joint_positions: JSON.stringify(getJointPositions(m)),
    joint_velocities: JSON.stringify(getJointVelocities(m)),
    root_position: JSON.stringify(getRootPosition(m)),
    action_taken: JSON.stringify(m.action_taken || null),
    requested_joint_action: JSON.stringify(getRequestedJointAction(m)),
    outcome: String(m.outcome || ''),
    reward_signal: typeof m.reward_signal === 'number' ? m.reward_signal : 0,
    goal_at_time: String(m.goal_at_time || ''),
    injected: Boolean(m.injected),
    created_at: String(m.created_at || ''),
  }));
  return writeParquet(rows);
}

// Helper: CSV Format
function formatCSV(memories: any[]): string {
  const header = 'agent_id,heartbeat,tier,thought,action_json,joint_positions,root_position,outcome,reward,session_id\n';
  const rows = memories.map((m) => {
    const actionJson = JSON.stringify(m.action_taken || {}).replace(/"/g, '""');
    const thought = (m.thought || '').replace(/"/g, '""');
    const jointPositions = JSON.stringify(getJointPositions(m)).replace(/"/g, '""');
    const rootPosition = JSON.stringify(getRootPosition(m)).replace(/"/g, '""');
    const outcome = m.outcome || '';
    const agentId = m.agent_id || 'agent_0';
    return `${agentId},${m.heartbeat},${m.tier},"${thought}","${actionJson}","${jointPositions}","${rootPosition}",${outcome},${m.reward_signal ?? 0},${m.session_id ?? ''}`;
  });
  return header + rows.join('\n');
}

// Helper: JSONL Format
function formatJSONL(memories: any[]): string {
  return memories
    .map((m) => {
      return JSON.stringify({
        agent_id: m.agent_id || 'agent_0',
        session_id: m.session_id ?? null,
        messages: [
          { role: 'system', content: 'You are SYNTHIA, an AI embodiment.' },
          { role: 'user', content: `${m.visual_description || 'Step'} Audio: ${m.audio_state || ''}` },
          {
            role: 'assistant',
            content: `${m.thought || ''}---ACTION---${JSON.stringify({
              actions: m.action_taken,
              memory_write: { tier: m.tier, summary: m.visual_description },
            })}`,
          },
          { role: 'observation', content: {
            joint_positions: getJointPositions(m),
            joint_velocities: getJointVelocities(m),
            root_position: getRootPosition(m),
          } },
        ],
      });
    })
    .join('\n');
}

// Helper: LeRobot Hugging Face Dataset Format
function formatLeRobot(memories: any[]): string {
  let episodeIndex = 0;
  let currentSession = '';
  return memories
    .map((m, idx) => {
      const sessionId = m.session_id || m.sessionId || 'session_0';
      if (sessionId !== currentSession) {
        currentSession = sessionId;
        episodeIndex++;
      }
      return JSON.stringify({
        episode_index: episodeIndex,
        frame_index: m.heartbeat || idx,
        timestamp: (m.heartbeat || idx) * 0.1,
        task: m.goal_at_time || m.goalAtTime || 'general_embodied_task',
        observation: {
          thought: m.thought || '',
          visual_description: m.visual_description || m.visualDescription || '',
          audio_state: m.audio_state || m.audioState || '',
          before: getTelemetry(m).observation_before || null,
          after: getTelemetry(m).observation_after || null,
        },
        state: getJointPositions(m),
        action: getRequestedJointAction(m),
        reward: m.reward_signal ?? m.rewardSignal ?? 0,
        done: m.outcome === 'success' || m.outcome === 'failure',
      });
    })
    .join('\n');
}

function getTelemetry(memory: any): any {
  if (memory?.self_questions && typeof memory.self_questions === 'object') {
    return memory.self_questions;
  }
  if (typeof memory?.self_questions === 'string') {
    try {
      return JSON.parse(memory.self_questions);
    } catch {
      return {};
    }
  }
  return {};
}

function getJointPositions(memory: any): number[] {
  const telemetry = getTelemetry(memory);
  const pose = telemetry.observation_after?.proprioception?.current_pose;
  if (Array.isArray(pose)) return pose.filter((value) => typeof value === 'number' && Number.isFinite(value));
  if (Array.isArray(memory?.joint_states)) return memory.joint_states;
  return [];
}

function getJointVelocities(memory: any): Record<string, number> | number[] {
  const telemetry = getTelemetry(memory);
  return telemetry.observation_after?.joint_velocities || memory?.joint_velocities || [];
}

function getRootPosition(memory: any): number[] {
  const telemetry = getTelemetry(memory);
  const position = telemetry.observation_after?.root_state?.position || memory?.root_position;
  return Array.isArray(position) ? position : [];
}

function getRequestedJointAction(memory: any): any {
  const telemetry = getTelemetry(memory);
  return telemetry.requested_action?.joint_overrides || memory?.action_taken?.joint_overrides || memory?.action_taken || {};
}

// Helper: Thoughts Report format
function formatThoughtsReport(memories: any[]): string {
  const lines: string[] = [];
  lines.push('# SYNTHIA Cognitive Thoughts Report');
  lines.push('');
  lines.push(`Generated: ${new Date().toLocaleString()}`);
  lines.push(`Total Memories: ${memories.length}`);
  lines.push('');

  const sessionGroups = new Map<string, any[]>();
  memories.forEach((m) => {
    const sId = m.session_id || 'unknown';
    if (!sessionGroups.has(sId)) {
      sessionGroups.set(sId, []);
    }
    sessionGroups.get(sId)!.push(m);
  });

  lines.push('## Table of Contents');
  lines.push('');
  let sessionIndex = 0;
  for (const [sessionId, sessionMemories] of sessionGroups) {
    sessionIndex++;
    const agentId = sessionMemories[0]?.agent_id || 'agent_0';
    lines.push(`${sessionIndex}. [Session ${sessionId} (${agentId})](#session-${sessionIndex}) — ${sessionMemories.length} memories`);
  }
  lines.push('');
  lines.push('---');
  lines.push('');

  sessionIndex = 0;
  for (const [sessionId, sessionMemories] of sessionGroups) {
    sessionIndex++;
    const firstMemory = sessionMemories[0];
    const lastMemory = sessionMemories[sessionMemories.length - 1];
    const agentId = firstMemory?.agent_id || 'agent_0';

    lines.push(`## Session ${sessionIndex} — ${sessionId} [Agent: ${agentId}]`);
    lines.push('');
    lines.push(`- **Heartbeats:** ${firstMemory.heartbeat} → ${lastMemory.heartbeat}`);
    lines.push(`- **Memories:** ${sessionMemories.length}`);
    lines.push(`- **Started:** ${firstMemory.created_at || 'N/A'}`);
    lines.push('');

    const tier1 = sessionMemories.filter((m) => m.tier === 1);
    const tier2 = sessionMemories.filter((m) => m.tier === 2);
    const tier3 = sessionMemories.filter((m) => m.tier === 3);
    lines.push('**Tier Breakdown:**');
    lines.push(`- Tier 1 (Working): ${tier1.length}`);
    lines.push(`- Tier 2 (Episodic): ${tier2.length}`);
    lines.push(`- Tier 3 (Long-term): ${tier3.length}`);
    lines.push('');

    for (const m of sessionMemories) {
      lines.push(`### Heartbeat ${m.heartbeat} — Tier ${m.tier} [Agent: ${m.agent_id || 'agent_0'}]`);
      lines.push('');
      lines.push(`**Thought:**`);
      lines.push('');
      lines.push(`> ${m.thought || '(no thought)'}`);
      lines.push('');
      if (m.visual_description) {
        lines.push(`**Visual:** ${m.visual_description}`);
      }
      if (m.outcome) {
        lines.push(`**Outcome:** ${m.outcome}`);
      }
      if (m.reward_signal !== undefined && m.reward_signal !== null) {
        lines.push(`**Reward:** ${m.reward_signal.toFixed(2)}`);
      }
      if (m.goal_at_time) {
        lines.push(`**Goal:** ${m.goal_at_time}`);
      }
      if (m.injected) {
        lines.push(`**Injected:** Yes`);
      }
      lines.push('');
      lines.push('---');
      lines.push('');
    }
  }

  lines.push('## Summary Statistics');
  lines.push('');
  const avgReward = memories.reduce((sum, m) => sum + (m.reward_signal || 0), 0) / memories.length;
  const successes = memories.filter((m) => m.outcome && !m.outcome.includes('fail')).length;
  lines.push(`- **Average Reward:** ${avgReward.toFixed(3)}`);
  lines.push(`- **Success Rate:** ${((successes / memories.length) * 100).toFixed(1)}%`);
  lines.push(`- **Injected Memories:** ${memories.filter((m) => m.injected).length}`);
  lines.push('');

  return lines.join('\n');
}
