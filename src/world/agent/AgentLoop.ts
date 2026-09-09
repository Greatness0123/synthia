/**
 * Main cycle loop per agent running fully client-side.
 * Originally ported from coordinator/src/agentLoop.ts; the coordinator codebase was
 * removed in Phase 12. This is now the single authoritative per-agent loop.
 */

import { PayloadBuilder } from './payloadBuilder';
import { InferenceClient } from './InferenceClient';
import { MemoryManager, MemoryEntry } from './memoryManager';
import { IdentityManager, AgentIdentity } from './identityManager';
import { useAgentStore } from '../../store/agentStore';
import { useAgentRuntimeStore } from '../../store/agentRuntimeStore';
import { useIdentityStore } from '../../store/identityStore';
import { stripSpeechTags } from '../../utils/speech';
import { cleanThoughtText } from '../../utils/thoughtUtils';
import { synthiaToast } from '../../utils/synthiaToast';

function decodeFrameBuffer(frame: unknown): Uint8Array | undefined {
  if (typeof frame !== 'string' || !frame) return undefined;
  const encoded = frame.includes(',') ? frame.split(',', 2)[1] : frame;
  try {
    const binary = atob(encoded);
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  } catch {
    return undefined;
  }
}

interface AgentLoopConfig {
  agentId: string;
  cycleMs: number;
  supabaseUrl: string;
  supabaseKey: string;
  captureWorldState: () => Promise<any>;
}

export class AgentLoop {
  private config: AgentLoopConfig;
  private interval: any = null;
  private isProcessing: boolean = false;

  private payloadBuilder: PayloadBuilder;
  private inferenceClient: InferenceClient;
  private memoryManager: MemoryManager;
  private identityManager: IdentityManager;
  private identity: AgentIdentity | null = null;
  private lastIdentityFeedback: any = null;

  private directives = { mode: 'free_will', goal: '' };
  private heartbeat = 0;
  private lastActionFeedback: any[] = [];
  private currentSessionId: string | null = null;
  private pendingCycles: Map<string, any> = new Map();
  private cycleTimers: Map<string, ReturnType<typeof setTimeout>> = new Map();
  private hasWarnedConnectionError: boolean = false;

  constructor(config: AgentLoopConfig) {
    this.config = config;
    this.memoryManager = new MemoryManager(config.supabaseUrl, config.supabaseKey);
    this.identityManager = new IdentityManager(config.supabaseUrl, config.supabaseKey);
    this.payloadBuilder = new PayloadBuilder(this.memoryManager);
    this.inferenceClient = new InferenceClient();
  }

  public setDirective(mode: string, goal: string) {
    this.directives = { mode, goal };
  }

  public recordActionFeedback(rejected: any[]) {
    this.lastActionFeedback = rejected;
    console.log(`[AgentLoop (${this.config.agentId})] recorded ${rejected.length} rejected joint action(s) for next payload`);
  }

  public setProvider(type: string, endpoint: string, apiKey?: string, model?: string) {
    this.inferenceClient.setProvider(type, endpoint, apiKey, model);
  }

  public setCycleMs(ms: number): void {
    this.config.cycleMs = ms;
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = setInterval(() => this.cycle(), ms);
    }
  }

  public updateSupabase(url: string, key: string) {
    this.config.supabaseUrl = url;
    this.config.supabaseKey = key;
    this.memoryManager = new MemoryManager(url, key);
    this.payloadBuilder = new PayloadBuilder(this.memoryManager);
  }

  public async manualIdentityUpdate(update: any, reason: string): Promise<{ ok: boolean; error?: string }> {
    const result = await this.identityManager.applyIdentityUpdate(this.config.agentId, {
      ...update,
      reason,
    }, true);
    if (result.ok && result.identity) {
      this.identity = result.identity;
      useIdentityStore.getState().setIdentity(this.config.agentId, this.identity);
    }
    return { ok: result.ok, error: result.error || result.rejection };
  }

  public async start() {
    if (this.interval) return;

    this.currentSessionId = `session_${Date.now()}_${this.config.agentId}`;

    useAgentRuntimeStore.getState().setLoopState(this.config.agentId, 'running');

    const rehydrationSummary = "Reconnecting to neural lattice... archives accessed... current status: operational.";
    const store = useAgentStore.getState() as any;
    const agentId = this.config.agentId;

    // Fetch or seed identity record
    this.identity = await this.identityManager.ensureIdentity(agentId);
    useIdentityStore.getState().setIdentity(agentId, this.identity);
    console.log(`[AgentLoop (${agentId})] identity loaded: name=${this.identity.name}`);
    const isActiveAgent = store.activeAgentId === agentId;

    // Send rehydration tokens to the agent's own record. Only the active agent
    // drives the flat mirror + startup modal so spawning additional agents never
    // re-triggers the RehydrationModal.
    if (store.setRehydrationSummaryForAgent) {
      store.setRehydrationSummaryForAgent(agentId, '');
      for (const token of rehydrationSummary.split(' ')) {
        store.appendRehydrationTokenForAgent(agentId, token + ' ');
        await new Promise(resolve => setTimeout(resolve, 80));
      }
      if (isActiveAgent) {
        store.setHasRehydrated(true);
      }
    }

    this.interval = setInterval(() => this.cycle(), this.config.cycleMs || 2000);
    console.log(`[AgentLoop (${this.config.agentId})] started independent inference loop`);
  }

  public stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    // Clear all pending auto-finalize timers
    for (const timerId of this.cycleTimers.values()) {
      clearTimeout(timerId);
    }
    this.cycleTimers.clear();
    this.pendingCycles.clear();
    if (this.currentSessionId) {
      this.memoryManager.endSession(this.currentSessionId);
      this.currentSessionId = null;
    }
    useAgentRuntimeStore.getState().setLoopState(this.config.agentId, 'stopped');
    console.log(`[AgentLoop (${this.config.agentId})] stopped`);
  }

  public pause() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    useAgentRuntimeStore.getState().setLoopState(this.config.agentId, 'paused');
    console.log(`[AgentLoop (${this.config.agentId})] paused`);
  }

  public resume() {
    if (this.interval) return; // already running
    this.hasWarnedConnectionError = false;
    useAgentRuntimeStore.getState().setLoopState(this.config.agentId, 'running');
    this.interval = setInterval(() => this.cycle(), this.config.cycleMs || 2000);
    console.log(`[AgentLoop (${this.config.agentId})] resumed`);
  }

  private async cycle() {
    if (this.isProcessing) {
      console.log(`[AgentLoop (${this.config.agentId})] Skipping cycle: previous inference is still in-flight.`);
      return;
    }

    // Nothing configured — do not run cycles. This prevents the status badge
    // from flickering thinking → acting → idle when no provider/key is set.
    if (!this.inferenceClient.hasCredentials()) {
      useAgentRuntimeStore.getState().setLoopState(this.config.agentId, 'not_started');
      return;
    }

    // Circuit breaker: skip entirely during backoff to avoid zombie HTTP spam
    if (this.inferenceClient.isInBackoff()) {
      useAgentRuntimeStore.getState().setLoopState(this.config.agentId, 'retrying');
      return;
    }

    // Call callback to capture the current world state
    const worldState = await this.config.captureWorldState();
    if (!worldState) {
      console.log(`[AgentLoop (${this.config.agentId})] Skipping cycle: world state not available yet.`);
      return;
    }

    // Sync agent heartbeat state
    this.heartbeat = worldState.heartbeat || this.heartbeat + 1;

    // Direct read of pending injection + directive/goal from store for this agent.
    // DirectivePanel/InjectionInput write to agents[agentId] via *ForAgent actions.
    const store = useAgentStore.getState() as any;
    const agentState = store.agents?.[this.config.agentId];
    let pendingInjection: string | null = null;
    if (agentState && agentState.pendingInjection) {
      pendingInjection = agentState.pendingInjection;
      if (store.setPendingInjectionForAgent) {
        store.setPendingInjectionForAgent(this.config.agentId, null);
      }
    }

    // Per-agent directive mode + goal (client path). Falls back to last set via
    // setDirective() so the loop always carries the latest intent.
    if (agentState) {
      if (agentState.directiveMode) {
        this.directives.mode = agentState.directiveMode === 'training' ? 'training' : 'free_will';
      }
      if (agentState.currentGoal != null) {
        this.directives.goal = agentState.currentGoal;
      }
    }

    if (pendingInjection) {
      worldState.injected_thought = pendingInjection;
      console.log(`[AgentLoop (${this.config.agentId})] Injection consumed: "${pendingInjection}"`);
    }

    this.isProcessing = true;
    try {
      const masteredSkills = await this.memoryManager.getMasteredSkills(this.config.agentId);

      // Force session ID
      worldState.sessionId = this.currentSessionId || `session_${this.config.agentId}`;

      const useActionDictionary = agentState?.useActionDictionary !== undefined
        ? agentState.useActionDictionary
        : (useAgentRuntimeStore.getState().configs[this.config.agentId]?.useActionDictionary ?? true);

      const payload = await this.payloadBuilder.build(worldState, this.config.agentId, {
        motorPrograms: [],
        masteredSkills,
        physicalFeedback: this.lastActionFeedback,
        identity: this.identity,
        identityFeedback: this.lastIdentityFeedback,
        useActionDictionary,
        ...this.directives
      });
      this.lastActionFeedback = [];
      this.lastIdentityFeedback = null;

      // Preserve previous uncommitted thought if present before resetting currentThought
      const existingThought = store.agents?.[this.config.agentId]?.currentThought;
      if (existingThought && existingThought.trim().length > 10 && store.addThoughtForAgent) {
        const cleanedPrev = cleanThoughtText(existingThought);
        if (cleanedPrev) {
          store.addThoughtForAgent(this.config.agentId, {
            id: `thought_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
            text: cleanedPrev,
            timestamp: Date.now(),
            isInjected: false,
            outcome: 'recorded'
          });
        }
      }

      // Clear pending thoughts for this agent in store before writing new stream
      if (store.setCurrentThoughtForAgent) {
        store.setCurrentThoughtForAgent(this.config.agentId, '');
        store.setStatusForAgent(this.config.agentId, 'thinking');
      }

      console.log(`[AgentLoop (${this.config.agentId})] Sending inference request...`);
      const result = await this.inferenceClient.infer(payload, (token) => {
        if (store.appendThoughtTokenForAgent) {
          store.appendThoughtTokenForAgent(this.config.agentId, token);
        }
      });

      if (store.setStatusForAgent) {
        store.setStatusForAgent(this.config.agentId, 'acting');
      }

      console.log(`[AgentLoop (${this.config.agentId})] Inference completed. Parsing action JSON.`);
      const cleanedThought = cleanThoughtText(result.thoughtTokens || '');
      if (cleanedThought && store.addThoughtForAgent) {
        store.addThoughtForAgent(this.config.agentId, {
          id: `thought_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
          text: cleanedThought,
          timestamp: Date.now(),
          isInjected: !!worldState.injected_thought,
        });
      }
      const actionData = this.parseAndValidateAction(result.actionJson);
      if (actionData) {
        // Dispatch parsed action custom event which useWorld's handleAction will receive
        const actionEvent = new CustomEvent('synthia:action', {
          detail: {
            programSequence: actionData.actions?.program_sequence || [],
            jointOverrides: actionData.actions?.joint_overrides || {},
            sequence: actionData.sequence || null,
            activeGaitPhase: typeof actionData.activeGaitPhase === 'boolean' ? actionData.activeGaitPhase : false,
            gazeTarget: actionData.gaze_target || null,
            agentId: this.config.agentId
          }
        });
        window.dispatchEvent(actionEvent);

        // Record pending cycle for memory write
        const cycleTimestamp = Date.now();
        const cycleId = `cycle_${cycleTimestamp}`;
        const cycleData = {
          id: cycleId,
          result,
          actionData,
          worldState,
          timestamp: cycleTimestamp,
          finalized: false
        };

        this.pendingCycles.set(cycleId, cycleData);

        // Auto finalize with 'unknown' after 4s
        const timerId = setTimeout(() => {
          this.cycleTimers.delete(cycleId);
          const cycle = this.pendingCycles.get(cycleId);
          if (cycle && !cycle.finalized) {
            this.finalizeCycle({ description: 'timeout', reward: 0 }, cycleId);
          }
        }, 4000);
        this.cycleTimers.set(cycleId, timerId);

      } else {
        console.warn(`[AgentLoop (${this.config.agentId})] Action parse failed. XML action raw text: ${result.actionJson}`);
      }

    } catch (err: any) {
      // Preserve partial thought streamed before error/disconnect occurred
      const partialThought = store.agents?.[this.config.agentId]?.currentThought;
      if (partialThought && partialThought.trim().length > 10 && store.addThoughtForAgent) {
        const cleanedPartial = cleanThoughtText(partialThought);
        if (cleanedPartial) {
          store.addThoughtForAgent(this.config.agentId, {
            id: `thought_err_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
            text: cleanedPartial,
            timestamp: Date.now(),
            isInjected: !!worldState.injected_thought,
            outcome: 'interrupted (offline/rate limit)'
          });
        }
      }

      const isConnectionError = err?.message?.includes('Failed to fetch') ||
                                err?.message?.includes('NetworkError') ||
                                err?.message?.includes('ERR_CONNECTION_REFUSED') ||
                                err?.name === 'TypeError';
      const isBackoff = err?.message?.includes('backoff active');

      if (isConnectionError || isBackoff) {
        if (!this.hasWarnedConnectionError) {
          this.hasWarnedConnectionError = true;
          console.warn(`[AgentLoop (${this.config.agentId})] Local inference server offline (${err.message}). Retrying...`);
          synthiaToast.warning(`Inference server offline. Retrying...`);
        }
      } else {
        console.error(`[AgentLoop (${this.config.agentId})] Cycle error:`, err);
        const shortError = err?.message ? String(err.message).slice(0, 120) : 'Inference failed';
        synthiaToast.error(`Inference failed for ${this.config.agentId}: ${shortError}`);
      }

      useAgentRuntimeStore.getState().setLoopState(this.config.agentId, 'error');
      if (store.setStatusForAgent) {
        store.setStatusForAgent(this.config.agentId, 'error');
      }
    } finally {
      this.isProcessing = false;
      const loopState = useAgentRuntimeStore.getState().getLoopState(this.config.agentId);
      if (loopState !== 'error' && loopState !== 'retrying' && store.setStatusForAgent) {
        store.setStatusForAgent(this.config.agentId, 'idle');
      }
    }
  }

  public async handleOutcome(outcome: any) {
    let latestCycleId: string | null = null;
    let latestTs = 0;
    for (const [id, cycle] of this.pendingCycles.entries()) {
      if (!cycle.finalized && cycle.timestamp > latestTs) {
        latestTs = cycle.timestamp;
        latestCycleId = id;
      }
    }
    if (latestCycleId) {
      this.finalizeCycle(outcome, latestCycleId);
    }
  }

  private async finalizeCycle(outcome: any, cycleId: string) {
    const cycle = this.pendingCycles.get(cycleId);
    if (!cycle || cycle.finalized) return;

    // Cancel auto-finalize timer if still pending
    const timerId = this.cycleTimers.get(cycleId);
    if (timerId !== undefined) {
      clearTimeout(timerId);
      this.cycleTimers.delete(cycleId);
    }

    cycle.finalized = true;
    this.pendingCycles.delete(cycleId);
    const { result, actionData, worldState } = cycle;
    let afterState = worldState;
    try {
      // Capture the latest physics state when the outcome is finalized. The
      // initial worldState is the observation that preceded the action.
      afterState = await this.config.captureWorldState() || worldState;
    } catch (error) {
      console.warn(`[AgentLoop (${this.config.agentId})] Post-action capture failed:`, error);
    }

    const transitionTelemetry = {
      schema_version: 1,
      observation_before: {
        heartbeat: worldState.heartbeat || this.heartbeat,
        joints: worldState.joints || {},
        proprioception: worldState.proprioception || null,
        contact_forces: worldState.contact_forces || {},
        grounded: worldState.isGrounded ?? null,
        joint_velocities: worldState.joint_velocities || {},
        root_state: worldState.root_state || null,
      },
      observation_after: {
        heartbeat: afterState.heartbeat || this.heartbeat,
        joints: afterState.joints || {},
        proprioception: afterState.proprioception || null,
        contact_forces: afterState.contact_forces || {},
        grounded: afterState.isGrounded ?? null,
        joint_velocities: afterState.joint_velocities || {},
        root_state: afterState.root_state || null,
      },
      requested_action: actionData.actions || { program_sequence: [], joint_overrides: {} },
      outcome: outcome.description || outcome || 'unknown',
    };

    const memoryEntry: MemoryEntry = {
      memory_id: actionData.memory_write.memory_id === 'auto' ? `mem_${Date.now()}` : actionData.memory_write.memory_id,
      heartbeat: worldState.heartbeat || this.heartbeat,
      day_cycle: 1,
      light_state: worldState.lightState || 'day',
      tier: actionData.memory_write.tier || 3,
      visual_description: actionData.memory_write.summary || "No summary provided",
      audio_state: JSON.stringify({
        ...(worldState.audio || {}),
        overheard_speech: worldState.overheard_speech || [],
      }),
      joint_state_summary: JSON.stringify(afterState.joints || worldState.joints || {}),
      self_questions: transitionTelemetry,
      thought: stripSpeechTags(result.thoughtTokens),
      action_taken: actionData.actions,
      outcome: outcome.description || outcome || 'unknown',
      reward_signal: outcome.reward || 0,
      goal_at_time: worldState.currentGoal || '',
      injected: !!worldState.injected_thought,
      session_id: worldState.sessionId || `session_${this.config.agentId}`,
      frame_buffer: decodeFrameBuffer(worldState.frame),
    };

    const writeOk = await this.memoryManager.write(memoryEntry, this.config.agentId);
    if (writeOk) {
      const store = useAgentStore.getState() as any;
      if (store.addMemoryForAgent) {
        store.addMemoryForAgent(this.config.agentId, {
          id: memoryEntry.memory_id,
          thought: memoryEntry.thought,
          summary: memoryEntry.visual_description,
          timestamp: Date.now(),
          reward: memoryEntry.reward_signal,
          outcome: memoryEntry.outcome,
          tier: memoryEntry.tier,
          heartbeat: memoryEntry.heartbeat,
        });
      }
      console.log(`[AgentLoop (${this.config.agentId})] Client-side memory saved successfully: ${memoryEntry.memory_id}`);
    }

    if (actionData._identityUpdate) {
      const result = await this.identityManager.applyIdentityUpdate(this.config.agentId, actionData._identityUpdate);
      if (result.ok && result.identity) {
        this.identity = result.identity;
        useIdentityStore.getState().setIdentity(this.config.agentId, result.identity);
        console.log(`[AgentLoop (${this.config.agentId})] identity updated: ${actionData._identityUpdate.field}`);
      } else {
        this.lastIdentityFeedback = { rejection: result.rejection, field: actionData._identityUpdate.field };
        console.warn(`[AgentLoop (${this.config.agentId})] identity update rejected: ${result.rejection}`);
      }
    }
  }

  private parseAndValidateAction(jsonStr: string): any {
    try {
      if (!jsonStr || !jsonStr.trim()) {
        console.warn(`[AgentLoop (${this.config.agentId})] Action JSON is empty. Using default balance pose.`);
        return {
          memory_write: { memory_id: 'auto', tier: 3, summary: 'Stand upright and observe environment' },
          actions: { program_sequence: ['stand_upright'], joint_overrides: {} },
        };
      }
      const cleanJson = jsonStr.replace(/```json/g, '').replace(/```/g, '').trim();
      const data = JSON.parse(cleanJson);

      if (!data.memory_write || typeof data.memory_write !== 'object') {
        const fallbackSummary = typeof data.memory_write === 'string' ? data.memory_write : 'No summary provided';
        data.memory_write = { memory_id: 'auto', tier: 3, summary: fallbackSummary };
      }

      if (Array.isArray(data.actions)) {
        const programs = data.actions.map((a: any) => a.program_name || a.program || a.action).filter(Boolean);
        const overrides: Record<string, any> = {};
        data.actions.forEach((a: any) => {
          if (a.joint_overrides) Object.assign(overrides, a.joint_overrides);
          if (a.joint && a.rotation) overrides[a.joint] = a.rotation;
        });
        data.actions = {
          program_sequence: programs,
          joint_overrides: overrides
        };
      }

      if (!data.actions || typeof data.actions !== 'object') {
        data.actions = { program_sequence: [], joint_overrides: {} };
      }
      if (!Array.isArray(data.actions.program_sequence)) {
        data.actions.program_sequence = [];
      }
      if (!data.actions.joint_overrides || typeof data.actions.joint_overrides !== 'object') {
        data.actions.joint_overrides = {};
      }

      const DEG_TO_RAD = Math.PI / 180;
      const normalizeRaw = (rawAction: any) => {
        if (typeof rawAction === 'number') {
          let value = rawAction;
          if (Math.abs(value) > Math.PI + 0.1) value *= DEG_TO_RAD;
          return Math.max(-Math.PI, Math.min(Math.PI, value));
        }

        if (Array.isArray(rawAction) && rawAction.length === 3) {
          return rawAction.map((v) => {
            const n = Number(v) || 0;
            return Math.abs(n) > Math.PI + 0.1 ? Math.max(-Math.PI, Math.min(Math.PI, n * DEG_TO_RAD)) : Math.max(-Math.PI, Math.min(Math.PI, n));
          }) as [number, number, number];
        }

        if (typeof rawAction === 'object' && rawAction !== null) {
          const x = Number(rawAction.x ?? rawAction.pitch ?? 0) || 0;
          const y = Number(rawAction.y ?? rawAction.yaw ?? 0) || 0;
          const z = Number(rawAction.z ?? rawAction.roll ?? 0) || 0;
          return [x, y, z].map((v) => {
            const n = Number(v) || 0;
            return Math.abs(n) > Math.PI + 0.1 ? Math.max(-Math.PI, Math.min(Math.PI, n * DEG_TO_RAD)) : Math.max(-Math.PI, Math.min(Math.PI, n));
          }) as [number, number, number];
        }

        return rawAction;
      };

      if (data.sequence && Array.isArray(data.sequence)) {
        for (const frame of data.sequence) {
          if (!frame.overrides || typeof frame.overrides !== 'object') continue;
          for (const joint in frame.overrides) {
            frame.overrides[joint] = normalizeRaw(frame.overrides[joint]);
          }
        }
      }

      if (data.actions && data.actions.joint_overrides) {
        for (const joint in data.actions.joint_overrides) {
          data.actions.joint_overrides[joint] = normalizeRaw(data.actions.joint_overrides[joint]);
        }
      }

      const gazeTarget = data.gaze_target || data.actions?.gaze_target || null;
      if (gazeTarget && typeof gazeTarget === 'object') {
        let gtYaw = gazeTarget.yaw ?? 0;
        let gtPitch = gazeTarget.pitch ?? 0;

        if (Math.abs(gtYaw) > Math.PI + 0.1) gtYaw *= DEG_TO_RAD;
        if (Math.abs(gtPitch) > Math.PI + 0.1) gtPitch *= DEG_TO_RAD;

        gtYaw = Math.max(-0.79, Math.min(0.79, gtYaw));
        gtPitch = Math.max(-0.79, Math.min(0.79, gtPitch));

        if (!data.actions.joint_overrides['mixamorighead']) {
          data.actions.joint_overrides['mixamorighead'] = [gtPitch, gtYaw, 0];
        }
      }

      if (data.identity_update && typeof data.identity_update === 'object') {
        data._identityUpdate = data.identity_update;
      }

      return data;
    } catch (e: any) {
      console.error(`[AgentLoop (${this.config.agentId})] JSON parse error — ${e.message}`);
      return null;
    }
  }
}
