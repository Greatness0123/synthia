import type { SupabaseClient } from '@supabase/supabase-js';
import { getSupabaseClient } from '../../utils/supabaseClient';

export interface AgentIdentity {
  agent_id: string;
  name: string;
  beliefs: any[];
  traits: Record<string, any>;
  window_started_at: string | null;
  edit_count_window: number;
  updated_at: string;
}

export interface IdentityUpdate {
  field?: 'name' | 'beliefs' | 'traits';
  new_value?: any;
  name?: string;
  beliefs?: any;
  traits?: Record<string, any>;
  reason?: string;
}

export interface IdentityApplyResult {
  ok: boolean;
  identity?: AgentIdentity;
  rejection?: string;
  error?: string;
}

const RATE_LIMIT_WINDOW_MS = 300_000; // 5 minutes
const MAX_EDITS_PER_WINDOW = 1;

export const DEFAULT_IDENTITY_TEMPLATE: Omit<AgentIdentity, 'agent_id' | 'window_started_at' | 'edit_count_window' | 'updated_at'> = {
  name: 'Agent',
  beliefs: [
    'I am a precision motor controller commanding a 3D humanoid body.',
    'Every joint angle I output must be calculated. Wrong angles cause falls.',
    'I check my vestibular balance before every motor decision.',
    'I observe, decide, act, verify — one cycle at a time.',
  ],
  traits: {
    precision: 0.9,
    caution: 0.7,
    curiosity: 0.5,
  },
};

export class IdentityManager {
  private supabase: SupabaseClient | null = null;
  private mockStore: Map<string, AgentIdentity> = new Map();
  private mockLog: Array<{ agent_id: string; field: string; old_value: any; new_value: any; reason: string; created_at: string }> = [];

  constructor(supabaseUrl: string, supabaseKey: string) {
    this.supabase = getSupabaseClient(supabaseUrl, supabaseKey);
    if (this.supabase) {
      console.log('[IdentityManager] Supabase client created (shared singleton)');
    } else {
      console.warn('[IdentityManager] Supabase not configured — using in-memory mock store');
    }
  }

  async ensureIdentity(agentId: string): Promise<AgentIdentity> {
    if (!this.supabase) {
      const existing = this.mockStore.get(agentId);
      if (existing) return existing;
      const defaults: AgentIdentity = {
        agent_id: agentId,
      name: agentId,
      beliefs: [...DEFAULT_IDENTITY_TEMPLATE.beliefs],
      traits: { ...DEFAULT_IDENTITY_TEMPLATE.traits },
      window_started_at: null,
      edit_count_window: 0,
      updated_at: new Date().toISOString(),
    };
    this.mockStore.set(agentId, defaults);
    return defaults;
    }

    try {
      const { data, error } = await this.supabase
        .from('agent_identity')
        .select('*')
        .eq('agent_id', agentId)
        .single();

      if (data && !error) return data as AgentIdentity;
    } catch (e) {
      console.warn(`[IdentityManager] ensureIdentity select error for ${agentId}:`, e);
    }

    const defaults: AgentIdentity = {
      agent_id: agentId,
      name: agentId,
      beliefs: [...DEFAULT_IDENTITY_TEMPLATE.beliefs],
      traits: { ...DEFAULT_IDENTITY_TEMPLATE.traits },
      window_started_at: null,
      edit_count_window: 0,
      updated_at: new Date().toISOString(),
    };

    try {
      const { error: upsertError } = await this.supabase
        .from('agent_identity')
        .upsert(defaults, { onConflict: 'agent_id' });

      if (upsertError) {
        console.error(`[IdentityManager] ensureIdentity upsert failed for ${agentId}:`, upsertError.message);
      }
    } catch (e) {
      console.warn(`[IdentityManager] ensureIdentity upsert caught error for ${agentId}:`, e);
    }

    return defaults;
  }

  async applyIdentityUpdate(agentId: string, update: IdentityUpdate, isManualAdmin: boolean = false): Promise<IdentityApplyResult> {
    if (!update.reason || update.reason.trim().length === 0) {
      return { ok: false, rejection: 'missing_reason', error: 'Reason is required for identity changes' };
    }

    const identity = await this.ensureIdentity(agentId);
    const now = new Date();
    const nowISO = now.toISOString();
    const windowStart = identity.window_started_at ? new Date(identity.window_started_at).getTime() : 0;
    const withinWindow = identity.window_started_at && (now.getTime() - windowStart) < RATE_LIMIT_WINDOW_MS;

    if (!isManualAdmin && withinWindow && identity.edit_count_window >= MAX_EDITS_PER_WINDOW) {
      return { ok: false, rejection: 'rate_limited', error: 'Identity edit rate limit reached (1 edit per 5 minutes)', identity };
    }

    // Handle compound updates (e.g. from AgentSettingsModal)
    const patch: Partial<AgentIdentity> = {};
    const logEntries: Array<{ field: string; old_value: any; new_value: any }> = [];

    if (update.name !== undefined && update.name !== identity.name) {
      patch.name = update.name;
      logEntries.push({ field: 'name', old_value: identity.name, new_value: update.name });
    }

    if (update.traits !== undefined && typeof update.traits === 'object') {
      patch.traits = update.traits;
      logEntries.push({ field: 'traits', old_value: identity.traits, new_value: update.traits });
    }

    if (update.beliefs !== undefined) {
      if (Array.isArray(update.beliefs)) {
        // Direct array replacement
        patch.beliefs = update.beliefs;
        logEntries.push({ field: 'beliefs', old_value: identity.beliefs, new_value: update.beliefs });
      } else {
        const enforced = this.enforceBeliefsOp(update.beliefs);
        if (!enforced.ok) {
          return { ok: false, rejection: enforced.reason, error: enforced.reason, identity };
        }
        const updatedBeliefs = this.applyBeliefsUpdate(identity.beliefs, enforced.value);
        patch.beliefs = updatedBeliefs;
        logEntries.push({ field: 'beliefs', old_value: identity.beliefs, new_value: enforced.value });
      }
    }

    // Handle single field update schema { field, new_value }
    if (update.field) {
      if (update.field === 'name') {
        patch.name = update.new_value;
        logEntries.push({ field: 'name', old_value: identity.name, new_value: update.new_value });
      } else if (update.field === 'traits') {
        patch.traits = update.new_value;
        logEntries.push({ field: 'traits', old_value: identity.traits, new_value: update.new_value });
      } else if (update.field === 'beliefs') {
        if (Array.isArray(update.new_value)) {
          patch.beliefs = update.new_value;
          logEntries.push({ field: 'beliefs', old_value: identity.beliefs, new_value: update.new_value });
        } else {
          const enforced = this.enforceBeliefsOp(update.new_value);
          if (!enforced.ok) {
            return { ok: false, rejection: enforced.reason, error: enforced.reason, identity };
          }
          const updatedBeliefs = this.applyBeliefsUpdate(identity.beliefs, enforced.value);
          patch.beliefs = updatedBeliefs;
          logEntries.push({ field: 'beliefs', old_value: identity.beliefs, new_value: enforced.value });
        }
      } else {
        return { ok: false, rejection: 'unknown_field', error: 'Unknown field for identity update', identity };
      }
    }

    if (Object.keys(patch).length === 0) {
      return { ok: true, identity };
    }

    const newWindowStarted = withinWindow ? identity.window_started_at : nowISO;
    const newEditCount = withinWindow ? identity.edit_count_window + 1 : 1;

    const updatedIdentity: AgentIdentity = {
      ...identity,
      ...patch,
      window_started_at: newWindowStarted,
      edit_count_window: newEditCount,
      updated_at: nowISO,
    };

    if (!this.supabase) {
      this.mockStore.set(agentId, updatedIdentity);
      for (const entry of logEntries) {
        this.mockLog.push({
          agent_id: agentId,
          field: entry.field,
          old_value: entry.old_value,
          new_value: entry.new_value,
          reason: update.reason!,
          created_at: nowISO,
        });
      }
      return { ok: true, identity: updatedIdentity };
    }

    try {
      const { error: updateError } = await this.supabase
        .from('agent_identity')
        .update({
          ...patch,
          window_started_at: newWindowStarted,
          edit_count_window: newEditCount,
          updated_at: nowISO,
        })
        .eq('agent_id', agentId);

      if (updateError) {
        console.error(`[IdentityManager] update failed for ${agentId}:`, updateError.message);
        return { ok: false, rejection: 'update_failed', error: updateError.message, identity };
      }

      for (const entry of logEntries) {
        await this.supabase
          .from('agent_identity_log')
          .insert({
            agent_id: agentId,
            field: entry.field,
            old_value: entry.old_value,
            new_value: entry.new_value,
            reason: update.reason!,
            created_at: nowISO,
          });
      }

      return { ok: true, identity: updatedIdentity };
    } catch (err: any) {
      console.error(`[IdentityManager] applyIdentityUpdate error:`, err);
      return { ok: false, rejection: 'update_failed', error: err?.message || 'Database update failed', identity };
    }
  }

  private enforceBeliefsOp(value: any): { ok: boolean; value?: any; reason?: string } {
    if (!value || typeof value !== 'object') {
      return { ok: false, reason: 'malformed_beliefs_op' };
    }

    if (Array.isArray(value)) {
      return { ok: true, value: { op: 'replace', entries: value } };
    }

    const { op } = value;

    if (op === 'append') {
      const entry = value.entry ?? value.value;
      if (typeof entry !== 'string') {
        return { ok: false, reason: 'malformed_beliefs_op' };
      }
      return { ok: true, value: { op: 'append', entry } };
    }

    if (op === 'modify') {
      const entry = value.entry ?? value.value;
      if (typeof entry !== 'string') {
        return { ok: false, reason: 'malformed_beliefs_op' };
      }
      if (typeof value.index !== 'number' || value.index < 0) {
        return { ok: false, reason: 'malformed_beliefs_op' };
      }
      return { ok: true, value: { op: 'modify', index: value.index, entry } };
    }

    if (op === 'replace') {
      const entries = Array.isArray(value.entries) ? value.entries : Array.isArray(value.value) ? value.value : null;
      if (!entries) return { ok: false, reason: 'malformed_beliefs_op' };
      return { ok: true, value: { op: 'replace', entries } };
    }

    return { ok: false, reason: 'malformed_beliefs_op' };
  }

  private applyBeliefsUpdate(beliefs: any[], updateOp: any): any[] {
    if (!updateOp) return beliefs;
    if (updateOp.op === 'replace') {
      return [...updateOp.entries];
    }
    const copy = [...beliefs];
    if (updateOp.op === 'append') {
      copy.push(updateOp.entry);
    } else if (updateOp.op === 'modify') {
      if (updateOp.index < copy.length) {
        copy[updateOp.index] = updateOp.entry;
      } else {
        copy.push(updateOp.entry);
      }
    }
    return copy;
  }

  getMockStore(): Map<string, AgentIdentity> {
    return this.mockStore;
  }

  getMockLog(): Array<{ agent_id: string; field: string; old_value: any; new_value: any; reason: string; created_at: string }> {
    return this.mockLog;
  }
}

export const createIdentityManager = (supabaseUrl: string, supabaseKey: string) =>
  new IdentityManager(supabaseUrl, supabaseKey);
