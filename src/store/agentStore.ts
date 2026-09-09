/**
 * Zustand store for agent-specific state supporting multi-agent record map
 * and active agent selection with backward-compatible flat field mirroring.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Thought, Memory, AgentStatus, DirectiveMode } from '../types/agent';
import { speakAgentThought, stripSpeechTags } from '../utils/speech';

export interface SingleAgentState {
  agentId: string;
  thoughts: Thought[];
  memories: Memory[];
  skills: string[];
  currentRung: number;
  currentGoal: string | null;
  directiveMode: DirectiveMode;
  heartbeat: number;
  status: AgentStatus;
  pendingInjection: string | null;
  currentThought: string;
  rehydrationSummary: string;
  injectionQueue: string[];
  injectionQueueCount: number;
  bodyMode: 'rigid' | 'ragdoll';
  useMultiBodyPD: boolean;
  useActionDictionary: boolean;
}

const createDefaultAgent = (agentId: string): SingleAgentState => ({
  agentId,
  thoughts: [],
  memories: [],
  skills: [],
  currentRung: 0,
  currentGoal: null,
  directiveMode: 'free_will',
  heartbeat: 0,
  status: 'idle',
  pendingInjection: null,
  currentThought: '',
  rehydrationSummary: '',
  injectionQueue: [],
  injectionQueueCount: 0,
  bodyMode: 'rigid',
  useMultiBodyPD: true,
  useActionDictionary: true,
});

interface AgentStoreState {
  activeAgentId: string;
  agents: Record<string, SingleAgentState>;

  // Flat fields mirrored from the activeAgentId for backward compatibility
  thoughts: Thought[];
  memories: Memory[];
  skills: string[];
  currentRung: number;
  currentGoal: string | null;
  directiveMode: DirectiveMode;
  heartbeat: number;
  lightState: 'day' | 'night';
  status: AgentStatus;
  pendingInjection: string | null;
  currentThought: string;
  rehydrationSummary: string;
  hasRehydrated: boolean;
  masteredSkills: string[];
  injectionQueue: string[];
  injectionQueueCount: number;
  bodyMode: 'rigid' | 'ragdoll';
  useMultiBodyPD: boolean;
  useActionDictionary: boolean;

  // ── Per-agent actions ─────────────────────────────
  setActiveAgentId: (id: string) => void;
  addAgent: (id: string) => void;
  addThoughtForAgent: (id: string, thought: Thought) => void;
  addMemoryForAgent: (id: string, memory: Memory) => void;
  setStatusForAgent: (id: string, status: AgentStatus) => void;
  setCurrentThoughtForAgent: (id: string, text: string) => void;
  appendThoughtTokenForAgent: (id: string, token: string) => void;
  setPendingInjectionForAgent: (id: string, text: string | null) => void;
  setDirectiveModeForAgent: (id: string, mode: DirectiveMode) => void;
  setCurrentGoalForAgent: (id: string, goal: string | null) => void;
  incrementHeartbeatForAgent: (id: string) => void;
  setHeartbeatForAgent: (id: string, hb: number) => void;
  setRungForAgent: (id: string, rung: number) => void;
  addSkillForAgent: (id: string, skill: string) => void;
  setRehydrationSummaryForAgent: (id: string, text: string) => void;
  appendRehydrationTokenForAgent: (id: string, token: string) => void;
  setInjectionQueueForAgent: (id: string, queue: string[]) => void;
  setInjectionQueueCountForAgent: (id: string, count: number) => void;
  incrementInjectionQueueCountForAgent: (id: string) => void;
  decrementInjectionQueueCountForAgent: (id: string) => void;
  setBodyModeForAgent: (id: string, mode: 'rigid' | 'ragdoll') => void;
  setUseMultiBodyPDForAgent: (id: string, enable: boolean) => void;
  setUseActionDictionaryForAgent: (id: string, enable: boolean) => void;
  clearThoughtsForAgent: (id: string) => void;
  clearMemoriesForAgent: (id: string) => void;

  // ── Mirrored Actions on the currently active agent ─
  addThought: (thought: Thought) => void;
  addMemory: (memory: Memory) => void;
  setDirectiveMode: (mode: DirectiveMode) => void;
  setCurrentGoal: (goal: string | null) => void;
  setPendingInjection: (text: string | null) => void;
  setStatus: (status: AgentStatus) => void;
  setCurrentThought: (text: string) => void;
  appendThoughtToken: (token: string) => void;
  setRehydrationSummary: (text: string) => void;
  appendRehydrationToken: (token: string) => void;
  setHasRehydrated: (val: boolean) => void;
  addMasteredSkill: (skill: string) => void;
  setInjectionQueue: (queue: string[]) => void;
  setInjectionQueueCount: (count: number) => void;
  setUseActionDictionary: (enable: boolean) => void;
  incrementInjectionQueueCount: () => void;
  decrementInjectionQueueCount: () => void;
  setRung: (rung: number) => void;
  incrementHeartbeat: () => void;
  setHeartbeat: (hb: number) => void;
  setBodyMode: (mode: 'rigid' | 'ragdoll') => void;
  setUseMultiBodyPD: (enable: boolean) => void;
  clearThoughts: () => void;
  clearMemories: () => void;
}

const initialAgentId = 'agent_0';
const initialAgents = {
  [initialAgentId]: createDefaultAgent(initialAgentId),
};

export const useAgentStore = create<AgentStoreState>()(
  persist(
    (set, get) => {
      // Helper to update flat mirrored fields in the state
      const mirrorActiveAgentFields = (activeId: string, agentsMap: Record<string, SingleAgentState>) => {
        const active = agentsMap[activeId] || createDefaultAgent(activeId);
        return {
          thoughts: active.thoughts || [],
          memories: active.memories || [],
          skills: active.skills || [],
          masteredSkills: active.skills || [],
          currentRung: active.currentRung ?? 0,
          currentGoal: active.currentGoal ?? null,
          directiveMode: active.directiveMode ?? 'free_will',
          heartbeat: active.heartbeat ?? 0,
          status: active.status ?? 'idle',
          pendingInjection: active.pendingInjection ?? null,
          currentThought: active.currentThought ?? '',
          injectionQueue: active.injectionQueue || [],
          injectionQueueCount: active.injectionQueueCount ?? 0,
          bodyMode: active.bodyMode ?? 'rigid',
          useMultiBodyPD: active.useMultiBodyPD ?? true,
        };
      };

      /** Update one agent record and mirror its fields only when it is the active agent. */
      const updateAgent = (
        state: AgentStoreState,
        id: string,
        patch: Partial<SingleAgentState>
      ) => {
        const agent = state.agents[id] || createDefaultAgent(id);
        const updatedAgent = { ...agent, ...patch };
        const newAgents = { ...state.agents, [id]: updatedAgent };
        const mirrors = id === state.activeAgentId ? mirrorActiveAgentFields(id, newAgents) : {};
        return { agents: newAgents, ...mirrors };
      };

      return {
        activeAgentId: initialAgentId,
        agents: initialAgents,

        // Initial mirror values
        ...createDefaultAgent(initialAgentId),
        bodyMode: 'rigid',
        useMultiBodyPD: true,
        lightState: 'day',
        rehydrationSummary: '',
        hasRehydrated: false,
        masteredSkills: [],
        injectionQueue: [],
        injectionQueueCount: 0,

        setActiveAgentId: (activeAgentId) => set((state) => {
          const mirrors = mirrorActiveAgentFields(activeAgentId, state.agents);
          return { activeAgentId, ...mirrors };
        }),

        addAgent: (id) => set((state) => {
          if (state.agents[id]) return {};
          const newAgents = {
            ...state.agents,
            [id]: createDefaultAgent(id),
          };
          return { agents: newAgents };
        }),

        addThoughtForAgent: (id, thought) => {
          // Trigger TTS speech synthesis as a side-effect. Only <speak> tagged
          // content is spoken/broadcast — the full thought stays silent.
          speakAgentThought(id, thought.text).catch((err) =>
            console.error(`Error in speakAgentThought for ${id}:`, err)
          );
          return set((state) => {
            const currentList = state.agents[id]?.thoughts || [];
            // Preserve up to 300 thoughts per agent in history
            const updatedThoughts = [
              ...currentList,
              { ...thought, text: stripSpeechTags(thought.text) }
            ].slice(-300);

            return updateAgent(state, id, {
              thoughts: updatedThoughts,
            });
          });
        },

        clearThoughtsForAgent: (id) => set((state) => updateAgent(state, id, { thoughts: [] })),

        clearMemoriesForAgent: (id) => set((state) => updateAgent(state, id, { memories: [] })),

        addMemoryForAgent: (id, memory) => set((state) => updateAgent(state, id, {
          memories: [...(state.agents[id]?.memories || []), memory],
        })),

        setStatusForAgent: (id, status) => set((state) => updateAgent(state, id, { status })),

        setCurrentThoughtForAgent: (id, currentThought) => set((state) => updateAgent(state, id, { currentThought })),

        appendThoughtTokenForAgent: (id, token) => set((state) => {
          const agent = state.agents[id] || createDefaultAgent(id);
          return updateAgent(state, id, { currentThought: (agent.currentThought || '') + token });
        }),

        setPendingInjectionForAgent: (id, text) => set((state) => updateAgent(state, id, { pendingInjection: text })),

        setDirectiveModeForAgent: (id, mode) => set((state) => updateAgent(state, id, { directiveMode: mode })),

        setCurrentGoalForAgent: (id, goal) => set((state) => updateAgent(state, id, { currentGoal: goal })),

        incrementHeartbeatForAgent: (id) => set((state) => {
          const agent = state.agents[id] || createDefaultAgent(id);
          return updateAgent(state, id, { heartbeat: agent.heartbeat + 1 });
        }),

        setHeartbeatForAgent: (id, heartbeat) => set((state) => updateAgent(state, id, { heartbeat })),

        setRungForAgent: (id, currentRung) => set((state) => updateAgent(state, id, { currentRung })),

        addSkillForAgent: (id, skill) => set((state) => {
          const agent = state.agents[id] || createDefaultAgent(id);
          const skills = [...(agent.skills || []), skill];
          return updateAgent(state, id, { skills });
        }),

        setRehydrationSummaryForAgent: (id, rehydrationSummary) => set((state) => updateAgent(state, id, { rehydrationSummary })),

        appendRehydrationTokenForAgent: (id, token) => set((state) => {
          const agent = state.agents[id] || createDefaultAgent(id);
          return updateAgent(state, id, { rehydrationSummary: agent.rehydrationSummary + token });
        }),

        setInjectionQueueForAgent: (id, queue) => set((state) => updateAgent(state, id, {
          injectionQueue: queue || [],
          injectionQueueCount: (queue || []).length,
        })),

        setInjectionQueueCountForAgent: (id, injectionQueueCount) => set((state) => updateAgent(state, id, { injectionQueueCount })),

        incrementInjectionQueueCountForAgent: (id) => set((state) => {
          const agent = state.agents[id] || createDefaultAgent(id);
          return updateAgent(state, id, { injectionQueueCount: agent.injectionQueueCount + 1 });
        }),

        decrementInjectionQueueCountForAgent: (id) => set((state) => {
          const agent = state.agents[id] || createDefaultAgent(id);
          return updateAgent(state, id, { injectionQueueCount: Math.max(0, agent.injectionQueueCount - 1) });
        }),

        setBodyModeForAgent: (id, bodyMode) => set((state) => updateAgent(state, id, { bodyMode })),

        setUseMultiBodyPDForAgent: (id, useMultiBodyPD) => set((state) => updateAgent(state, id, { useMultiBodyPD })),

        // ── Mirrored Legacy Actions targeting the active agent ──
        addThought: (thought) => {
          get().addThoughtForAgent(get().activeAgentId, thought);
        },
        clearThoughts: () => {
          get().clearThoughtsForAgent(get().activeAgentId);
        },
        clearMemories: () => {
          get().clearMemoriesForAgent(get().activeAgentId);
        },
        addMemory: (memory) => {
          get().addMemoryForAgent(get().activeAgentId, memory);
        },
        setDirectiveMode: (mode) => {
          get().setDirectiveModeForAgent(get().activeAgentId, mode);
        },
        setCurrentGoal: (goal) => {
          get().setCurrentGoalForAgent(get().activeAgentId, goal);
        },
        setPendingInjection: (text) => {
          get().setPendingInjectionForAgent(get().activeAgentId, text);
        },
        setStatus: (status) => {
          get().setStatusForAgent(get().activeAgentId, status);
        },
        setCurrentThought: (text) => {
          get().setCurrentThoughtForAgent(get().activeAgentId, text);
        },
        appendThoughtToken: (token) => {
          get().appendThoughtTokenForAgent(get().activeAgentId, token);
        },

        setRehydrationSummary: (rehydrationSummary) => set({ rehydrationSummary }),
        appendRehydrationToken: (token) => set((state) => ({ rehydrationSummary: state.rehydrationSummary + token })),
        setHasRehydrated: (hasRehydrated) => set({ hasRehydrated }),
        addMasteredSkill: (skill) => {
          get().addSkillForAgent(get().activeAgentId, skill);
        },
        setInjectionQueue: (queue) => {
          get().setInjectionQueueForAgent(get().activeAgentId, queue);
        },
        setInjectionQueueCount: (count) => {
          get().setInjectionQueueCountForAgent(get().activeAgentId, count);
        },
        incrementInjectionQueueCount: () => {
          get().incrementInjectionQueueCountForAgent(get().activeAgentId);
        },
        decrementInjectionQueueCount: () => {
          get().decrementInjectionQueueCountForAgent(get().activeAgentId);
        },
        setRung: (currentRung) => {
          get().setRungForAgent(get().activeAgentId, currentRung);
        },
        incrementHeartbeat: () => {
          get().incrementHeartbeatForAgent(get().activeAgentId);
        },
        setHeartbeat: (heartbeat) => {
          get().setHeartbeatForAgent(get().activeAgentId, heartbeat);
        },
        setBodyMode: (mode) => {
          get().setBodyModeForAgent(get().activeAgentId, mode);
        },
        setUseMultiBodyPD: (enable) => {
          get().setUseMultiBodyPDForAgent(get().activeAgentId, enable);
        },
        setUseActionDictionaryForAgent: (id, useActionDictionary) => set((state) => updateAgent(state, id, { useActionDictionary })),
        setUseActionDictionary: (enable) => {
          get().setUseActionDictionaryForAgent(get().activeAgentId, enable);
        },
      };
    },
    {
      name: 'synthia_agent_store',
      partialize: (state) => ({
        activeAgentId: state.activeAgentId,
        agents: state.agents,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return;
        // Clear thoughts from every agent so they don't survive a reload
        if (state.agents) {
          Object.values(state.agents).forEach((agent: any) => {
            agent.thoughts = [];
            agent.currentThought = '';
          });
        }
        const active = state.agents[state.activeAgentId] || createDefaultAgent(state.activeAgentId);
        state.thoughts = [];
        state.currentThought = '';
        state.memories = active.memories || [];
        state.skills = active.skills || [];
        state.masteredSkills = active.skills || [];
        state.currentRung = active.currentRung ?? 0;
        state.currentGoal = active.currentGoal ?? null;
        state.directiveMode = active.directiveMode ?? 'free_will';
        state.heartbeat = active.heartbeat ?? 0;
        state.status = 'idle';
      },
    }
  )
);
