/**
 * Assembles InferPayload from world state, memories, and settings.
 * Ported client-side for Phase 2.
 */

import { MemoryManager } from './memoryManager';
import { embeddingEngine } from './embeddingEngine';
import { MotorCodexService } from './motorCodexService';
import { useVideoTaskStore } from '../../store/videoTaskStore';

function degradeSpeech(text: string, lossPercentage: number): string {
  const words = text.split(' ');
  return words.map(word => {
    if (Math.random() < lossPercentage * 0.6) {
      return '[inaudible]';
    }
    return word;
  }).join(' ');
}

export class PayloadBuilder {
  private memoryManager: MemoryManager;

  constructor(memoryManager: MemoryManager) {
    this.memoryManager = memoryManager;
  }

  private heartbeatCounter: number = 0;

  private previousJoints: Record<string, any> = {};
  private static SPATIAL_ANCHORS = ['mixamorighead', 'mixamorigspine', 'mixamorighips', 'mixamorigleftupleg', 'mixamorigrightupleg'];
  private static JOINT_DELTA_THRESHOLD = 2.0;

  /**
   * Compressed tactile context: only report contacts with impulse > 1.0 N·s.
   * Drops the 80-bone "not in contact" noise.
   */
  private buildTactileContext(contactForces: Record<string, any>): string {
    const entries = Object.entries(contactForces);
    if (entries.length === 0) return 'Tactile: No active contact.';

    const active = entries
      .filter(([, data]: [string, any]) => data.contact && data.impulse_magnitude > 1.0)
      .map(([part, data]: [string, any]) => {
        const mag = data.impulse_magnitude;
        let label: string;
        if (mag > 20) label = 'strong ground support';
        else if (mag > 5) label = 'firm contact';
        else label = 'moderate force';
        const name = part.replace('capsule_body', 'body').replace(/_/g, ' ').replace('mixamorig', '');
        return `${name}: ${label} (${mag.toFixed(1)} N·s)`;
      });

    return active.length > 0
      ? `Tactile: ${active.join('; ')}.`
      : 'Tactile: No active contact.';
  }

  /**
   * Delta joints: always include spatial anchors, only include others if changed > threshold.
   * Reduces joints payload from ~80 entries to ~5-15 per tick.
   */
  private buildDeltaJoints(currentJoints: Record<string, any>): Record<string, any> {
    const delta: Record<string, any> = {};

    for (const anchor of PayloadBuilder.SPATIAL_ANCHORS) {
      if (currentJoints[anchor]) delta[anchor] = currentJoints[anchor];
    }

    for (const [name, value] of Object.entries(currentJoints)) {
      if (PayloadBuilder.SPATIAL_ANCHORS.includes(name)) continue;
      const prev = this.previousJoints[name];
      if (!prev) {
        delta[name] = value;
      } else if (JSON.stringify(prev) !== JSON.stringify(value)) {
        delta[name] = value;
      }
    }

    this.previousJoints = JSON.parse(JSON.stringify(currentJoints));
    return delta;
  }

  /**
   * Build a perception summary for spatial grounding when the visual field is uninformative.
   * Converts joint state + world state to human-readable text with real-time vestibular & balance awareness.
   */
  private buildPerceptionSummary(payload: any): string {
    const joints = payload.joints || {};

    // Head orientation → cardinal direction
    const head = joints['mixamorighead'] || {};
    const headRotation = head.rotation || [0, 0, 0, 1];
    const headY = typeof headRotation === 'object' && Array.isArray(headRotation)
      ? headRotation[1] : 0;
    // Quaternion Y component → approximate yaw in degrees
    const yawDeg = Math.round((2 * Math.asin(Math.max(-1, Math.min(1, headY || 0)))) * (180 / Math.PI));
    let facing = 'forward';
    if (yawDeg > 45) facing = 'right';
    else if (yawDeg < -45) facing = 'left';
    else if (Math.abs(yawDeg) <= 45) facing = 'forward';

    // Hip height → upright detection
    const hips = joints['mixamorighips'] || joints['capsule'] || joints['mixamorigspine'] || {};
    const hipPos = hips.position || [0, 1, 0];
    const bodyHeight = Array.isArray(hipPos) ? hipPos[1] : 1;

    // Extract root rotation from hips or capsule or spine
    const hipRot = hips.rotation || [0, 0, 0, 1];
    const qx = typeof hipRot === 'object' && Array.isArray(hipRot) ? (hipRot[0] || 0) : 0;
    const qy = typeof hipRot === 'object' && Array.isArray(hipRot) ? (hipRot[1] || 0) : 0;
    const qz = typeof hipRot === 'object' && Array.isArray(hipRot) ? (hipRot[2] || 0) : 0;
    const qw = typeof hipRot === 'object' && Array.isArray(hipRot) ? (hipRot[3] !== undefined ? hipRot[3] : 1) : 1;

    // Local UP vector transformed by root quaternion: u = q * (0,1,0) * q^-1
    const upX = 2 * (qx * qy - qw * qz);
    const upY = 1 - 2 * (qx * qx + qz * qz);
    const upZ = 2 * (qy * qz + qw * qx);

    // Total tilt angle from vertical (0° = perfectly upright, 90° = horizontal)
    const clampedUpY = Math.max(-1, Math.min(1, upY));
    const tiltDeg = Math.round(Math.acos(clampedUpY) * (180 / Math.PI) * 10) / 10;

    // Pitch: positive is leaning forward, negative is leaning backward
    const pitchDeg = Math.round(Math.atan2(upZ, Math.max(0.001, Math.abs(upY))) * (180 / Math.PI) * 10) / 10;
    // Roll: positive is leaning right, negative is leaning left
    const rollDeg = Math.round(Math.atan2(upX, Math.max(0.001, Math.abs(upY))) * (180 / Math.PI) * 10) / 10;

    // Directional classification and active maneuvering advice
    let leanDirection = 'UPRIGHT (centered)';
    let maneuverTip = 'Maintain balanced posture.';

    if (tiltDeg >= 5) {
      const isPitchFwd = pitchDeg > 3;
      const isPitchBwd = pitchDeg < -3;
      const isRollRight = rollDeg > 3;
      const isRollLeft = rollDeg < -3;

      if (isPitchFwd && isRollRight) {
        leanDirection = 'FORWARD-RIGHT';
        maneuverTip = 'Shift torso backward-left, or step forward-right with right leg to catch your balance.';
      } else if (isPitchFwd && isRollLeft) {
        leanDirection = 'FORWARD-LEFT';
        maneuverTip = 'Shift torso backward-right, or step forward-left with left leg to catch your balance.';
      } else if (isPitchBwd && isRollRight) {
        leanDirection = 'BACKWARD-RIGHT';
        maneuverTip = 'Lean torso forward and step backward-right to catch your balance.';
      } else if (isPitchBwd && isRollLeft) {
        leanDirection = 'BACKWARD-LEFT';
        maneuverTip = 'Lean torso forward and step backward-left to catch your balance.';
      } else if (isPitchFwd) {
        leanDirection = 'FORWARD';
        maneuverTip = 'Pull torso/spine backward or swing lead leg forward to plant foot and catch the forward fall.';
      } else if (isPitchBwd) {
        leanDirection = 'BACKWARD';
        maneuverTip = 'Flex spine/hips forward or step backward to arrest the backward fall.';
      } else if (isRollRight) {
        leanDirection = 'RIGHT';
        maneuverTip = 'Counter-lean left with spine or step right to widen base of support.';
      } else if (isRollLeft) {
        leanDirection = 'LEFT';
        maneuverTip = 'Counter-lean right with spine or step left to widen base of support.';
      }
    }

    // Determine posture and balance situation
    let postureLabel: string;
    let balanceState: string;
    let situationBlock: string;

    if (bodyHeight <= 0.35 || tiltDeg >= 60) {
      postureLabel = 'FALLEN / PRONE';
      balanceState = `FALLEN (${tiltDeg}° off-vertical)`;
      situationBlock = `SITUATION: You have FALLEN to the floor (hip height: ${bodyHeight.toFixed(2)}m, tilt: ${tiltDeg}°).
PRIORITY ACTION: Execute 'get_up_from_front', 'get_up_from_back', or program_sequence: ["reset_pose"] to return upright.`;
    } else if (tiltDeg >= 18 || bodyHeight < 0.7) {
      postureLabel = `CRITICAL LEAN — IMMINENT FALL (${leanDirection})`;
      balanceState = `CRITICAL TILT (${tiltDeg}° ${leanDirection})`;
      situationBlock = `CRITICAL BALANCE WARNING: You are leaning ${leanDirection} by ${tiltDeg}° (pitch: ${pitchDeg}°, roll: ${rollDeg}°).
Inability to rapidly correct this tilt will cause an IMMINENT FALL!
MANEUVER TO CATCH BALANCE: ${maneuverTip}`;
    } else if (tiltDeg >= 7) {
      postureLabel = `SLIGHT LEAN (${leanDirection})`;
      balanceState = `LEANING (${tiltDeg}° ${leanDirection})`;
      situationBlock = `BALANCE NOTICE: Currently tilted ${leanDirection} by ${tiltDeg}° (pitch: ${pitchDeg}°, roll: ${rollDeg}°).
ADVICE: ${maneuverTip}`;
    } else {
      postureLabel = 'STANDING UPRIGHT & BALANCED';
      balanceState = `BALANCED (${tiltDeg}° tilt)`;
      situationBlock = `SITUATION: Standing upright on the floor with stable balance (hip height: ${bodyHeight.toFixed(2)}m, tilt: ${tiltDeg}°). Both feet have ground support.`;
    }

    // Overheard speech context
    let overheardSection = '';
    const overheardList = payload.overheard_speech || [];
    if (overheardList.length > 0) {
      const lines = overheardList.map((entry: any) => {
        const baseLoss = entry.distance / 15.0;
        const penalty = entry.occluded ? 0.4 : 0.0;
        const lossPercentage = Math.max(0.0, Math.min(1.0, baseLoss + penalty));
        const degradedText = degradeSpeech(entry.text, lossPercentage);
        return `- From ${entry.speakerId} (distance: ${entry.distance.toFixed(1)}m, occluded: ${entry.occluded ? 'yes' : 'no'}): "${degradedText}"`;
      });
      overheardSection = `\nOVERHEARD SPEECH:\n${lines.join('\n')}\n`;
    }

    // Nearby objects (within 5m of head)
    const headPos = head.position || [0, 1.6, 0];
    const nearbyObjects = (payload.objects_in_world || [])
      .map((obj: any) => {
        const pos = obj.position || obj.mesh?.position || [0, 0, 0];
        const dx = (pos.x || pos[0] || 0) - (headPos[0] || 0);
        const dy = (pos.y || pos[1] || 0) - (headPos[1] || 0);
        const dz = (pos.z || pos[2] || 0) - (headPos[2] || 0);
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        return { type: obj.type || obj.name || 'object', dist };
      })
      .filter((o: any) => o.dist < 5)
      .sort((a: any, b: any) => a.dist - b.dist)
      .slice(0, 5);

    const objectLines = nearbyObjects.length > 0
      ? nearbyObjects.map((o: any) => `${o.type} (${o.dist.toFixed(1)}m away)`).join('\n')
      : 'None detected nearby';

    // Contact forces
    const contactForces = payload.contact_forces || {};
    const contactEntries = Object.entries(contactForces);
    let contactText = 'No active contact';
    if (contactEntries.length > 0) {
      const parts = contactEntries.map(([part, data]: [string, any]) => {
        if (!data.contact) return null;
        const mag = data.impulse_magnitude || 0;
        let label = 'touching';
        if (mag > 20) label = 'strong ground support';
        else if (mag > 5) label = 'firm contact';
        else if (mag > 1) label = 'moderate force';
        else if (mag > 0.01) label = 'light touch';
        return `${part.replace('capsule_body', 'body')} ${label} ${data.touching || ''}`;
      }).filter(Boolean);
      contactText = parts.length > 0 ? parts.join('; ') : 'No active contact';
    }

    let summary = `Body: ${facing} (${yawDeg}°) | ${postureLabel} | Tilt: ${tiltDeg}° ${leanDirection} [P:${pitchDeg}° R:${rollDeg}°] | Hip: ${bodyHeight.toFixed(2)}m | Beat: ${payload.heartbeat} | ${payload.light_state}`;
    if (situationBlock) summary += ` | ${situationBlock.split('\n')[0]}`;
    if (objectLines !== 'None detected nearby') summary += ` | Objects: ${objectLines}`;
    summary += ` | Contact: ${contactText}`;
    if (overheardSection) summary += ` | Speech: ${overheardList.length} sources`;

    if (summary.length > 300) summary = summary.substring(0, 297) + '...';
    return summary;
  }

  public getHeartbeat(): number {
    return this.heartbeatCounter;
  }

  async build(worldState: any, agentId: string, options: any): Promise<any> {
    const contextString = `${worldState.currentGoal || ''} ${(worldState.objects || []).map((o: any) => o.name).join(', ')}`;
    const embedding = await embeddingEngine.embed(contextString);

    const relevantMemories = await this.memoryManager.retrieveRelevant(embedding, agentId, 5);
    const recentWorkingMemories = await this.memoryManager.retrieveRecent(agentId, 3);

    const pendingInjection: string | null = worldState.injected_thought || null;

    // Strip data URL prefix from frame
    let rawFrame: string = worldState.frame || '';
    if (rawFrame.includes(',')) {
      rawFrame = rawFrame.split(',')[1];
    }

    const audioPcm: string = worldState.audio_pcm || worldState.audio?.pcm || '';

    this.heartbeatCounter += 1;
    const heartbeat: number = typeof worldState.heartbeat === 'number'
      ? worldState.heartbeat
      : this.heartbeatCounter;

    const contactForces: Record<string, any> = worldState.contact_forces || {};

    console.log(`[PayloadBuilder (${agentId})] heartbeat=${heartbeat}, frame_raw_len=${rawFrame.length}, audio_len=${audioPcm.length}, joints=${Object.keys(worldState.joints || {}).length}`);

    const payload: any = {
      frame: rawFrame,
      audio_pcm: audioPcm,
      joints: worldState.joints || {},
      valid_joints: Object.keys(worldState.joints || {}),
      upright_preset: worldState.uprightPreset || {},
      heartbeat,
      light_state: worldState.lightState || 'day',
      session_id: worldState.sessionId || `session_${agentId}`,
      body_type: worldState.bodyType || 'humanoid',
      current_goal: worldState.currentGoal ?? options.goal ?? null,
      current_rung: worldState.currentRung ?? 0,
      objects_in_world: worldState.objects || [],
      relevant_memories: relevantMemories.map(m => ({ ...m, summary: m.visual_description || 'No summary' })),
      recent_working_memories: recentWorkingMemories.map(m => ({ ...m, summary: m.visual_description || 'No summary' })),
      known_skills: options.masteredSkills || [],
      pending_injection: pendingInjection || worldState.injected_thought || null,
      motor_program_library: options.motorPrograms || [],
      directive_mode: options.mode || 'free_will',
      agent_id: agentId,
      contact_forces: contactForces,
      overheard_speech: worldState.overheard_speech || [],
      identity: options.identity || null,
      identity_feedback: null as string | null,
    };

    const identityFeedback = options.identityFeedback as any;
    if (identityFeedback && identityFeedback.rejection) {
      const rejection = identityFeedback.rejection;
      let reason: string;
      if (rejection === 'missing_reason') reason = 'You must provide a reason for identity changes.';
      else if (rejection === 'rate_limited') reason = 'You can only edit identity once per 5 minutes. Try again later.';
      else if (rejection === 'malformed_beliefs_op') reason = 'Beliefs must be modified incrementally using { op: "append", entry: "..." } or { op: "modify", index: N, entry: "..." }. Raw array replacement is not allowed.';
      else if (rejection === 'unknown_field') reason = 'Only name, beliefs, and traits fields are modifiable.';
      else reason = `Identity update failed: ${rejection}`;
      payload.identity_feedback = reason;
    }

    payload.tactile_context = this.buildTactileContext(contactForces);

    // Delta joints: spatial anchors + changed joints only
    payload._delta_joints = this.buildDeltaJoints(worldState.joints || {});

    // Directive text for hosted API user message (removed from system prompt for caching)
    if (options.mode === 'training') {
      payload._directive_text = `DIRECTIVE: TRAINING. Goal: ${options.goal || 'None'}.\n`;
    } else {
      payload._directive_text = 'DIRECTIVE: FREE WILL — explore, act, observe every cycle.\n';
    }

    payload.gaze_context = `You control your view by rotating your head (set mixamorighead joint overrides).
The first-person camera is attached to your head bone. It does NOT move independently.
The chase/second-person camera is a fixed spectator camera — it never follows your movement.

Your eyes can make small shifts (gaze_target yaw/pitch in radians, range -0.15 to 0.15)
but this is a subtle eye movement within the head, not turning your head.`;

    payload.perception_summary = this.buildPerceptionSummary(payload);

    const feedback = options.physicalFeedback as any[] | undefined;
    if (feedback && feedback.length > 0) {
      payload.physical_feedback = feedback.map(r =>
        `Your attempt to move ${r.joint} to ${Number(r.requested).toFixed(2)} ` +
        `radians was physically impossible — your body's limit for ` +
        `this joint is ${Number(r.limit_min).toFixed(2)} to ${Number(r.limit_max).toFixed(2)} ` +
        `radians. The joint did not move. Try a smaller adjustment.`
      ).join(' ');
    } else {
      payload.physical_feedback = null;
    }

    const useActionDictionary = options.useActionDictionary !== false;
    payload.use_action_dictionary = useActionDictionary;

    if (useActionDictionary) {
      const queryContext = `${payload.current_goal || ''} ${payload.pending_injection || ''} ${payload.directive_mode || ''}`;
      const relevantRecipes = MotorCodexService.findRelevant(queryContext, 2);
      if (relevantRecipes.length > 0) {
        payload.motor_codex_hints = MotorCodexService.formatForPrompt(relevantRecipes);
      }
    }

    const videoTaskStore = useVideoTaskStore.getState();
    if (videoTaskStore.hasActiveTask() && payload.directive_mode === 'training') {
      const currentMilestone = videoTaskStore.getCurrentMilestoneFrame();
      const activeTask = videoTaskStore.activeTask!;
      let rawTargetFrame: string | null = null;
      if (currentMilestone?.dataUrl) {
        rawTargetFrame = currentMilestone.dataUrl.includes(',')
          ? currentMilestone.dataUrl.split(',')[1]
          : currentMilestone.dataUrl;
      }

      payload.video_task = {
        active: true,
        name: activeTask.name,
        ingestion_mode: videoTaskStore.ingestionMode,
        milestone_index: videoTaskStore.currentMilestoneIndex + 1,
        total_milestones: activeTask.keyframeIndices.length,
        timestamp: currentMilestone?.timestamp ?? 0,
        label: currentMilestone?.label ?? `Milestone ${videoTaskStore.currentMilestoneIndex + 1}`,
        target_frame: rawTargetFrame,
      };
    }

    return payload;
  }
}
