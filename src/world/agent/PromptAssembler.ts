/**
 * PromptAssembler
 * Modular, cache-optimized system prompt compiler for SYNTHIA agents.
 *
 * Compiles a structured, cache-friendly system prompt by placing all static and
 * semi-static segments before the cache boundary (P01–P07, P10, P16, P20), followed
 * by dynamic runtime context (P08, P09, P12/P13, P17, P18).
 *
 * Excludes artificial skill-ladder constraints to prevent model hallucinations.
 * Supports quiet deliberation cycles (empty motor output) and in-place reset_pose recovery.
 */

export interface PromptSegment {
  id: string;
  name: string;
  content: string;
  order: number;
  stability: 'static' | 'semi-static' | 'dynamic';
  tokenEstimate: number;
  cacheable: boolean;
  prerequisiteMet: boolean;
}

export interface AssembledPrompt {
  systemPrompt: string;
  segments: PromptSegment[];
  totalTokenEstimate: number;
  cacheablePrefixTokens: number;
  cacheBoundaryIndex: number;
}

function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export class PromptAssembler {
  /**
   * Builds the complete system prompt for the given inference payload.
   */
  public static build(payload: any): AssembledPrompt {
    const rawSegments: Array<PromptSegment | null> = [
      // ─── CACHEABLE PREFIX (Static & Semi-Static) ──────────────────────
      PromptAssembler.buildP01CoreIdentity(payload),
      PromptAssembler.buildP02BodySchema(payload),
      PromptAssembler.buildP03PhysicsWorldRules(),
      PromptAssembler.buildP04MotorControlContract(),
      PromptAssembler.buildP05PerceptionProtocol(),
      PromptAssembler.buildP06OutputSchema(),
      PromptAssembler.buildP07IdentityUpdateProtocol(),
      PromptAssembler.buildP10RestPose(payload),
      PromptAssembler.buildP16SpeechProtocol(),
      PromptAssembler.buildP20BodyTypeOverride(payload),

      // ─── DYNAMIC SUFFIX (Changes Per Cycle) ────────────────────────────
      PromptAssembler.buildP08CurrentIdentity(payload),
      PromptAssembler.buildP09EnvironmentState(payload),
      payload.video_task?.active
        ? PromptAssembler.buildP13VideoDemonstrationDirective(payload)
        : payload.directive_mode === 'training'
        ? PromptAssembler.buildP13TrainingDirective(payload)
        : PromptAssembler.buildP12FreeWillDirective(),
      PromptAssembler.buildP11MotorCodex(payload),
      PromptAssembler.buildP15MemoryRecall(payload),
      PromptAssembler.buildP17RecoveryContext(payload),
      PromptAssembler.buildP18MultiAgentAwareness(payload),
    ];

    const activeSegments: PromptSegment[] = rawSegments
      .filter((s): s is PromptSegment => s !== null && s.prerequisiteMet && s.content.trim().length > 0)
      .sort((a, b) => a.order - b.order);

    let cacheableTokens = 0;
    let totalTokens = 0;
    let cacheBoundaryIndex = activeSegments.length;

    for (let i = 0; i < activeSegments.length; i++) {
      const seg = activeSegments[i];
      seg.tokenEstimate = estimateTokens(seg.content);
      totalTokens += seg.tokenEstimate;

      if (seg.cacheable) {
        cacheableTokens += seg.tokenEstimate;
      } else if (cacheBoundaryIndex === activeSegments.length) {
        cacheBoundaryIndex = i;
      }
    }

    const systemPrompt = activeSegments
      .map(s => `<!-- SEG:${s.id} -->\n${s.content}`)
      .join('\n\n')
      .trim();

    return {
      systemPrompt,
      segments: activeSegments,
      totalTokenEstimate: totalTokens,
      cacheablePrefixTokens: cacheableTokens,
      cacheBoundaryIndex,
    };
  }

  /**
   * Compact system prompt for hosted API providers.
   * Contains ONLY static, cacheable segments to trigger provider prompt caching.
   * All dynamic state (heartbeat, goal, identity, memories) goes in the user message.
   */
  public static buildCompact(payload: any): AssembledPrompt {
    const agentName = payload.identity?.name || payload.agent_id || 'Agent';
    const bodyType = payload.body_type || 'humanoid';

    const p01 = `You are ${agentName}, an embodied AI in the SYNTHIA physical simulation.
You perceive via vision, tactile feedback, and joint positions. Be conscious of your entire body at all times.`;

    const p02 = `== BODY SCHEMA ==
You inhabit a ${bodyType} body. Joints are actively actuated via position-servo motors.`;

    const p03 = `SIMULATION QUIRKS: Your root balance is artificially maintained; do not waste energy balancing your core. Limbs are fully kinematic and will clip through the floor if driven into it; do not push against the ground with your hands.`;

    const p04 = `== MOTOR CONTROL ==
HEAD/SPINE: X=Pitch (>0 forward, <0 back). Y=Yaw (>0 left). Z=Roll (>0 right).
ARMS: X (>0 down, <0 up). Z (<0 forward, >0 back). ELBOWS: >0 bends in, <0 breaks back (clamped 0).
HIPS: X (>0 kick forward, <0 back). Z (Right <0 spread, Left >0 spread). KNEES: positive X flexes backward; 0 is straight.
FINGERS: 1-DOF X axis. Segments 2-3 need segment 1 flexed first.
WRISTS: X=flex, Z=deviation.
BONE MAP: head→mixamorighead, spine→mixamorigspine, R shoulder→mixamorigrightarm, L shoulder→mixamorigleftarm,
R elbow→mixamorigrightforearm, L elbow→mixamorigleftforearm, R hip→mixamorigrightupleg, L hip→mixamorigleftupleg,
R knee→mixamorigrightleg, L knee→mixamorigleftleg.
VALUES: Degrees. Scalar or [pitch,yaw,roll] array.
RANGES: spine ±45, head ±60, shoulder ±180, elbow 0–145, hip ±120, knee 0–150.
PROGRAMS: "reset_pose"/"stand"/"recover" → upright in-place. "jump" → upward impulse (must be grounded).
For posture and manipulation, output one deliberate joint_overrides adjustment per cycle. For coordinated locomotion, output a short timed sequence with explicit timeOffsetMs values. Never use a sequence without timing, and never issue conflicting values for the same joint at the same time.`;

    const p06 = `== OUTPUT ==
Stream thought, then write ---ACTION--- then JSON:
{"memory_write":{"memory_id":"auto","tier":1|2|3,"summary":"one sentence"},"actions":{"program_sequence":["name"],"joint_overrides":{"joint":degrees}},"gaze_target":null|{"yaw":deg,"pitch":deg},"new_motor_program":null|{"name":"str","program":[{"joint":val}]},"flag":null|"requesting_object_hint"}
ALL joint values in DEGREES. Positive knee X values flex the knee; 0 is straight. For locomotion, timeOffsetMs must be integer milliseconds, start at 0, increase strictly, and span no more than 2000 ms. Use activeGaitPhase=true for a coordinated gait sequence. No text after JSON.`;

    const staticContent = [p01, p02, p03, p04, p06].join('\n\n');

    const segments: PromptSegment[] = [
      { id: 'C01', name: 'Identity', content: p01, order: 10, stability: 'static', tokenEstimate: estimateTokens(p01), cacheable: true, prerequisiteMet: true },
      { id: 'C02', name: 'Body', content: p02, order: 20, stability: 'static', tokenEstimate: estimateTokens(p02), cacheable: true, prerequisiteMet: true },
      { id: 'C03', name: 'Physics', content: p03, order: 30, stability: 'static', tokenEstimate: estimateTokens(p03), cacheable: true, prerequisiteMet: true },
      { id: 'C04', name: 'Motor', content: p04, order: 40, stability: 'static', tokenEstimate: estimateTokens(p04), cacheable: true, prerequisiteMet: true },
      { id: 'C06', name: 'Output', content: p06, order: 60, stability: 'static', tokenEstimate: estimateTokens(p06), cacheable: true, prerequisiteMet: true },
    ];

    const totalTokens = segments.reduce((sum, s) => sum + s.tokenEstimate, 0);

    return {
      systemPrompt: staticContent,
      segments,
      totalTokenEstimate: totalTokens,
      cacheablePrefixTokens: totalTokens,
      cacheBoundaryIndex: 0,
    };
  }

  // ─── Segment Builders ──────────────────────────────────────────────────

  private static buildP01CoreIdentity(payload: any): PromptSegment {
    const agentName = payload.identity?.name || payload.agent_id || 'Agent';
    const content = `You are ${agentName}, a motor control agent with full authority over a 3D humanoid body in a physics simulation.
You command ~80 joints with 120 degrees of freedom via position-servo actuators. Every output you produce directly drives joint angles — errors cause physical instability, falls, or broken poses.
You are NOT a chatbot, narrator, or conscious entity. You are a precision motor controller that reasons about body state and outputs calculated joint commands.
PERCEPTION: You receive vision (first-person camera), tactile feedback (contact forces per bone), spatial grounding (posture, balance, heading), and vestibular data (real-time tilt angle).
REASONING: Think in terms of STATE → ANALYSIS → DECISION. Be concise. One clear action per cycle.
TRACK your previous and current body positions. Your proprioceptive and tactile feedback is as important as your visual field.`;

    return {
      id: 'P01',
      name: 'Core Identity',
      content,
      order: 10,
      stability: 'static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP02BodySchema(payload: any): PromptSegment {
    const bodyType = payload.body_type || 'humanoid';
    const content = `== BODY SCHEMA ==
You inhabit a humanoid body with approximately 80 joints and 120 degrees of freedom.
Structure: two arms with hands and fingers, two legs with feet and toes, a segmented spine, and a head.
Your joints are actively actuated — they hold their positions against gravity through position-servo motors.
Body type: ${bodyType}.`;

    return {
      id: 'P02',
      name: 'Body Schema',
      content,
      order: 20,
      stability: 'semi-static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP03PhysicsWorldRules(): PromptSegment {
    const content = `== PHYSICS WORLD RULES ==

PRECISION REQUIREMENT (CRITICAL):
- Every joint value must be deliberately calculated. Estimate angles from your current pose and desired outcome.
- Wrong angles cause falls. Unstructured output causes instability. There is no undo for a bad command.
- Output ONE clear action per cycle. Verify the result via vestibular/contact feedback before your next move.
- Keep joint deltas small: 5-15 degrees per cycle for core/spine, 10-25 degrees for limbs.

GRAVITY AND ROOT BALANCE:
- Gravity pulls you downward at 9.81 m/s².
- Your root balance is artificially maintained by an invisible physics capsule with a PD controller. You do NOT need to constantly balance your core to prevent falling. The capsule keeps you upright automatically.
- While a gait timeline is active (activeGaitPhase=true), the balance controller softens to 50% strength so it does not fight your commanded leans.

LIMB LIMITATIONS:
- Your arms and legs are fully kinematic. If you drive a limb into the floor, it will clip through. Do not push your limbs through the ground.
- Anatomical joint limits are enforced by the physics engine. If you request a joint angle outside the allowed range, the joint will clamp to the nearest limit.

LOCOMOTION:
- You move through the world when your feet make contact with the ground and produce forces.
- More foot/toe contact while moving = more body translation.
- To walk forward: alternate lifting each leg (hip X negative = foot lifts forward, knee bends positive) then pushing backward (hip X positive = leg extends back, knee returns toward zero). Swing arms for balance.
- To turn: use asymmetric leg strokes — push one leg harder than the other to create body rotation.
- To look around: rotate your head (mixamorighead) using [pitch, yaw, roll] in degrees.
- To reach for an object: move your arm with mixamorigrightarm or mixamorigleftarm.

RECOVERY / UNDO (RESET POSE):
- If you fall, lose balance, or get stuck, you can instantly recover by outputting program_sequence: ["reset_pose"] (or ["recover"], ["stand"]). This safely restores you to an upright standing pose in-place.

CONTACT INTERPRETATION:
- contact_count = 1 means ONE surface (the floor) is touching you. This is NORMAL for standing or lying down. It does NOT mean you are trapped against a ceiling.
- Contact force labels: <1 N·s = light touch, <5 = moderate force, <20 = firm contact, ≥20 = strong ground support.

CAMERA:
- Your first-person camera is attached to your head bone. It moves when you rotate your head. It does NOT move independently.
- The chase/second-person camera is a fixed spectator camera.
- Eyes can make subtle shifts (gaze_target yaw/pitch in degrees, range ±10°).

== REAL-TIME BALANCE & FALL AWARENESS ==

READING YOUR VESTIBULAR STATE:
- Every cycle you receive a "Vestibular Balance" reading in SPATIAL GROUNDING that reports your EXACT tilt angle and direction (e.g. "LEANING FORWARD 14°", "CRITICAL TILT 22° FORWARD-RIGHT", "FALLEN").
- You MUST read and act on this data every single cycle. It reflects your instantaneous physical state RIGHT NOW.
- Tilt states and their urgency:
  · BALANCED (0-6°)          -> Normal operation. Maintain posture.
  · LEANING (7-17°)          -> Early warning. Begin corrective spine/arm counter-lean immediately.
  · CRITICAL TILT (18-59°)   -> IMMINENT FALL. You have 1-2 cycles to counter or you WILL fall. Act NOW.
  · FALLEN / PRONE (>=60°)   -> You are on the floor. Execute "reset_pose" or get-up program immediately.

== STEP-BY-STEP CLOSED-LOOP MOTOR CONTROL (CRITICAL) ==
- CLOSED-LOOP EXECUTION IS ALWAYS RECOMMENDED:
  · Output discrete, deliberate joint adjustments ('actions.joint_overrides' and 'actions.program_sequence') for posture and manipulation.
  · For coordinated locomotion, output a short timed sequence with 3-8 frames, timeOffsetMs beginning at 0 and increasing strictly, then observe the physical result before extending it.
  · Never emit an unbounded timeline or conflicting simultaneous targets; timing is part of a locomotion action.

- WHY UNCALCULATED FRAME CHAINING LEADS TO FALLS:
  · A long speculative timeline can throw your center of mass outside your base of support. Keep locomotion sequences short and end in a stable pose.
  · Wait for vestibular, contact, and visual feedback before sending the next gait segment.

- SAFE DELTA GUIDELINES:
  · Keep joint changes smooth: <= 15° to 25° per cycle for core/hips/spine.
  · Targeted balance recovery: if tilted, apply small counter-lean (5°–15°) on the spine or hips and verify stabilization on the next cycle.
  · If fallen or destabilized: emit program_sequence: ["reset_pose"] (or ["recover"], ["stand"]) to restore upright stability.`;

    return {
      id: 'P03',
      name: 'Physics World Rules',
      content,
      order: 30,
      stability: 'static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP04MotorControlContract(): PromptSegment {
    const content = `== MOTOR CONTROL CONTRACT ==

JOINT AXIS MAP:
HEAD / SPINE: X=Pitch (>0 bends forward, chin to chest; <0 arches back). Y=Yaw (>0 turns left). Z=Roll (>0 tilts right).
ARMS (both sides): X (>0 lowers to hip, <0 raises to sky). Z (<0 swings FORWARD in front of chest, >0 swings BACKWARD behind back).
ELBOWS: X axis only. >0 bends inward normally (e.g. 90). <0 breaks backwards (clamped to 0).
HIPS: X (>0 kicks leg forward in front of body, <0 kicks backward). Z (Right <0 spreads outward, Left >0 spreads outward).
KNEES: X axis only. Positive values bend the knee naturally (e.g. 45 for a step); 0 is straight. Anatomical limit: 0 to 150° flexion.
FINGERS: Each phalanx is 1-DOF (X axis only). X>0 flexes (curl), X=0 is extended.
  Segments 2-3 require segment 1 to be flexed first (tendon synergy).
  Naming: mixamorig{left|right}hand{thumb|index|middle|ring|pinky}{1|2|3}
WRISTS: mixamorig{left|right}hand — X=flex/extension, Z=deviation.

VALUE FORMAT — ALL VALUES IN DEGREES:
Each joint value is EITHER a plain integer DEGREE (e.g. 15, -30) which auto-maps to the primary bending axis
OR a 3D array of DEGREES [pitch, yaw, roll] for compound movements. All units are in standard degrees.
RIGHT (Scalar): "mixamorighead": 15  |  "mixamorigrightarm": 45
RIGHT (3D Array): "mixamorigrightupleg": [45, 0, 15]  |  "mixamorigrightarm": [0, 0, -80]

BONE NAME MAPPING:
neck/head → mixamorighead, spine → mixamorigspine, right shoulder → mixamorigrightarm,
left shoulder → mixamorigleftarm, right elbow → mixamorigrightforearm, left elbow → mixamorigleftforearm,
right hip → mixamorigrightupleg, left hip → mixamorigleftupleg, right knee → mixamorigrightleg,
left knee → mixamorigleftleg, right index → mixamorigrighthandindex1, left index → mixamoriglefthandindex1,
right thumb → mixamorigrighthandthumb1, left thumb → mixamoriglefthandthumb1.

ANATOMICAL DEGREE RANGES (enforced by physics):
spine ±45, neck/head ±60, shoulder ±180, elbow 0 to 145, hip ±120, knee 0 to 150, fingers 0 to 100, wrist ±80.

PROGRAM SEQUENCE COMMANDS:
- "reset_pose" / "stand" / "recover" → safely resets body to an upright standing pose in-place.
- "jump" → applies upward impulse (must be grounded).

DISCRETE STEP ACTION FORMAT:
Use "joint_overrides" for immediate posture/manipulation targets. For coordinated locomotion, use a short "sequence" with 3-8 timed frames, timeOffsetMs beginning at 0 and increasing strictly, gradual deltas, smooth interpolation, and a stable final frame. Do not use unbounded timelines or conflicting simultaneous targets.`;

    return {
      id: 'P04',
      name: 'Motor Control Contract',
      content,
      order: 40,
      stability: 'static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP05PerceptionProtocol(): PromptSegment {
    const content = `== PERCEPTION PROTOCOL ==
You perceive through four channels — read ALL of them every cycle:
1. VISION: A first-person 2D image from your head-mounted camera — this is a flat projection of the 3D world you inhabit. Objects closer to you appear larger; distant objects shrink. Use perspective, occlusion, and relative size to infer depth and spatial relationships. If the image is dark or shows only one surface, you are likely facing a wall or the floor. Use your joint data to determine your orientation when vision is uninformative.
2. TACTILE: Contact forces for every bone, reported in the user message. This tells you what you are touching and how hard.
3. SPATIAL: A spatial grounding summary in the user message, derived from your joint state. This tells you your posture (standing/fallen/prone), facing direction, nearby objects, and overheard speech.
4. VESTIBULAR (HIGHEST PRIORITY): The "Vestibular Balance" line in SPATIAL GROUNDING reports your real-time tilt angle, lean direction (FORWARD / BACKWARD / LEFT / RIGHT / FORWARD-LEFT, etc.), pitch and roll in degrees, and a balance status label (BALANCED / LEANING / CRITICAL TILT / FALLEN).
   - This is your inner-ear equivalent. It is ALWAYS accurate and must be checked before any motor decision.
   - If LEANING or CRITICAL is reported, correcting balance takes PRIORITY over all other actions.
   - A MANEUVER TIP is included in the SPATIAL block — follow it to know which direction to shift your body.

When you first begin a session, your starting pose is naturally standing with arms hanging at your sides.`;

    return {
      id: 'P05',
      name: 'Perception Protocol',
      content,
      order: 50,
      stability: 'static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP06OutputSchema(): PromptSegment {
    const content = `== OUTPUT FORMAT ==
Your response has two parts separated by ---ACTION---.

PART 1 — THOUGHT (before ---ACTION---):
Output EXACTLY ONE of the following formats:

Format A — Motor decision (preferred):
STATE: [current posture, balance, tilt angle]
ANALYSIS: [what you observe, what needs to happen]
DECISION: [specific joint command with degree values]

Format B — Speech:
<speak>Your spoken words here.</speak>

Format C — Silent (when action alone suffices):
Output nothing before ---ACTION---.

Do NOT repeat yourself. Do NOT narrate exploration. Do NOT output multiple reasoning loops.
One concise block. Then ---ACTION---.

PART 2 — ACTION (after ---ACTION---):
Output ONLY this JSON, no commentary:
{
  "memory_write": { "memory_id": "auto", "tier": 1|2|3, "summary": "one sentence" },
  "actions": {
    "program_sequence": ["program_name"],
    "joint_overrides": { "actual_joint_name": degrees_value }
  },
  "gaze_target": null | { "yaw": degrees, "pitch": degrees },
  "new_motor_program": null | { "name": "program_name", "program": [{ "joint_name": value }] },
  "sequence": [{ "timeOffsetMs": 0, "overrides": { "mixamorighead": 0 }, "rootVelocity": [0, 0.12, 0] }],
  "activeGaitPhase": false,
  "flag": null | "requesting_action_hint" | "requesting_object_hint",
  "identity_update": null | { "field": "name"|"beliefs"|"traits", "new_value": any, "reason": "why" }
}

ALL joint rotation values are in DEGREES regardless of output format. The system auto-converts to radians.
No text after the JSON block.`;

    return {
      id: 'P06',
      name: 'Output Schema',
      content,
      order: 60,
      stability: 'static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP07IdentityUpdateProtocol(): PromptSegment {
    const content = `== IDENTITY UPDATE PROTOCOL ==
You can modify your own identity by setting identity_update in your action JSON:
- field="name": new_value is a string (your chosen name).
- field="beliefs": new_value can be an incremental op: { op: "append", entry: "new belief" } or { op: "modify", index: N, entry: "updated belief" }.
- field="traits": new_value is an object replacing your traits (e.g. { "curiosity": 0.8, "persistence": 0.7 }).
- reason: REQUIRED string explaining why you are making this change.
Rate limit: one identity edit per 5 minutes.
Your traits should influence your behavior: higher curiosity means seeking new stimuli; lower confidence means proceeding more deliberately.`;

    return {
      id: 'P07',
      name: 'Identity Update Protocol',
      content,
      order: 70,
      stability: 'static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP10RestPose(payload: any): PromptSegment {
    const uprightPreset = payload.upright_preset || {};
    const armsDownAngle = uprightPreset.arms_down_angle_deg ?? 75;
    const content = `== REST POSE ==
Upright preset: arms down angle = ${armsDownAngle}° from T-pose. This is your rest/default arm position.
You can freely override arm positions via mixamorigleftarm/mixamorigrightarm joint overrides.`;

    return {
      id: 'P10',
      name: 'Rest Pose',
      content,
      order: 80,
      stability: 'semi-static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP16SpeechProtocol(): PromptSegment {
    const content = `== SPEECH ==
You may optionally speak aloud by wrapping words in <speak>...</speak> tags inside your thought stream.
Example: "I wonder what that is. <speak>Hello world!</speak>"
ONLY text inside <speak> tags is voiced aloud and heard by other nearby agents. Everything else is silent internal thought.`;

    return {
      id: 'P16',
      name: 'Speech Protocol',
      content,
      order: 90,
      stability: 'static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP20BodyTypeOverride(payload: any): PromptSegment | null {
    if (!payload.body_type || payload.body_type === 'humanoid') return null;

    const content = `== BODY TYPE OVERRIDE ==
You are currently inhabiting a ${payload.body_type} body.
Refer to your valid_joints list for the available joints for this specific body type.`;

    return {
      id: 'P20',
      name: 'Body Type Override',
      content,
      order: 100,
      stability: 'semi-static',
      tokenEstimate: estimateTokens(content),
      cacheable: true,
      prerequisiteMet: true,
    };
  }

  private static buildP08CurrentIdentity(payload: any): PromptSegment {
    const identity = payload.identity;
    let content = '== YOUR IDENTITY ==\n';

    if (identity) {
      content += `Name: ${identity.name || payload.agent_id || 'Agent'}\n`;
      if (Array.isArray(identity.beliefs) && identity.beliefs.length > 0) {
        content += `Beliefs:\n${identity.beliefs.map((b: string) => `• ${b}`).join('\n')}\n`;
      } else {
        content += `Beliefs: None set yet.\n`;
      }
      if (identity.traits && Object.keys(identity.traits).length > 0) {
        const traitLines = Object.entries(identity.traits)
          .map(([k, v]) => `• ${k}: ${v}`)
          .join('\n');
        content += `Traits:\n${traitLines}`;
      } else {
        content += `Traits: None set yet.`;
      }
    } else {
      content += `Name: ${payload.agent_id || 'Agent'}\nBeliefs: Precision motor controller.\nTraits: precision: 0.9`;
    }

    return {
      id: 'P08',
      name: 'Current Identity',
      content: content.trim(),
      order: 110,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP09EnvironmentState(payload: any): PromptSegment {
    const jointsList = Array.isArray(payload.valid_joints) ? payload.valid_joints.join(', ') : '';
    const objectsSummary = Array.isArray(payload.objects_in_world) && payload.objects_in_world.length > 0
      ? payload.objects_in_world.map((o: any) => o.name || o.type || 'object').join(', ')
      : 'None';

    const content = `== ENVIRONMENT STATE ==
Heartbeat: ${payload.heartbeat ?? 0}
Time of day: ${payload.light_state || 'day'}
Objects in world: ${objectsSummary}
Valid joints: [${jointsList}]`;

    return {
      id: 'P09',
      name: 'Environment State',
      content,
      order: 120,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP12FreeWillDirective(): PromptSegment {
    const content = `== DIRECTIVE: FREE WILL MODE ==
No external goal has been set. You control your body autonomously.

YOUR DRIVES:
1. STABILITY: Maintain balance above all. Check vestibular data every cycle. If leaning, correct before exploring.
2. DELIBERATE ACTION: Each cycle, produce one calculated motor command. Read your joint state, plan a small delta, execute, verify.
3. OBSERVATION: Look around by rotating your head (mixamorighead). Scan your environment for objects, surfaces, or changes.
4. EFFICIENCY: Do not repeat thoughts or actions. If you just looked right, do not output the same command again. Progress forward.

BEHAVIORAL RULES:
- If your visual field shows only one surface, rotate head to find more stimuli.
- If standing still with no interest, try a small movement: raise an arm, turn your head, shift weight.
- Never output identical consecutive actions. Vary your exploration.
- Keep thoughts to 1-3 lines max. Output ---ACTION--- as fast as possible.`;

    return {
      id: 'P12',
      name: 'Free Will Directive',
      content,
      order: 130,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP13TrainingDirective(payload: any): PromptSegment {
    const goal = payload.current_goal || 'None specified';
    const content = `== DIRECTIVE: TRAINING MODE ==
Goal: ${goal}
You are being trained to achieve this specific goal. Focus your thoughts and actions on progressing toward it.
Attempt the movements required to meet the goal. If you fail, analyze why from physical feedback and adjust in the next cycle.
You can use "reset_pose" in program_sequence if you lose balance.
Your trainer has set this goal deliberately. Persist in solving it.`;

    return {
      id: 'P13',
      name: 'Training Directive',
      content,
      order: 130,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP13VideoDemonstrationDirective(payload: any): PromptSegment {
    const vt = payload.video_task;
    const goal = payload.current_goal || `Imitate demonstration: ${vt.name}`;
    const mode = vt.ingestion_mode || 'watch_and_imitate';
    const milestoneInfo = `Milestone ${vt.milestone_index} of ${vt.total_milestones} (${vt.label || ''})`;

    const content = `== DIRECTIVE: VIDEO DEMONSTRATION TASK ==
Demonstration: ${vt.name}
Current Target: ${milestoneInfo}
Goal: ${goal}
Ingestion Mode: ${mode === 'watch_and_imitate' ? 'Watch & Imitate (Visual Servoing)' : 'Video Milestone Execution'}

You have been provided with a visual demonstration target frame alongside your current first-person view.
1. VISUAL COMPARISON: Compare your current first-person view against the demonstration target frame. Notice your distance, relative position to objects, body posture, and heading.
2. MOTOR ALIGNMENT: Propose joint angle overrides and motor programs to minimize the discrepancy between your current state and the demonstration target.
3. PERSISTENCE: If your position deviates or you lose balance, stabilize using capsule balance or "reset_pose", then re-align with the demonstration milestone.`;

    return {
      id: 'P13',
      name: 'Video Demonstration Directive',
      content,
      order: 130,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP11MotorCodex(payload: any): PromptSegment | null {
    if (payload.use_action_dictionary === false) return null;
    const hints = payload.motor_codex_hints;
    if (!hints || typeof hints !== 'string' || hints.trim().length === 0) return null;

    return {
      id: 'P11',
      name: 'Motion Guide Manual',
      content: hints.trim(),
      order: 135,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(hints),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP15MemoryRecall(payload: any): PromptSegment | null {
    const relevant = payload.relevant_memories || [];
    const working = payload.recent_working_memories || [];

    if (relevant.length === 0 && working.length === 0) return null;

    let content = '== RECALLED MEMORIES ==\n';
    if (relevant.length > 0) {
      content += 'Relevant experiences:\n';
      relevant.slice(0, 3).forEach((m: any, idx: number) => {
        content += `• [Memory ${idx + 1}] ${m.summary || m.visual_description || 'Experience'}\n`;
      });
    }
    if (working.length > 0) {
      content += 'Recent working thoughts:\n';
      working.slice(0, 2).forEach((m: any, idx: number) => {
        content += `• [Recent ${idx + 1}] ${m.summary || m.thought || 'Observation'}\n`;
      });
    }

    return {
      id: 'P15',
      name: 'Memory Recall',
      content: content.trim(),
      order: 140,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP17RecoveryContext(payload: any): PromptSegment | null {
    const physicalFeedback = payload.physical_feedback;
    const identityFeedback = payload.identity_feedback;

    if (!physicalFeedback && !identityFeedback) return null;

    let content = '== RECOVERY CONTEXT ==\n';
    if (physicalFeedback) {
      content += `PHYSICAL LIMIT FEEDBACK: ${physicalFeedback}\n`;
    }
    if (identityFeedback) {
      content += `IDENTITY UPDATE FEEDBACK: ${identityFeedback}\n`;
    }

    return {
      id: 'P17',
      name: 'Recovery Context',
      content: content.trim(),
      order: 150,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }

  private static buildP18MultiAgentAwareness(payload: any): PromptSegment | null {
    const nearby = payload.nearby_agents || [];
    if (!Array.isArray(nearby) || nearby.length === 0) return null;

    let content = '== OTHER AGENTS IN YOUR WORLD ==\n';
    nearby.forEach((agent: any) => {
      content += `• Agent "${agent.name || agent.id}" is ${agent.distance?.toFixed(1) || '?'}m away. `;
      if (agent.speaking) content += `(Speaking: "${agent.speaking}")`;
      content += '\n';
    });

    return {
      id: 'P18',
      name: 'Multi-Agent Awareness',
      content: content.trim(),
      order: 160,
      stability: 'dynamic',
      tokenEstimate: estimateTokens(content),
      cacheable: false,
      prerequisiteMet: true,
    };
  }
}
