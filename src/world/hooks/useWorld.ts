/**
 * React hook to initialize and manage the World Engine.
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { WorldEngine } from "../engine/WorldEngine";
import { PhysicsEngine } from "../engine/PhysicsEngine";
import { AudioEngine } from "../engine/AudioEngine";
import { ObjectManager } from "../engine/ObjectManager";
import { HumanoidPhysicsBinder } from "../engine/HumanoidPhysicsBinder";
import { generateCombinedMultiAgentMJCF } from "../engine/MJCFHumanoidTemplate";
import { StateRehydrator } from "../engine/StateRehydrator";
import { AgentLoop } from "../agent/AgentLoop";
import { initMemoryMonitor, stopMemoryMonitor } from "../engine/memoryMonitor";
import { useWorldStore } from "../../store/worldStore";
import { useAgentStore } from "../../store/agentStore";
import { useTrainingStore } from "../../store/trainingStore";
import { useConnectionStore } from "../../store/connectionStore";
import { useSpeechStore } from "../../store/speechStore";
import { useAgentRuntimeStore } from "../../store/agentRuntimeStore";
import { useUIStore } from "../../store/uiStore";
import { useMemoryStore } from "../../store/memoryStore";
import { synthiaToast } from "../../utils/synthiaToast";
import { debouncedToast } from "../../utils/toastUtils";
import { STRINGS } from "../../constants/strings";
import { logger as Logger } from "../../utils/logger";
import * as THREE from "three";

export const useWorld = (containerRef: React.RefObject<HTMLDivElement>) => {
  const [isReady, setIsReady] = useState(false);
  const worldEngineRef = useRef<WorldEngine | null>(null);
  const physicsEngineRef = useRef<PhysicsEngine | null>(null);
  const audioEngineRef = useRef<AudioEngine | null>(null);
  const objectManagerRef = useRef<ObjectManager | null>(null);
  const humanoidPhysicsBindersRef = useRef<Map<string, HumanoidPhysicsBinder>>(new Map());
  const activeAgentLoopsRef = useRef<Map<string, AgentLoop>>(new Map());
  const isSpawningRef = useRef(false);
  // Spawn index counter: monotonic, persists across page reloads via sessionStorage.
  // Used to generate stable agent_N IDs so that identity records and physics
  // binder indices never collide after a hypothetical admin remove/despawn flow.
  const spawnIndexRef = useRef<number>(
    Number(sessionStorage.getItem('synthia_spawn_index') ?? '0')
  );

  const worldStore = useWorldStore();
  const agentStore = useAgentStore();
  const pendingOutcomesRef = useRef<any[]>([]);
  const lastJointStateRef = useRef<Record<string, any>>({});
  // Per-agent cooldown for in-place OOB resets. Prevents the check from
  // firing every frame when the model is reset at an out-of-bounds position.
  const oobResetCooldownRef = useRef<Map<string, number>>(new Map());

  // ─── Fall diagnostics ring buffer ────────────────────
  const DIAG_RING_SIZE = 300; // ~5 seconds at 60fps (throttled from 500Hz physics)
  const diagRingRef = useRef<any[]>([]);
  const diagRingIdx = useRef(0);
  const diagRingFull = useRef(false);
  const diagRingFrameCount = useRef(0);
  const diagCaptureDone = useRef(false);
  const diagJointCacheRef = useRef<Map<string, { bodyId: number; qposAdr: number; dofAdr: number; dofCount: number; qposCount: number; name: string }> | null>(null);
  const diagGeomCacheRef = useRef<Map<string, { geomId: number; bodyName: string; type: number }> | null>(null);
  const diagBodyCacheRef = useRef<Map<string, { bodyId: number; mass: number; parentBodyId: number; geomIds: number[] }> | null>(null);

  // ── Early-termination emitter ────────────────────────────────────────
  // Called from the per-frame sync loop when an ET trigger fires (OOB /
  // unhealthy_height / timeout). Pushes a -1.0 outcome (same shape as the
  // existing fall outcome), stands the agent up in-place (only for non-OOB
  // paths — the existing OOB block already does its own reset), and starts
  // a new episode. Idempotent within ~100ms to prevent double-fire.
  const etFiredAtMsRef = useRef<number>(-1);
  const maybeEmitEpisodeTermination = useCallback((reason: 'out_of_bounds' | 'unhealthy_height' | 'timeout') => {
    const training = useTrainingStore.getState();
    if (!training.etEnabled) return;
    if (useAgentStore.getState().directiveMode !== 'training') return;

    const nowMs = performance.now();
    if (etFiredAtMsRef.current > 0 && nowMs - etFiredAtMsRef.current < 100) return;
    etFiredAtMsRef.current = nowMs;

    const activeId = useAgentStore.getState().activeAgentId || 'agent_0';
    const binder = humanoidPhysicsBindersRef.current.get(activeId);
    if (!binder) return;

    // The OOB path is invoked from inside the existing isOutOfWorldBounds()
    // branch which already calls binder.resetPose(). Skip the second reset.
    if (reason !== 'out_of_bounds') {
      // Stand up in-place at the model's current XZ, don't teleport.
      const capsuleBody = binder.getCapsuleBody();
      let x = 0;
      let z = 0;
      if (capsuleBody?.isValid()) {
        const t = capsuleBody.translation();
        x = t.x;
        z = t.z;
      }
      binder.resetPose(new THREE.Vector3(x, 0, z));
    }

    pendingOutcomesRef.current.push({
      type: 'outcome',
      data: {
        success: false,
        reward: -1.0,
        description: `Episode terminated (${reason})`,
        episode_terminated: true,
        termination_reason: reason,
      },
      agentId: activeId,
    });

    training.setLastTerminationReason(reason);
    training.startNewEpisode(performance.now());
    Logger.info(`[useWorld] ET fired: ${reason} for ${activeId}`);
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;

    const init = async () => {
      try {
        Logger.info("useWorld: Initializing MuJoCo physics...");
        const physicsEngine = new PhysicsEngine();
        physicsEngineRef.current = physicsEngine;
        await physicsEngine.init();
        if (cancelled) {
          physicsEngine.cleanup();
          return;
        }

        Logger.info("useWorld: Initializing audio...");
        const audioEngine = new AudioEngine();
        audioEngineRef.current = audioEngine;
        audioEngine
          .init()
          .catch((err) => Logger.error("Audio init failed", err));

        (window as any)._synthia_audio_engine = audioEngine;
        (window as any)._synthia_connection_store_metrics =
          useConnectionStore.getState().setMetrics;

        Logger.info("useWorld: Initializing world engine...");
        const worldEngine = new WorldEngine(
          containerRef.current!,
          physicsEngine,
        );
        worldEngineRef.current = worldEngine;

        const objectManager = new ObjectManager(
          physicsEngine,
          worldEngine.getScene()
        );
        objectManagerRef.current = objectManager;

        // Initialize memory monitor — feeds live WASM/JS heap data to the status bar
        initMemoryMonitor({
          wasmModule: PhysicsEngine.getModule(),
          physicsEngine,
          objectManager,
          intervalMs: 5000,
          onSnapshot: (snap) => useMemoryStore.getState().setSnapshot(snap),
        });

        if (cancelled) {
          physicsEngine.cleanup();
          return;
        }

        objectManager.setEventCallback((type: string, data: any) => {
          if (type === "button_press") {
            pendingOutcomesRef.current.push({
              type: "outcome",
              data: {
                success: true,
                reward: 0.5,
                description: `Pressed button: ${data.id}`,
              },
            });
          }
        });

        const cam = worldEngine.getCameraManager();
        cam.onDragChanged = (dragging, object) => {
          const activeObjManager = objectManagerRef.current;
          if (!activeObjManager) return;

          if (!object) {
            activeObjManager.setDraggingObject(null);
            return;
          }
          let target: THREE.Object3D | null = object;
          while (target && !target.userData.objectId && target.parent) {
            target = target.parent;
          }
          activeObjManager.setDraggingObject(
            dragging && target?.userData.objectId ? target.userData.objectId : null
          );
        };
        cam.onDragEnd = (object) => {
          const activeObjManager = objectManagerRef.current;
          if (!activeObjManager) return;

          let target: THREE.Object3D | null = object;
          while (target && !target.userData.objectId && target.parent) {
            target = target.parent;
          }
          if (target?.userData.objectId) {
            activeObjManager.setObjectPosition(
              target.userData.objectId,
              target.position,
              target.quaternion
            );
          } else {
            let draggedBinder: any = null;
            for (const binder of humanoidPhysicsBindersRef.current.values()) {
              if (object === binder.getModelRoot()) {
                draggedBinder = binder;
                break;
              }
            }
            if (draggedBinder) {
              draggedBinder.setCapsulePosition(
                object.position.x,
                object.position.y,
                object.position.z
              );
            }
          }
        };

        // Expose humanoid binder to window for step-by-step testing
        (window as any).__SYNTHIA_HUMANOID_BINDERS__ = humanoidPhysicsBindersRef.current;
        (window as any).__SYNTHIA_PHYSICS_ENGINE__ = physicsEngine;
        (window as any).__SYNTHIA_MUJOCO_MODULE__ = PhysicsEngine.getModule();
        (window as any).__SYNTHIA_CAMERA__ = worldEngineRef.current.getCamera();
        (window as any).__SYNTHIA_RENDERER__ = worldEngineRef.current.getRenderer();
        (window as any).__SYNTHIA_SCENE__ = worldEngineRef.current.getScene();
        (window as any).THREE = THREE;
        (window as any).__SYNTHIA_FLOOR_MESH__ = worldEngineRef.current.getFloorMesh();

        // Auto-load console highlight geoms (provides highlight_embed, check_floors, etc.)
        const hlScript = document.createElement('script');
        hlScript.src = '/console_highlight_geoms.js';
        document.head.appendChild(hlScript);

        // ── Fall diagnostics: expose to window ────────────
        const getDiagRingFrames = () => {
          return diagRingRef.current.slice();
        };

        (window as any).__SYNTHIA_DIAG_RING__ = getDiagRingFrames;
        (window as any).__SYNTHIA_DIAG_RING_INFO__ = () => {
          const ring = diagRingRef.current;
          console.log(`[DIAG RING] Buffer: ${ring.length}/${DIAG_RING_SIZE} frames, done: ${diagCaptureDone.current}`);
        };
        (window as any).diag_reset = () => {
          diagRingRef.current.length = 0;
          diagRingIdx.current = 0;
          diagRingFull.current = false;
          diagRingFrameCount.current = 0;
          diagCaptureDone.current = false;
          diagJointCacheRef.current = null;
          diagGeomCacheRef.current = null;
          diagBodyCacheRef.current = null;
          console.log('[DIAG] Ring buffer + caches reset. Capturing will restart on next frame.');
        };
        (window as any).diagnose_fall_quick = () => {
          const frames = getDiagRingFrames();
          if (frames.length === 0) {
            console.log('[DIAG] No frames captured yet');
            const pe = physicsEngineRef.current;
            const hb = humanoidPhysicsBindersRef.current.get(useAgentStore.getState().activeAgentId) || humanoidPhysicsBindersRef.current.get('agent_0');
            console.log('[DIAG DEBUG] physics:', !!pe, 'isReady:', pe?.isReady, 'isStepping:', pe?.isStepping, 'isMutating:', pe?.isMutating);
            console.log('[DIAG DEBUG] humanoidBinder:', !!hb);
            if (hb) {
              const capId = hb.getMultiBodyManager()?.getCapsuleBody();
              console.log('[DIAG DEBUG] capsuleBody:', capId);
            }
            console.log('[DIAG DEBUG] ringBuffer length:', diagRingRef.current.length);
            return;
          }
          console.log(`[DIAG] ${frames.length} frames captured. Analyzing...`);
          console.log('[DIAG] Reference frame: MuJoCo world = X-forward, Y-left, Z-up');
          console.log('[DIAG] xfrc = world-frame torque on root body');
          console.log('[DIAG] Spherical joint qpos = [qw,qx,qy,qz], qvel = [wx,wy,wz]');
          console.log('[DIAG] Revolute joint qpos = [angle], qvel = [angVel]');

          // Time series — sample every Nth frame
          const step = Math.max(1, Math.floor(frames.length / 30));
          console.log('\nFrame | RootH  | Tilt  | ComZ  | lFoot  | rFoot  | UpVel  | BkVel  | GRD | spine_p | lHip_p | rHip_p | lKnee | rKnee');
          console.log('------+--------+-------+-------+--------+--------+--------+--------+-----+---------+--------+--------+-------+------');
          for (let i = 0; i < frames.length; i += step) {
            const s = frames[i];
            const j = s.joints ?? {};
            const spineP = j.mixamorigspine?.qvel?.[1]?.toFixed(2) ?? '?';
            const lHipP = j.mixamorigleftupleg?.qvel?.[1]?.toFixed(2) ?? '?';
            const rHipP = j.mixamorigrightupleg?.qvel?.[1]?.toFixed(2) ?? '?';
            const lKnee = j.mixamorigleftleg?.qpos?.[0]?.toFixed(2) ?? '?';
            const rKnee = j.mixamorigrightleg?.qpos?.[0]?.toFixed(2) ?? '?';
            console.log(
              String(s.f).padStart(5) + ' | ' +
              (s.rootH ?? 0).toFixed(3).padStart(5) + ' | ' +
              (s.tilt ?? 0).toFixed(1).padStart(5) + ' | ' +
              (s.comZ ?? 0).toFixed(3).padStart(5) + ' | ' +
              (s.lFootH ?? 0).toFixed(3).padStart(5) + ' | ' +
              (s.rFootH ?? 0).toFixed(3).padStart(5) + ' | ' +
              (s.rootVy ?? 0).toFixed(2).padStart(6) + ' | ' +
              (s.rootLinVz ?? 0).toFixed(2).padStart(6) + ' | ' +
              String(s.grounded).padStart(3) + ' | ' +
              String(spineP).padStart(7) + ' | ' +
              String(lHipP).padStart(6) + ' | ' +
              String(rHipP).padStart(6) + ' | ' +
              String(lKnee).padStart(5) + ' | ' +
              String(rKnee).padStart(5)
            );
          }

          // Root cause
          const first = frames[0], last = frames[frames.length - 1];
          console.log('\n--- ROOT CAUSE ---');
          console.log('Height drop:', (first.rootH - last.rootH).toFixed(3), 'MuJoCo-Z');
          console.log('Tilt increase:', (last.tilt - first.tilt).toFixed(1), 'deg');
          console.log('Initial tilt:', first.tilt.toFixed(1), 'deg → Final tilt:', last.tilt.toFixed(1), 'deg');
          const avgFootZ = (first.lFootH + first.rFootH) / 2;
          const comOffset = first.comZ - avgFootZ;
          console.log('CoM vs avg foot (MuJoCo-Z):', comOffset.toFixed(3), comOffset < -0.05 ? '(BEHIND!)' : '(OK)');

          // Joint state at first and last frame
          console.log('\n--- JOINT STATE (first frame → last frame) ---');
          const jf = first.joints ?? {};
          const jl = last.joints ?? {};
          const jointNames = Object.keys(jf);
          console.log(`  Total joints: ${jointNames.length}`);
          for (const jname of jointNames) {
            const fq = jf[jname]?.qvel ?? [];
            const lq = jl[jname]?.qvel ?? [];
            const fqp = jf[jname]?.qpos ?? [];
            const lqp = jl[jname]?.qpos ?? [];
            if (fq.length > 0) {
              console.log(`  ${jname}: qpos [${fqp.map((v:number) => v.toFixed(3)).join(', ')}] → [${lqp.map((v:number) => v.toFixed(3)).join(', ')}]  qvel [${fq.map((v:number) => v.toFixed(3)).join(', ')}] → [${lq.map((v:number) => v.toFixed(3)).join(', ')}]`);
            }
          }

          // Body positions at first and last frame
          const bf = first.bodies ?? {};
          const bl = last.bodies ?? {};
          const bodyNames = Object.keys(bf);
          if (bodyNames.length > 0) {
            console.log(`\n--- BODY POSITIONS (first frame → last frame) [${bodyNames.length} bodies] ---`);
            console.log('  Body                 | First pos (x,y,z MuJoCo)      | Last pos (x,y,z MuJoCo)       | Delta pos');
            console.log('  ---------------------+-------------------------------+-------------------------------+----------');
            for (const bname of bodyNames) {
              const fp = bf[bname]?.pos ?? [0, 0, 0];
              const lp = bl[bname]?.pos ?? fp;
              const dx = lp[0] - fp[0], dy = lp[1] - fp[1], dz = lp[2] - fp[2];
              console.log(`  ${bname.padEnd(20)} | [${fp.map((v:number) => v.toFixed(3)).join(', ')}] | [${lp.map((v:number) => v.toFixed(3)).join(', ')}] | [${dx.toFixed(3)}, ${dy.toFixed(3)}, ${dz.toFixed(3)}]`);
            }
          }

          // Body velocities at first and last frame
          if (bodyNames.length > 0) {
            console.log(`\n--- BODY VELOCITIES (first frame → last frame) ---`);
            console.log('  Body                 | First linVel (x,y,z)          | First angVel (x,y,z)           | Last linVel');
            console.log('  ---------------------+-------------------------------+-------------------------------+------------');
            for (const bname of bodyNames) {
              const flv = bf[bname]?.linVel ?? [0, 0, 0];
              const fav = bf[bname]?.angVel ?? [0, 0, 0];
              const llv = bl[bname]?.linVel ?? flv;
              console.log(`  ${bname.padEnd(20)} | [${flv.map((v:number) => v.toFixed(3)).join(', ')}] | [${fav.map((v:number) => v.toFixed(3)).join(', ')}] | [${llv.map((v:number) => v.toFixed(3)).join(', ')}]`);
            }
          }

          // Body external forces (cfrc_ext)
          if (bodyNames.length > 0) {
            console.log(`\n--- BODY EXTERNAL FORCES (cfrc_ext) first → last ---`);
            for (const bname of bodyNames) {
              const fc = bf[bname]?.cfrc ?? [0,0,0,0,0,0];
              const lc = bl[bname]?.cfrc ?? fc;
              const fMag = Math.sqrt(fc[0]*fc[0]+fc[1]*fc[1]+fc[2]*fc[2]);
              const lMag = Math.sqrt(lc[0]*lc[0]+lc[1]*lc[1]+lc[2]*lc[2]);
              if (fMag > 0.01 || lMag > 0.01) {
                console.log(`  ${bname.padEnd(20)} | force [${fc.map((v:number) => v.toFixed(2)).join(', ')}] (${fMag.toFixed(2)}N) → [${lc.map((v:number) => v.toFixed(2)).join(', ')}] (${lMag.toFixed(2)}N)`);
              }
            }
          }

          // Geom positions
          const gf = first.geoms ?? {};
          const gl = last.geoms ?? {};
          const geomNames = Object.keys(gf);
          if (geomNames.length > 0) {
            console.log(`\n--- GEOM POSITIONS (first frame → last frame) [${geomNames.length} geoms] ---`);
            for (const gname of geomNames) {
              const fp = gf[gname]?.pos ?? [0, 0, 0];
              const lp = gl[gname]?.pos ?? fp;
              const body = gf[gname]?.bodyName ?? '?';
              console.log(`  ${gname.padEnd(24)} (${body.padEnd(20)}): [${fp.map((v:number) => v.toFixed(3)).join(', ')}] → [${lp.map((v:number) => v.toFixed(3)).join(', ')}]`);
            }
          }

          // Contact summary
          const cf = first.contacts ?? [];
          const cl = last.contacts ?? [];
          console.log(`\n--- CONTACTS (first frame: ${cf.length} contacts, last frame: ${cl.length} contacts) ---`);
          if (cf.length > 0) {
            console.log('  First frame contacts:');
            for (const c of cf) {
              console.log(`    ${c.geom1} ↔ ${c.geom2} | dist=${c.dist.toFixed(4)} | normal=[${c.normal.map((v:number)=>v.toFixed(3)).join(', ')}] | force=[${c.force.map((v:number)=>v.toFixed(2)).join(', ')}] N`);
            }
          }
          if (cl.length > 0) {
            console.log('  Last frame contacts:');
            for (const c of cl) {
              console.log(`    ${c.geom1} ↔ ${c.geom2} | dist=${c.dist.toFixed(4)} | normal=[${c.normal.map((v:number)=>v.toFixed(3)).join(', ')}] | force=[${c.force.map((v:number)=>v.toFixed(2)).join(', ')}] N`);
            }
          }

          // Contact count over time
          console.log('\n--- CONTACT COUNT OVER TIME ---');
          for (let i = 0; i < frames.length; i += step) {
            const s = frames[i];
            const nc = s.contacts?.length ?? 0;
            console.log(`  Frame ${String(s.f).padStart(3)}: ${nc} contacts`);
          }

          // xfrc summary
          console.log('\n--- ROOT xfrc (world-frame torque) ---');
          console.log('First:', first.xfrc?.map((v:number) => v.toFixed(2)).join(', '));
          console.log('Last: ', last.xfrc?.map((v:number) => v.toFixed(2)).join(', '));

          // Download
          const blob = new Blob([JSON.stringify(frames, null, 2)], { type: 'application/json' });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a'); a.href = url; a.download = 'fall_diagnosis.json'; a.click();
          URL.revokeObjectURL(url);
          console.log('[DIAG] JSON downloaded');
        };
        Logger.info('useWorld: Fall diagnostics ring buffer active (300 frames)');

        // ── Capture initial state BEFORE any physics step ──
        try {
          const physEng = physicsEngineRef.current;
          const humBinder = humanoidPhysicsBindersRef.current.get(useAgentStore.getState().activeAgentId) || humanoidPhysicsBindersRef.current.get('agent_0');
          if (physEng && physEng.isReady && humBinder) {
            const capId = humBinder.getMultiBodyManager().getCapsuleBody();
            if (capId !== null && capId >= 0) {
              const w = physEng.getWorld();
              const mdl = w.model;
              const d = w.data;

              // Build caches now so initial frame has data
              const mujocoModule = PhysicsEngine.getModule();
              if (mujocoModule) {
                if (!diagJointCacheRef.current) {
                  const cache = new Map<string, { bodyId: number; qposAdr: number; dofAdr: number; dofCount: number; qposCount: number; name: string }>();
                  for (let ji = 0; ji < mdl.njnt; ji++) {
                    const bodyId = mdl.jnt_bodyid[ji];
                    const bodyName = mujocoModule.mj_id2name(mdl, 1, bodyId);
                    if (!bodyName || bodyName === 'root_capsule') continue;
                    const jntType = mdl.jnt_type[ji];
                    if (jntType === 0) continue;
                    const qp = jntType === 1 ? 4 : 1;
                    const dc = jntType === 1 ? 3 : 1;
                    const jointName = mujocoModule.mj_id2name(mdl, 3, ji) || bodyName;
                    cache.set(jointName, {
                      bodyId,
                      qposAdr: mdl.jnt_qposadr[ji],
                      dofAdr: mdl.jnt_dofadr[ji],
                      dofCount: dc,
                      qposCount: qp,
                      name: jointName,
                    });
                  }
                  diagJointCacheRef.current = cache;
                }
                if (!diagGeomCacheRef.current) {
                  const geomCache = new Map<string, { geomId: number; bodyName: string; type: number }>();
                  for (let gi = 0; gi < mdl.ngeom; gi++) {
                    const geomName = mujocoModule.mj_id2name(mdl, 5, gi);
                    if (!geomName || geomName.startsWith('env_slot_') || geomName === 'floor') continue;
                    const bodyId = mdl.geom_bodyid[gi];
                    const bodyName = mujocoModule.mj_id2name(mdl, 1, bodyId) || '';
                    geomCache.set(geomName, { geomId: gi, bodyName, type: mdl.geom_type[gi] });
                  }
                  diagGeomCacheRef.current = geomCache;
                }
                if (!diagBodyCacheRef.current) {
                  const bodyCache = new Map<string, { bodyId: number; mass: number; parentBodyId: number; geomIds: number[] }>();
                  for (let bi = 0; bi < mdl.nbody; bi++) {
                    const bodyName = mujocoModule.mj_id2name(mdl, 1, bi);
                    if (!bodyName || bodyName.startsWith('env_slot_') || bodyName === 'floor' || bodyName === 'world') continue;
                    const geomIds: number[] = [];
                    const gadr = mdl.body_geomadr[bi];
                    const gnum = mdl.body_geomnum[bi];
                    for (let gi = 0; gi < gnum; gi++) geomIds.push(gadr + gi);
                    bodyCache.set(bodyName, {
                      bodyId: bi,
                      mass: mdl.body_mass[bi],
                      parentBodyId: mdl.body_parentid[bi],
                      geomIds,
                    });
                  }
                  diagBodyCacheRef.current = bodyCache;
                }
              }

              let totalM = 0, comX = 0, comY = 0, comZ = 0;
              for (let bi = 0; bi < mdl.nbody; bi++) {
                const m = mdl.body_mass[bi];
                if (m <= 0) continue;
                const bname = mujocoModule ? mujocoModule.mj_id2name(mdl, 1, bi) : '';
                if (bname.startsWith('env_slot_') || bname === 'floor' || bname === 'world') continue;
                comX += m * d.xpos[bi * 3];
                comY += m * d.xpos[bi * 3 + 1];
                comZ += m * d.xpos[bi * 3 + 2];
                totalM += m;
              }
              if (totalM > 0) { comX /= totalM; comY /= totalM; comZ /= totalM; }
              const qx = d.xquat[capId * 4 + 1], qy = d.xquat[capId * 4 + 2];
              const upZ = 1 - 2 * (qx * qx + qy * qy);
              const tiltDeg = Math.acos(Math.min(1, Math.max(-1, upZ))) * 180 / Math.PI;
              const feetMap = humBinder.getMultiBodyManager().getRigidBodiesMap();
              const lFootId = feetMap.get('mixamorigleftfoot');
              const rFootId = feetMap.get('mixamorigrightfoot');
              const lFootH = lFootId !== undefined ? d.xpos[lFootId * 3 + 2] : 0;
              const rFootH = rFootId !== undefined ? d.xpos[rFootId * 3 + 2] : 0;
              const rootDof = mdl.body_dofadr[capId];

              // Build joint data for initial frame
              const jointsInit: Record<string, { qpos: number[]; qvel: number[]; bodyId: number; bodyName: string }> = {};
              const jCache = diagJointCacheRef.current;
              if (jCache) {
                for (const [, info] of jCache) {
                  const qpos: number[] = [];
                  for (let k = 0; k < info.qposCount; k++) qpos.push(d.qpos[info.qposAdr + k]);
                  const qvel: number[] = [];
                  for (let k = 0; k < info.dofCount; k++) qvel.push(d.qvel[info.dofAdr + k]);
                  jointsInit[info.name] = { qpos, qvel, bodyId: info.bodyId, bodyName: info.name };
                }
              }

              // Build body data for initial frame
              const bodiesInit: Record<string, { pos: number[]; quat: number[]; linVel: number[]; angVel: number[]; cfrc: number[]; mass: number }> = {};
              const bCache = diagBodyCacheRef.current;
              if (bCache) {
                for (const [name, info] of bCache) {
                  const bi = info.bodyId;
                  const pos = [d.xpos[bi * 3], d.xpos[bi * 3 + 1], d.xpos[bi * 3 + 2]];
                  const quat = [d.xquat[bi * 4], d.xquat[bi * 4 + 1], d.xquat[bi * 4 + 2], d.xquat[bi * 4 + 3]];
                  const cv = bi * 6;
                  const linVel = [d.cvel[cv], d.cvel[cv + 1], d.cvel[cv + 2]];
                  const angVel = [d.cvel[cv + 3], d.cvel[cv + 4], d.cvel[cv + 5]];
                  const cfrc = [d.cfrc_ext[cv], d.cfrc_ext[cv + 1], d.cfrc_ext[cv + 2], d.cfrc_ext[cv + 3], d.cfrc_ext[cv + 4], d.cfrc_ext[cv + 5]];
                  bodiesInit[name] = { pos, quat, linVel, angVel, cfrc, mass: info.mass };
                }
              }

              // Build geom data for initial frame
              const geomsInit: Record<string, { pos: number[]; bodyName: string }> = {};
              const gCache = diagGeomCacheRef.current;
              if (gCache) {
                for (const [name, info] of gCache) {
                  const gi = info.geomId;
                  geomsInit[name] = {
                    pos: [d.geom_xpos[gi * 3], d.geom_xpos[gi * 3 + 1], d.geom_xpos[gi * 3 + 2]],
                    bodyName: info.bodyName,
                  };
                }
              }

              // Build contact data for initial frame
              const contactsInit: Array<{ geom1: string; geom2: string; pos: number[]; dist: number; normal: number[]; force: number[] }> = [];
              if (mujocoModule && d.ncon > 0 && d.ncon <= 200) {
                const forceBuffer = new mujocoModule.DoubleBuffer(6);
                try {
                  for (let ci = 0; ci < d.ncon; ci++) {
                    try {
                      const contact = d.contact.get(ci);
                      if (!contact) continue;
                      const g1 = mujocoModule.mj_id2name(mdl, 5, contact.geom1) || `geom_${contact.geom1}`;
                      const g2 = mujocoModule.mj_id2name(mdl, 5, contact.geom2) || `geom_${contact.geom2}`;
                      mujocoModule.mj_contactForce(mdl, d, ci, forceBuffer);
                      const fv = forceBuffer.GetView();
                      contactsInit.push({
                        geom1: g1, geom2: g2,
                        pos: [contact.pos[0], contact.pos[1], contact.pos[2]],
                        dist: contact.dist,
                        normal: [contact.frame[0], contact.frame[1], contact.frame[2]],
                        force: [fv[0], fv[1], fv[2]],
                      });
                    } catch { /* ignore */ }
                  }
                } finally {
                  forceBuffer.delete();
                }
              }

              diagRingRef.current.push({
                f: diagRingFrameCount.current++,
                t: Date.now(),
                rootH: d.xpos[capId * 3 + 2],
                rootX: d.xpos[capId * 3],
                rootY: d.xpos[capId * 3 + 1],
                tilt: tiltDeg,
                comZ, comX, comY,
                rootVy: d.qvel[rootDof + 1],
                rootLinVz: d.qvel[rootDof + 2],
                lFootH, rFootH,
                ncon: d.ncon,
                grounded: humBinder.getIsGrounded(),
                xfrc: [d.xfrc_applied[capId * 6 + 3], d.xfrc_applied[capId * 6 + 4], d.xfrc_applied[capId * 6 + 5]],
                joints: jointsInit,
                bodies: bodiesInit,
                geoms: geomsInit,
                contacts: contactsInit,
              });
              console.log(`[DIAG] Initial state captured: tilt=${tiltDeg.toFixed(1)}° rootH=${(d.xpos[capId * 3 + 2]).toFixed(3)} joints=${Object.keys(jointsInit).length} bodies=${Object.keys(bodiesInit).length} geoms=${Object.keys(geomsInit).length} contacts=${contactsInit.length}`);
            }
          }
        } catch { /* ignore */ }
        if (diagRingRef.current.length >= DIAG_RING_SIZE) diagCaptureDone.current = true;

        Logger.info("useWorld: Starting animation loop...");
        worldEngineRef.current.start(
          () => {
            // ── Per-step (500Hz): Root balance correction + diagnostics capture ──
            const physics = physicsEngineRef.current;

            if (!physics || physics.isStepping || physics.isMutating) {
              return;
            }

            // Per-physics-step (500 Hz) locomotion control:
            //   1. Root balance corrector (from Road-2, now on the heavy root).
            //   2. Critically-damped root velocity drive (Road-3) — replaces the old
            //      30 Hz root TELEPORT with a real velocity servo. Suspends airborne.
            //   3. COM lean-reflex + capture-step (Road-4) — applied ON TOP of the
            //      flushed pose ctrl every step so the reflex leads the walk while
            //      the root drive (now 0.15 m/s) stays a gentle assist.
            for (const binder of humanoidPhysicsBindersRef.current.values()) {
              try {
                binder.applyBalanceStep();
                binder.applyRootVelocityDrive(performance.now());
                binder.applyComReflexStep(0.002);
                // RMBS v1 (opt-in): reaction-mass balance. Self-gated on
                // reactionMassEnabled — a silent no-op when disabled, on a
                // legacy world without the reaction_mass body, or before
                // build step D.
                binder.applyReactionMassStep(0.002);
              } catch (error) {
                Logger.warn(`per-step control (${binder.agentId}) failed:`, error);
              }
            }

            // Throttle: capture snapshot once per ~8 steps (= once per frame at 500Hz/60fps)
            if (diagCaptureDone.current) return;
            const stepCount = physics.getStepCount();
            if (stepCount % 8 !== 0) return;

            try {
              const physEng = physicsEngineRef.current;
              const humBinder = humanoidPhysicsBindersRef.current.get(useAgentStore.getState().activeAgentId) || humanoidPhysicsBindersRef.current.get('agent_0');
              if (physEng && physEng.isReady && humBinder) {
                const capId = humBinder.getMultiBodyManager().getCapsuleBody();
                if (capId !== null && capId >= 0) {
                  const w = physEng.getWorld();
                  const mdl = w.model;
                  const d = w.data;

                  const mujocoModule = PhysicsEngine.getModule();
                  if (mujocoModule) {
                    if (!diagJointCacheRef.current) {
                      const cache = new Map<string, { bodyId: number; qposAdr: number; dofAdr: number; dofCount: number; qposCount: number; name: string }>();
                      for (let ji = 0; ji < mdl.njnt; ji++) {
                        const bodyId = mdl.jnt_bodyid[ji];
                        const bodyName = mujocoModule.mj_id2name(mdl, 1, bodyId);
                        if (!bodyName || bodyName === 'root_capsule') continue;
                        const jntType = mdl.jnt_type[ji];
                        if (jntType === 0) continue;
                        const qp = jntType === 1 ? 4 : 1;
                        const dc = jntType === 1 ? 3 : 1;
                        const jointName = mujocoModule.mj_id2name(mdl, 3, ji) || bodyName;
                        cache.set(jointName, {
                          bodyId,
                          qposAdr: mdl.jnt_qposadr[ji],
                          dofAdr: mdl.jnt_dofadr[ji],
                          dofCount: dc,
                          qposCount: qp,
                          name: jointName,
                        });
                      }
                      diagJointCacheRef.current = cache;
                      console.log(`[DIAG] Joint cache: ${cache.size} joints mapped`);
                    }

                    if (!diagGeomCacheRef.current) {
                      const geomCache = new Map<string, { geomId: number; bodyName: string; type: number }>();
                      for (let gi = 0; gi < mdl.ngeom; gi++) {
                        const geomName = mujocoModule.mj_id2name(mdl, 5, gi);
                        if (!geomName || geomName.startsWith('env_slot_') || geomName === 'floor') continue;
                        const bodyId = mdl.geom_bodyid[gi];
                        const bodyName = mujocoModule.mj_id2name(mdl, 1, bodyId) || '';
                        geomCache.set(geomName, { geomId: gi, bodyName, type: mdl.geom_type[gi] });
                      }
                      diagGeomCacheRef.current = geomCache;
                      console.log(`[DIAG] Geom cache: ${geomCache.size} geoms mapped`);
                    }

                    if (!diagBodyCacheRef.current) {
                      const bodyCache = new Map<string, { bodyId: number; mass: number; parentBodyId: number; geomIds: number[] }>();
                      for (let bi = 0; bi < mdl.nbody; bi++) {
                        const bodyName = mujocoModule.mj_id2name(mdl, 1, bi);
                        if (!bodyName || bodyName.startsWith('env_slot_') || bodyName === 'floor' || bodyName === 'world') continue;
                        const geomIds: number[] = [];
                        const gadr = mdl.body_geomadr[bi];
                        const gnum = mdl.body_geomnum[bi];
                        for (let gi = 0; gi < gnum; gi++) geomIds.push(gadr + gi);
                        bodyCache.set(bodyName, {
                          bodyId: bi,
                          mass: mdl.body_mass[bi],
                          parentBodyId: mdl.body_parentid[bi],
                          geomIds,
                        });
                      }
                      diagBodyCacheRef.current = bodyCache;
                      console.log(`[DIAG] Body cache: ${bodyCache.size} bodies mapped`);
                    }
                  }

                  const rootDof = mdl.body_dofadr[capId];
                  let totalM = 0, comX = 0, comY = 0, comZ = 0;
                  for (let bi = 0; bi < mdl.nbody; bi++) {
                    const m = mdl.body_mass[bi];
                    if (m <= 0) continue;
                    const bname = mujocoModule ? mujocoModule.mj_id2name(mdl, 1, bi) : '';
                    if (bname.startsWith('env_slot_') || bname === 'floor' || bname === 'world') continue;
                    comX += m * d.xpos[bi * 3];
                    comY += m * d.xpos[bi * 3 + 1];
                    comZ += m * d.xpos[bi * 3 + 2];
                    totalM += m;
                  }
                  if (totalM > 0) { comX /= totalM; comY /= totalM; comZ /= totalM; }
                  const qx = d.xquat[capId * 4 + 1], qy = d.xquat[capId * 4 + 2];
                  const upZ = 1 - 2 * (qx * qx + qy * qy);
                  const tiltDeg = Math.acos(Math.min(1, Math.max(-1, upZ))) * 180 / Math.PI;
                  const feetMap = humBinder.getMultiBodyManager().getRigidBodiesMap();
                  const lFootId = feetMap.get('mixamorigleftfoot');
                  const rFootId = feetMap.get('mixamorigrightfoot');
                  const lFootH = lFootId !== undefined ? d.xpos[lFootId * 3 + 2] : 0;
                  const rFootH = rFootId !== undefined ? d.xpos[rFootId * 3 + 2] : 0;

                  const jointCache = diagJointCacheRef.current;
                  const joints: Record<string, { qpos: number[]; qvel: number[]; bodyId: number; bodyName: string }> = {};
                  if (jointCache) {
                    for (const [, info] of jointCache) {
                      const qpos: number[] = [];
                      for (let k = 0; k < info.qposCount; k++) qpos.push(d.qpos[info.qposAdr + k]);
                      const qvel: number[] = [];
                      for (let k = 0; k < info.dofCount; k++) qvel.push(d.qvel[info.dofAdr + k]);
                      joints[info.name] = { qpos, qvel, bodyId: info.bodyId, bodyName: info.name };
                    }
                  }

                  const bodyCache = diagBodyCacheRef.current;
                  const bodies: Record<string, { pos: number[]; quat: number[]; linVel: number[]; angVel: number[]; cfrc: number[]; mass: number }> = {};
                  if (bodyCache) {
                    for (const [name, info] of bodyCache) {
                      const bi = info.bodyId;
                      const pos = [d.xpos[bi * 3], d.xpos[bi * 3 + 1], d.xpos[bi * 3 + 2]];
                      const quat = [d.xquat[bi * 4], d.xquat[bi * 4 + 1], d.xquat[bi * 4 + 2], d.xquat[bi * 4 + 3]];
                      const cv = bi * 6;
                      const linVel = [d.cvel[cv], d.cvel[cv + 1], d.cvel[cv + 2]];
                      const angVel = [d.cvel[cv + 3], d.cvel[cv + 4], d.cvel[cv + 5]];
                      const cfrc = [d.cfrc_ext[cv], d.cfrc_ext[cv + 1], d.cfrc_ext[cv + 2], d.cfrc_ext[cv + 3], d.cfrc_ext[cv + 4], d.cfrc_ext[cv + 5]];
                      bodies[name] = { pos, quat, linVel, angVel, cfrc, mass: info.mass };
                    }
                  }

                  const geomCache = diagGeomCacheRef.current;
                  const geoms: Record<string, { pos: number[]; bodyName: string }> = {};
                  if (geomCache) {
                    for (const [name, info] of geomCache) {
                      const gi = info.geomId;
                      geoms[name] = {
                        pos: [d.geom_xpos[gi * 3], d.geom_xpos[gi * 3 + 1], d.geom_xpos[gi * 3 + 2]],
                        bodyName: info.bodyName,
                      };
                    }
                  }

                  const snapshot = {
                    f: diagRingFrameCount.current++,
                    t: Date.now(),
                    rootH: d.xpos[capId * 3 + 2],
                    rootX: d.xpos[capId * 3],
                    rootY: d.xpos[capId * 3 + 1],
                    tilt: tiltDeg,
                    comZ, comX, comY,
                    rootVy: d.qvel[rootDof + 1],
                    rootLinVz: d.qvel[rootDof + 2],
                    lFootH, rFootH,
                    ncon: d.ncon,
                    grounded: humBinder.getIsGrounded(),
                    xfrc: [d.xfrc_applied[capId * 6 + 3], d.xfrc_applied[capId * 6 + 4], d.xfrc_applied[capId * 6 + 5]],
                    joints,
                    bodies,
                    geoms,
                    contacts: [],
                  };
                  if (!diagCaptureDone.current) {
                    const ring = diagRingRef.current;
                    ring.push(snapshot);
                    diagRingIdx.current = ring.length;
                    if (ring.length >= DIAG_RING_SIZE) {
                      diagCaptureDone.current = true;
                      console.log(`[DIAG] Captured ${DIAG_RING_SIZE} frames. Ring buffer full.`);
                    }
                  }
                }
              }
            } catch (diagErr) { if (diagRingFrameCount.current < 3) console.warn('[DIAG CAPTURE ERROR]', diagErr); }
          },
          () => {
            // ── Per-frame (60Hz): Object sync, motor updates, camera, boundary ──
            try {
              objectManagerRef.current?.update();
              objectManagerRef.current?.syncVisuals();
            } catch (error) {
              Logger.warn("ObjectManager update error caught safely", error);
            }

            if (worldStore.bodyType === 'humanoid') {
              for (const [id, binder] of humanoidPhysicsBindersRef.current.entries()) {
                try {
                  binder.updateMotorTargets();
                  binder.syncVisuals();

                  const activeId = useAgentStore.getState().activeAgentId || 'agent_0';
                  if (id === activeId) {
                    binder.renderAICameraHelper(
                      useWorldStore.getState().showAICameraHelper,
                      worldEngineRef.current?.getCameraManager().getCameraData()
                    );
                    const state = binder.getJointState();
                    lastJointStateRef.current = state;

                    const headTransform = binder.getHeadTransform();
                    if (headTransform) {
                      const headMatrix = new THREE.Matrix4().compose(
                        headTransform.position,
                        headTransform.quaternion,
                        new THREE.Vector3(1, 1, 1)
                      );

                      let capsuleQuat: THREE.Quaternion | undefined;
                      let capsulePos: THREE.Vector3 | undefined;
                      const capsuleBody = binder.getCapsuleBody();
                      if (capsuleBody?.isValid()) {
                        const t = capsuleBody.translation();
                        const r = capsuleBody.rotation();
                        capsulePos = new THREE.Vector3(t.x, t.y, t.z);
                        capsuleQuat = new THREE.Quaternion(r.x, r.y, r.z, r.w);
                      }

                      worldEngineRef.current?.getCameraManager().update(headMatrix, headTransform.position, capsuleQuat, capsulePos, activeId);
                    }
                  }

                  if (binder.isOutOfWorldBounds()) {
                    // Cooldown: skip if we reset this agent within the last
                    // 500ms — gives the model time to settle after an in-place
                    // reset that lands near the boundary.
                    const nowMs = performance.now();
                    const lastReset = oobResetCooldownRef.current.get(id) || 0;
                    if (nowMs - lastReset < 500) {
                      // Still in cooldown — skip this frame's OOB check.
                    } else {
                      // Stand up in-place at the model's current XZ, y=0.
                      const capsuleBody = binder.getCapsuleBody();
                      let x = 0;
                      let z = 0;
                      if (capsuleBody?.isValid()) {
                        const t = capsuleBody.translation();
                        x = t.x;
                        z = t.z;
                      }
                      binder.resetPose(new THREE.Vector3(x, 0, z));
                      oobResetCooldownRef.current.set(id, nowMs);

                      // If training mode + ET are on, the OOB counts as a
                      // terminated episode. Emit an outcome and start the next.
                      if (id === activeId) {
                        maybeEmitEpisodeTermination('out_of_bounds');
                      }
                    }
                  }

                  // Early-termination (height + timeout). Only fires for the
                  // active agent in training mode with ET enabled.
                  if (id === activeId) {
                    const training = useTrainingStore.getState();
                    const inTrainingMode =
                      useAgentStore.getState().directiveMode === 'training';
                    if (inTrainingMode && training.etEnabled) {
                      const capsuleBody = binder.getCapsuleBody();
                      if (capsuleBody?.isValid()) {
                        const t = capsuleBody.translation();
                        const y = t.y;
                        if (
                          y < training.healthyHeightMin ||
                          y > training.healthyHeightMax
                        ) {
                          maybeEmitEpisodeTermination('unhealthy_height');
                        }
                      }
                      const nowMs = performance.now();
                      const elapsedSec =
                        training.episodeStartTime > 0
                          ? (nowMs - training.episodeStartTime) / 1000
                          : 0;
                      if (
                        elapsedSec >= training.maxEpisodeSeconds
                      ) {
                        maybeEmitEpisodeTermination('timeout');
                      }
                    }
                  }
                } catch (error) {
                  Logger.warn(`HumanoidPhysicsBinder (${id}) sync failed:`, error);
                }
              }
            }
          }
        );

        setIsReady(true);
        // Mark rehydration/loading as complete so the startup modal will close.
        agentStore.setHasRehydrated(true);
        Logger.info('useWorld: Initialization complete.');
      } catch (error) {
        Logger.error("useWorld: Initialization failed", error);
        debouncedToast("physics-init-fail", () => {
          synthiaToast.error(STRINGS.TOASTS.RAPIER_LOAD_FAIL);
        });
      }
    };

    init();

    return () => {
      cancelled = true;
      stopMemoryMonitor();
      worldEngineRef.current?.stop();
      physicsEngineRef.current?.cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [containerRef]);

  useEffect(() => {
    if (physicsEngineRef.current) {
      physicsEngineRef.current.setGravity(worldStore.gravity);
    }
  }, [worldStore.gravity]);

  useEffect(() => {
    humanoidPhysicsBindersRef.current.forEach((binder) => {
      binder.friction = worldStore.globalFriction;
    });
    // Also update friction on all spawned objects
    if (objectManagerRef.current) {
      objectManagerRef.current.setGlobalFriction(worldStore.globalFriction);
    }
  }, [worldStore.globalFriction]);

  useEffect(() => {
    humanoidPhysicsBindersRef.current.forEach((binder) => {
      binder.setLerpSpeed(worldStore.movementSmoothing);
    });
  }, [worldStore.movementSmoothing]);

  useEffect(() => {
    humanoidPhysicsBindersRef.current.forEach((binder) => {
      binder.renderDebugSpheres(worldStore.showDebugJoints);
    });
  }, [worldStore.showDebugJoints]);

  // RMBS (reaction-mass balance) — opt-in assistive torque. The setter also
  // reconfigures the capsule-balance gains to a shock-absorber pair when on,
  // and restores Road-2 defaults when off.
  useEffect(() => {
    humanoidPhysicsBindersRef.current.forEach((binder) => {
      try {
        binder.setReactionMassEnabled(worldStore.reactionMassEnabled);
      } catch (err) {
        Logger.warn(`setReactionMassEnabled on ${binder.agentId} failed:`, err);
      }
    });
  }, [worldStore.reactionMassEnabled]);

  // Capsule balance (Road-2 root-balance corrector). Master switch; works
  // independently of RMBS so researchers can isolate which assist is active.
  useEffect(() => {
    humanoidPhysicsBindersRef.current.forEach((binder) => {
      try {
        binder.setCapsuleBalanceEnabled(worldStore.capsuleBalanceEnabled);
      } catch (err) {
        Logger.warn(`setCapsuleBalanceEnabled on ${binder.agentId} failed:`, err);
      }
    });
  }, [worldStore.capsuleBalanceEnabled]);

  // ET lifecycle: when both training mode and ET toggle are on and no
  // episode is in progress (episodeStartTime === 0), kick off episode #1.
  // This prevents the timeout check from firing immediately on first enable.
  // Subscribes to both trainingStore.etEnabled and agentStore.directiveMode.
  const etEnabledFlag = useTrainingStore((s) => s.etEnabled);
  const directiveModeForActive = useAgentStore((s) => s.directiveMode);
  useEffect(() => {
    if (!etEnabledFlag) return;
    if (directiveModeForActive !== 'training') return;
    const training = useTrainingStore.getState();
    if (training.episodeStartTime > 0) return;
    training.startNewEpisode(performance.now());
    Logger.info('[useWorld] ET enabled while in training mode — started episode 1');
  }, [etEnabledFlag, directiveModeForActive]);


  // Handle floor, grid, and sky state
  useEffect(() => {
    if (worldEngineRef.current) {
      worldEngineRef.current.updateFloor(worldStore.showFloor, worldStore.floorColor);
      worldEngineRef.current.updateGrid(worldStore.showGrid);
      worldEngineRef.current.updateSkyColor(worldStore.skyColor);
    }
  }, [worldStore.showFloor, worldStore.floorColor, worldStore.showGrid, worldStore.skyColor]);

  // Handle object renaming, physics update, and deletion
  useEffect(() => {
    const handleRename = (e: any) => {
      const { id, name } = e.detail;
      const activeObjManager = objectManagerRef.current;
      activeObjManager?.renameObject(id, name);
    };

    const handleUpdatePhysics = (e: any) => {
      const { id, updates } = e.detail;
      const activeObjManager = objectManagerRef.current;
      activeObjManager?.updateObjectPhysics(id, updates);
    };

    const handleDeleteObject = (e: any) => {
      const { id } = e.detail;
      const activeObjManager = objectManagerRef.current;
      // Detach TransformControls before removing the object to prevent
      // "not part of scene graph" errors from the gizmo tracking a removed object.
      worldEngineRef.current?.getCameraManager().attachTransform(null);
      activeObjManager?.deleteObject(id);
      if (useUIStore.getState().selectedEntityId === id) {
        useUIStore.getState().setSelectedEntityId(null);
      }
    };

    window.addEventListener('synthia:rename', handleRename);
    window.addEventListener('synthia:updatePhysics', handleUpdatePhysics);
    window.addEventListener('synthia:deleteObject', handleDeleteObject);
    return () => {
      window.removeEventListener('synthia:rename', handleRename);
      window.removeEventListener('synthia:updatePhysics', handleUpdatePhysics);
      window.removeEventListener('synthia:deleteObject', handleDeleteObject);
    };
  }, []);

  // Bug 4 Fix: Subscribe to camera mode and relay to CameraManager
  useEffect(() => {
    worldEngineRef.current?.getCameraManager().setMode(worldStore.cameraMode);
  }, [worldStore.cameraMode]);

  // Vision settings: relay the configurable AI perception FOV + render size
  // to the CameraManager whenever they change.
  useEffect(() => {
    worldEngineRef.current?.getCameraManager().setAIVisionConfig(worldStore.aiVisionFov, worldStore.aiVisionSize);
  }, [worldStore.aiVisionFov, worldStore.aiVisionSize]);

  // Instant Camera target snap on active agent change
  const activeAgentId = useAgentStore((state) => state.activeAgentId);
  const lastActiveAgentIdRef = useRef<string>('agent_0');

  useEffect(() => {
    if (!isReady || !worldEngineRef.current) return;
    if (activeAgentId !== lastActiveAgentIdRef.current) {
      lastActiveAgentIdRef.current = activeAgentId;

      if (useWorldStore.getState().cameraMode !== 'third_person') {
        const binder = humanoidPhysicsBindersRef.current.get(activeAgentId);
        if (binder) {
          const headTransform = binder.getHeadTransform();
          if (headTransform && headTransform.position) {
            const camManager = worldEngineRef.current.getCameraManager();
            camManager.getTransformControls().detach();
            camManager.getMainCamera().lookAt(headTransform.position);
            (camManager as any).controls.target.copy(headTransform.position);
            (camManager as any).controls.update();
            console.log(`[useWorld] Instantly snapped camera focus to selected agent: ${activeAgentId}`);
          }
        }
      }
    }
  }, [activeAgentId, isReady]);

  const findSpawnPosition = useCallback((skipHumanoidCheck = false): THREE.Vector3 => {
    const humanoidPos = new THREE.Vector3(0, 0, 5);
    const binder = humanoidPhysicsBindersRef.current.get(useAgentStore.getState().activeAgentId) || humanoidPhysicsBindersRef.current.get('agent_0');
    if (binder) {
      // Use capsule body position (pelvis) for more reliable location than headTransform
      const capsuleBody = binder.getCapsuleBody();
      if (capsuleBody && capsuleBody.isValid()) {
        const t = capsuleBody.translation();
        humanoidPos.set(t.x, 0, t.z);
      } else {
        const headTransform = binder.getHeadTransform();
        if (headTransform) {
          humanoidPos.set(headTransform.position.x, 0, headTransform.position.z);
        }
      }
    }

    const activeObjManager = objectManagerRef.current;
    const spawnRadius = 3.0;
    const spawnPos = new THREE.Vector3();
    let placed = false;

    for (let attempt = 0; attempt < 12; attempt++) {
      const angle = (attempt / 12) * Math.PI * 2;
      const candidateX = humanoidPos.x + Math.sin(angle) * spawnRadius;
      const candidateZ = humanoidPos.z + Math.cos(angle) * spawnRadius;
      const candidateY = 0.6;

      let overlaps = false;
      activeObjManager?.getObjects().forEach((obj) => {
        if (overlaps) return;
        const dx = obj.mesh.position.x - candidateX;
        const dz = obj.mesh.position.z - candidateZ;
        if (dx * dx + dz * dz < 1.44) overlaps = true; // 1.2m Euclidean distance
      });
      if (!skipHumanoidCheck) {
        const dhx = humanoidPos.x - candidateX;
        const dhz = humanoidPos.z - candidateZ;
        if (dhx * dhx + dhz * dhz < 1.0) overlaps = true; // 1.0m Euclidean distance from humanoid
      }

      if (!overlaps) {
        spawnPos.set(candidateX, candidateY, candidateZ);
        placed = true;
        break;
      }
    }

    if (!placed) {
      // Spiral outward to find a clear spot
      const fallbackAngle = Math.random() * Math.PI * 2;
      spawnPos.set(
        humanoidPos.x + Math.cos(fallbackAngle) * 5,
        0.6,
        humanoidPos.z + Math.sin(fallbackAngle) * 5
      );
    }
    return spawnPos;
  }, []);

  // Bug 3 Fix: Listen for object spawn events dispatched by ObjectSpawner UI
  useEffect(() => {
    const handleSpawnEvent = (e: Event) => {
      const { presetId } = (e as CustomEvent).detail;
      const activeObjManager = objectManagerRef.current;
      if (!activeObjManager) {
        Logger.error('useWorld: Object Manager is not ready for spawning');
        synthiaToast.error('Spawning failed: Object Manager is not ready');
        return;
      }
      const physicsEngine = physicsEngineRef.current;
      if (!physicsEngine) {
        Logger.error('useWorld: Physics engine ref is null');
        synthiaToast.error('Spawning failed: Physics engine not initialized');
        return;
      }
      if (!physicsEngine.isReady) {
        Logger.warn('useWorld: Physics engine not ready yet, retrying...');
        synthiaToast.warning('Physics engine is still loading, please wait...');
        return;
      }

      try {
        const spawnPos = findSpawnPosition();
        Logger.info(`useWorld: Spawning '${presetId}' at (${spawnPos.x.toFixed(2)}, ${spawnPos.y.toFixed(2)}, ${spawnPos.z.toFixed(2)})`);
        const obj = activeObjManager.spawnObject(presetId, spawnPos);
        if (obj) {
          Logger.info(`useWorld: Object '${presetId}' spawned successfully (id=${obj.id}). Total objects: ${activeObjManager.getObjects().size}`);
          synthiaToast.success(`${obj.name} spawned near the agent`);
        } else {
          const unclaimedIndex = (activeObjManager as any).slotClaimed.indexOf(false);
          if (unclaimedIndex < 0) {
            synthiaToast.error('Spawning failed: Primitive slot pool exhausted (20/20)');
          } else {
            Logger.error(`useWorld: spawnObject returned null for preset '${presetId}'`);
            synthiaToast.error(`Spawning failed: Could not create object for preset '${presetId}'`);
          }
        }
      } catch (err: any) {
        Logger.error(`useWorld: spawnObject threw an error`, err);
        synthiaToast.error(`Spawning failed: ${err.message || err}`);
      }
    };

    window.addEventListener('synthia:spawn', handleSpawnEvent);
    return () => window.removeEventListener('synthia:spawn', handleSpawnEvent);
  }, [findSpawnPosition]);

  useEffect(() => {
    const handleSpawnCustom = (e: Event) => {
      const { name, scene, isTerrain, skipCollision, processed } = (e as CustomEvent).detail as {
        name: string;
        scene: THREE.Group;
        isTerrain: boolean;
        skipCollision?: boolean;
        processed?: any;
      };
      const activeObjManager = objectManagerRef.current;
      if (!activeObjManager) {
        synthiaToast.error('Spawning failed: Object Manager is not ready');
        return;
      }
      const physicsEngine = physicsEngineRef.current;
      if (!physicsEngine || !physicsEngine.isReady) {
        synthiaToast.error('Spawning failed: Physics engine is not ready');
        return;
      }

      try {
        const box = new THREE.Box3().setFromObject(scene);
        const size = box.getSize(new THREE.Vector3());
        const spawnPos = findSpawnPosition(isTerrain);
        if (isTerrain) {
          spawnPos.y = -box.min.y;
        } else {
          spawnPos.y = Math.max(0.1, size.y / 2 + 0.01);
        }

        const obj = activeObjManager.spawnCustomModel(scene, name, spawnPos, { isTerrain, skipCollision, processed });
        if (obj) {
          Logger.info(`useWorld: Custom model '${name}' spawned (id=${obj.id})`);
          synthiaToast.success(`${name} spawned successfully`);
          window.dispatchEvent(new CustomEvent('synthia:spawnCustomComplete', { detail: { success: true, name } }));
        } else {
          synthiaToast.error(`Spawning failed: Could not create custom model '${name}'`);
          window.dispatchEvent(new CustomEvent('synthia:spawnCustomComplete', { detail: { success: false, name } }));
        }
      } catch (err: any) {
        Logger.error(`useWorld: spawnCustomModel threw an error`, err);
        synthiaToast.error(`Spawning failed: ${err.message || err}`);
        window.dispatchEvent(new CustomEvent('synthia:spawnCustomComplete', { detail: { success: false, name } }));
      }
    };

    window.addEventListener('synthia:spawnCustom', handleSpawnCustom);
    return () => window.removeEventListener('synthia:spawnCustom', handleSpawnCustom);
  }, [findSpawnPosition]);

  // Helpers to capture and build state/loops per agent
  const captureWorldStateForAgent = useCallback(async (agentId: string) => {
    if (!worldEngineRef.current || !audioEngineRef.current) return null;

    const binder = humanoidPhysicsBindersRef.current.get(agentId);
    if (!binder) return null;

    const renderer = worldEngineRef.current.getRenderer();
    const scene = worldEngineRef.current.getScene();
    const cameraManager = worldEngineRef.current.getCameraManager();
    const camera = worldEngineRef.current.getCamera();

    // Render main view
    renderer.render(scene, camera);

    // ── Per-agent first-person vision ─────────────────────────────
    // Reposition the AI perception camera to THIS agent's head transform,
    // render the 448x448 frame from it, then restore the active agent's view.
    // This makes every agent's perception genuinely first-person rather than
    // all agents receiving the active agent's frame.
    const headTransform = binder.getHeadTransform();
    const aiCamera = cameraManager.getHeadCamera();
    const prevPos = aiCamera.position.clone();
    const prevQuat = aiCamera.quaternion.clone();

    let rawFrame = '';
    if (headTransform && headTransform.position) {
      aiCamera.position.copy(headTransform.position);
      aiCamera.quaternion.copy(headTransform.quaternion);
      aiCamera.up.set(0, 1, 0);
      try {
        rawFrame = cameraManager.captureFrameFromCamera(scene, aiCamera);
      } catch (err) {
        Logger.warn(`captureWorldStateForAgent (${agentId}): per-agent frame render failed`, err);
      }
      // Restore the display/active-agent camera
      aiCamera.position.copy(prevPos);
      aiCamera.quaternion.copy(prevQuat);
      aiCamera.up.set(0, 1, 0);
      aiCamera.updateProjectionMatrix();
    }

    // Fallback: active-agent shared frame (matches legacy behavior)
    if (!rawFrame || rawFrame === '') {
      rawFrame = worldEngineRef.current.getLastAIFrame() || '';
    }
    if (!rawFrame || rawFrame === '') {
      Logger.warn(`captureWorldState (${agentId}): frame not yet available, skipping cycle`);
      return null;
    }

    const frame = rawFrame;
    const fileSize = (frame.length * 0.75) / 1024;
    (window as any)._synthia_connection_store_metrics?.({
      frameSize: fileSize,
    });

    const joints = binder.getJointState();

    let proprioception: any = null;
    if (binder.mbActive) {
      const obsBuilder = binder.getObservationBuilder();
      const capsuleBody = binder.getCapsuleBody();
      if (capsuleBody?.isValid()) {
        proprioception = obsBuilder.buildVLMProprioception(capsuleBody);
      }
    }

    const audioBuffer = await audioEngineRef.current.getBuffer();
    const audioPcm = audioBuffer ? btoa(String.fromCharCode(...new Uint8Array(audioBuffer.buffer))) : "";

    let contact_forces: Record<string, any> = {};
    if (useWorldStore.getState().bodyType === 'humanoid') {
      contact_forces = binder.getContactForces();
    }

    const activeObjManager = objectManagerRef.current;
    const objects = activeObjManager
      ? Array.from(activeObjManager.getObjects().values()).map((obj: any) => ({
          id: obj.id,
          type: obj.type,
          name: obj.name || obj.type,
          position: {
            x: obj.mesh?.position.x ?? 0,
            y: obj.mesh?.position.y ?? 0,
            z: obj.mesh?.position.z ?? 0
          },
          dimensions: obj.dimensions || { w: 1, h: 1, d: 1 },
          isStatic: obj.isStatic ?? true,
          interactionZones: obj.interactionZones?.map((z: any) => ({
            zoneId: z.id || z.zoneId,
            note: z.note,
            onContact: z.onContact
          })) || []
        }))
      : [];

    const uprightPreset = binder.getUprightPreset();
    const isGrounded = binder.getIsGrounded();

    const capsuleBody = binder.getCapsuleBody();
    const rootState = capsuleBody?.isValid()
      ? (() => {
          const translation = capsuleBody.translation();
          const rotation = capsuleBody.rotation();
          const linearVelocity = capsuleBody.linvel();
          const angularVelocity = capsuleBody.angvel();
          return {
            position: [translation.x, translation.y, translation.z],
            orientation: [rotation.x, rotation.y, rotation.z, rotation.w],
            linear_velocity: [linearVelocity.x, linearVelocity.y, linearVelocity.z],
            angular_velocity: [angularVelocity.x, angularVelocity.y, angularVelocity.z],
          };
        })()
      : null;

    const jointVelocities = binder.mbActive
      ? Object.fromEntries(
          Array.from(binder.getObservationBuilder().getJointVelocities().entries())
        )
      : {};

    const agentState = useAgentStore.getState().agents[agentId] || { heartbeat: 0, currentRung: 0, currentGoal: '' };

    // ── Overheard Speech (Agent-to-Agent text tunnel) ────────────────────
    const MAX_HEARING_DISTANCE = 15; // 15 meters
    const UTTERANCE_EXPIRY_MS = 10000; // 10 seconds expiry

    // Clean up expired utterances
    useSpeechStore.getState().clearExpiredUtterances(UTTERANCE_EXPIRY_MS);

    const activeUtterances = useSpeechStore.getState().utterances || [];
    const listenerHeadTransform = binder.getHeadTransform();
    const listenerHeadPos = listenerHeadTransform?.position || new THREE.Vector3(0, 1.6, 0);

    const overheard: Array<{ speakerId: string; distance: number; occluded: boolean; text: string }> = [];

    for (const u of activeUtterances) {
      if (u.speakerId === agentId) continue;
      if (u.deliveredTo.includes(agentId)) continue;

      const speakerPosVec = new THREE.Vector3(u.position.x, u.position.y, u.position.z);
      const distance = listenerHeadPos.distanceTo(speakerPosVec);

      if (distance <= MAX_HEARING_DISTANCE) {
        let occluded = false;
        const currentScene = worldEngineRef.current?.getScene();
        if (currentScene) {
          const raycaster = new THREE.Raycaster();
          const direction = new THREE.Vector3().subVectors(speakerPosVec, listenerHeadPos).normalize();
          raycaster.set(listenerHeadPos, direction);
          raycaster.far = distance;

          const intersects = raycaster.intersectObjects(currentScene.children, true);
          const speakerBinder = humanoidPhysicsBindersRef.current.get(u.speakerId);
          const speakerModelRoot = speakerBinder?.getModelRoot();
          const listenerModelRoot = binder.getModelRoot();
          const floorMesh = (window as any).__SYNTHIA_FLOOR_MESH__;

          occluded = intersects.some((intersect) => {
            let p: THREE.Object3D | null = intersect.object;
            while (p) {
              if (p === speakerModelRoot || p === listenerModelRoot || p === floorMesh) {
                return false; // ignore self, speaker, and floor mesh
              }
              p = p.parent;
            }
            return true; // it's a real obstacle in between!
          });
        }

        overheard.push({
          speakerId: u.speakerId,
          distance,
          occluded,
          text: u.text,
        });

        // Mark this utterance as delivered to this listener agentId
        useSpeechStore.getState().markUtteranceDelivered(u.id, agentId);
      }
    }

    return {
      frame,
      joints,
      proprioception,
      audio_pcm: audioPcm,
      contact_forces,
      objects,
      uprightPreset,
      isGrounded,
      joint_velocities: jointVelocities,
      root_state: rootState,
      heartbeat: agentState.heartbeat,
      currentRung: agentState.currentRung,
      bodyType: useWorldStore.getState().bodyType,
      currentGoal: agentState.currentGoal,
      lightState: useWorldStore.getState().lightState,
      timestamp: Date.now(),
      overheard_speech: overheard,
    };
  }, []);

  const generateCombinedMCF = useCallback(() => {
    const agentsList: any[] = [];
    for (const [id, binder] of humanoidPhysicsBindersRef.current.entries()) {
      // Fix 3 (fallback): For any binder whose capsuleCenterY is still 0 (legacy spawned
      // agents before ensureCapsuleGeometry was added), compute a safe value from modelHeight.
      const capsuleCenterY = binder.getCapsuleCenterY() || (binder as any).modelHeight / 2;

      // Sync the boneInfoMap's bindWorldPosition for this agent to its current capsule world
      // position. This ensures the MJCF places the agent's body tree at the correct location
      // in the new world without drifting from their original spawn X/Z or mixing with
      // another agent's bone positions. bindWorldQuaternion is intentionally NOT updated —
      // it stays as the immutable T-pose so the MJCF always bakes T-pose joint structure.
      binder.syncBindWorldPositionsFromPhysics();

      const boneInfoMap = binder.getBoneInfoMap();
      console.log(`[MCF_GEN] Agent ${id}: prefix=${binder.prefix}, bones=${boneInfoMap.size}, capsuleCenterY=${capsuleCenterY.toFixed(3)}`);
      agentsList.push({
        prefix: binder.prefix,
        boneInfoMap,
        capsuleCenterY,
      });
      void id;
    }

    const customSpecs = objectManagerRef.current ? (objectManagerRef.current as any).customMeshesSpec : [];
    return generateCombinedMultiAgentMJCF(agentsList, customSpecs);
  }, []);

  const startAgentClientLoop = useCallback((agentId: string) => {
    if (activeAgentLoopsRef.current.has(agentId)) {
      activeAgentLoopsRef.current.get(agentId)!.stop();
    }

    const connStore = useConnectionStore.getState();
    const runtime = useAgentRuntimeStore.getState().getConfig(agentId);
    const loop = new AgentLoop({
      agentId,
      cycleMs: runtime.cycleMs || connStore.cycleMs || 2000,
      supabaseUrl: runtime.supabaseUrl || connStore.supabaseUrl,
      supabaseKey: runtime.supabaseKey || connStore.supabaseKey,
      captureWorldState: async () => {
        return await captureWorldStateForAgent(agentId);
      }
    });

    loop.setProvider(runtime.provider, runtime.endpoint, runtime.apiKey, runtime.model);
    loop.start().catch((err) => Logger.error(`[AgentLoop (${agentId})] Failed to start client loop`, err));

    activeAgentLoopsRef.current.set(agentId, loop);
    console.log(`[useWorld] Started client-side cognitive loop for ${agentId} (provider=${runtime.provider})`);
  }, [captureWorldStateForAgent]);

  const pauseAgentClientLoop = useCallback((agentId: string) => {
    const loop = activeAgentLoopsRef.current.get(agentId);
    if (loop) {
      loop.pause();
      console.log(`[useWorld] Paused client-side cognitive loop for ${agentId}`);
    }
  }, []);

  const resumeAgentClientLoop = useCallback((agentId: string) => {
    const loop = activeAgentLoopsRef.current.get(agentId);
    if (loop) {
      loop.resume();
      console.log(`[useWorld] Resumed client-side cognitive loop for ${agentId}`);
    }
  }, []);

  const sleepAllAgents = useCallback(() => {
    let count = 0;
    for (const [id, loop] of activeAgentLoopsRef.current.entries()) {
      loop.pause();
      count++;
      console.log(`[useWorld] sleepAllAgents: paused ${id}`);
    }
    return count;
  }, []);

  const resumeAllAgents = useCallback(() => {
    let count = 0;
    for (const [id, loop] of activeAgentLoopsRef.current.entries()) {
      loop.resume();
      count++;
      console.log(`[useWorld] resumeAllAgents: resumed ${id}`);
    }
    return count;
  }, []);

  // Expose pause/resume on window for settings modal access
  useEffect(() => {
    window.__synthia = {
      ...(window.__synthia || {}),
      pauseAgent: pauseAgentClientLoop,
      resumeAgent: resumeAgentClientLoop,
      sleepAllAgents,
      resumeAllAgents,
      updateAgentProvider: (agentId: string, provider: string, endpoint: string, apiKey?: string, model?: string) => {
        const loop = activeAgentLoopsRef.current.get(agentId);
        if (loop) {
          loop.setProvider(provider, endpoint, apiKey, model);
          console.log(`[useWorld] Updated provider for ${agentId}: ${provider} -> ${endpoint}`);
        }
      },
      manualIdentityUpdate: async (agentId: string, update: any, reason: string) => {
        const loop = activeAgentLoopsRef.current.get(agentId);
        if (!loop) return { ok: false, error: 'Agent loop not found' };
        return loop.manualIdentityUpdate(update, reason);
      },
    };
  }, [pauseAgentClientLoop, resumeAgentClientLoop, sleepAllAgents, resumeAllAgents]);

  const spawnAgent = useCallback(async () => {
    if (isSpawningRef.current) {
      Logger.info("useWorld: spawnAgent - Spawn already in progress, ignoring duplicate request");
      return null;
    }
    if (!worldEngineRef.current || !physicsEngineRef.current) return null;

    isSpawningRef.current = true;
    try {
      const physicsEngine = physicsEngineRef.current;
      const scene = worldEngineRef.current.getScene();

      // Reset counter to 0 when no agents are active so the first spawn
      // of any fresh session always lands at origin as agent_0.
      if (activeAgentLoopsRef.current.size === 0) {
        spawnIndexRef.current = 0;
        sessionStorage.removeItem('synthia_spawn_index');
      }

      const offsetIndex = spawnIndexRef.current;
      spawnIndexRef.current = offsetIndex + 1;
      sessionStorage.setItem('synthia_spawn_index', String(spawnIndexRef.current));
      const agentId = `agent_${offsetIndex}`;

      // Linear offset spaced 1.75 meters apart
      let spawnX = 0;
      if (offsetIndex === 1) spawnX = 1.75;
      else if (offsetIndex === 2) spawnX = -1.75;
      else if (offsetIndex > 2) {
        spawnX = offsetIndex % 2 === 1 ? 1.75 * Math.ceil(offsetIndex / 2) : -1.75 * (offsetIndex / 2);
      }
      const spawnPoint = new THREE.Vector3(spawnX, 0, 0);

      const binder = new HumanoidPhysicsBinder(physicsEngine, scene, agentId);

      // STEP A: Load model bind pose
      const probePoint = new THREE.Vector3(0, 0, 0);
      const stepA = await binder.loadAndVisualizeBindPose(probePoint);
      if (!stepA) {
        Logger.error(`useWorld: spawnAgent - STEP A failed for ${agentId}`);
        return null;
      }

      // Fix 1: Set capsuleCenterY BEFORE adding to the map / before generateCombinedMCF.
      // Without this, capsuleCenterY stays 0, causing the root capsule to be placed at
      // floor level and catapulting the agent skyward on first contact resolution.
      binder.ensureCapsuleGeometry();

      binder.repositionModel(spawnPoint.x, spawnPoint.y, spawnPoint.z);

      binder.friction = worldStore.globalFriction;
      binder.setLerpSpeed(worldStore.movementSmoothing);
      binder.renderDebugSpheres(worldStore.showDebugJoints);
      // NOTE: setMode is NOT called here for the new binder — it is called inside
      // the per-binder loop below (Fix 3), gated to the new binder only.

      humanoidPhysicsBindersRef.current.set(agentId, binder);
      (window as any).__SYNTHIA_HUMANOID_BINDER__ = binder; // keep active
      (window as any).__SYNTHIA_HUMANOID_BINDERS__ = humanoidPhysicsBindersRef.current;

      // Rebuild physics world
      physicsEngine.setMutating(true);
      physicsEngine.setReady(false);

      try {
        const existingAgentIds = Array.from(humanoidPhysicsBindersRef.current.keys()).filter(id => id !== agentId);
        const objectsList = objectManagerRef.current ? Array.from(objectManagerRef.current.getObjects().values()) : [];
        const capturedState = StateRehydrator.capture(physicsEngine, existingAgentIds, objectsList);

        const baseXml = generateCombinedMCF();
        physicsEngine.loadMJCFModel(baseXml);
        physicsEngine.setReady(true);

        for (const [id, activeBinder] of humanoidPhysicsBindersRef.current.entries()) {
          const bm = activeBinder.getMultiBodyManager();
          bm.remapIdsAgainstLoadedWorld(activeBinder.getBoneInfoMap());
          activeBinder.initMotorController();

          if (id === agentId) {
            // Fix 3: Only the NEW agent gets the full pose reset (createJointsWithZeroMotors,
            // activateMotors, setMode/resetToBindPose). Old agents must NOT have their
            // currentTargets wiped or their ctrl ramp restarted — doing so causes the
            // "arms crossed behind back" degradation on every spawn.
            await activeBinder.createJointsWithZeroMotors();
            await activeBinder.activateMotorsWithStiffnessAndDamping(80, 10);
            activeBinder.deactivateMultiBody();
            if (worldStore.useMultiBodyPD) {
              await activeBinder.activateMultiBody();
            }
            activeBinder.setMode('rigid');
          } else {
            // Old binders: re-bind their multi-body proxies to the new world IDs without
            // touching currentTargets, ramp, or ctrl.
            activeBinder.deactivateMultiBody();
            if (worldStore.useMultiBodyPD) {
              await activeBinder.activateMultiBody();
            }
          }
        }

        StateRehydrator.restore(physicsEngine, capturedState, objectsList);

        // Fix 4: Re-arm spawn grounding ONLY for the NEW agent.
        // Existing agents must NOT have targetSpawnGrounded re-armed — doing so causes
        // syncVisuals() to run a grounding pass against stale Three.js bone positions
        // (the skeleton hasn't re-rendered yet after MuJoCo world recompile). The stale
        // positions produce an incorrect vertical delta, setCapsulePosition() teleports the
        // existing capsule into the floor, and MuJoCo's contact solver catapults it skyward.
        const newBinder = humanoidPhysicsBindersRef.current.get(agentId);
        if (newBinder) {
          (newBinder as any).targetSpawnGrounded = true;
          (newBinder as any).previousFootPositions?.clear();
          // RMBS stability-pass gate: `?rmbs=1` pre-enables the reaction-mass
          // balance system on this agent (dev diagnostic; no behavior change
          // without the flag). `?t2=1` additionally schedules the 0.4 rad
          // forward-leg perturbation once the agent has settled.
          const qp = new URLSearchParams(window.location.search);
          if (qp.has('rmbs')) {
            newBinder.setReactionMassEnabled(true);
            console.info('[DIAG] RMBS pre-enabled via ?rmbs=1 (Road-5.1 diagnostic)');
          }
          if (qp.has('t2')) {
            setTimeout(() => {
              try {
                window.dispatchEvent(new CustomEvent('synthia:action', {
                  detail: {
                    jointOverrides: { mixamorigleftupleg: 0.4 },
                    agentId,
                    activeGaitPhase: false,
                  },
                } as any));
                console.info('[T2] 0.4 rad forward-leg perturbation dispatched');
              } catch (err) {
                console.warn('[T2] dispatch failed', err);
              }
            }, 6000);
          }
        }

        const newAgentCapsule = binder.getCapsuleBody();
        if (newAgentCapsule && newAgentCapsule.isValid()) {
          binder.setCapsulePosition(spawnPoint.x, spawnPoint.y, spawnPoint.z);
          binder.resetPose(spawnPoint);
        }
      } catch (err) {
        Logger.error(`useWorld: spawnAgent - physics rebuild failed:`, err);
      } finally {
        physicsEngine.setMutating(false);
      }

      // Add agent to Zustand state
      const { addAgent } = useAgentStore.getState() as any;
      if (addAgent) {
        addAgent(agentId);
      }

      startAgentClientLoop(agentId);
      Logger.info(`useWorld: Spawned agent ${agentId} at X=${spawnX}`);
      return binder;
    } catch (err) {
      Logger.error(`useWorld: spawnAgent - outer spawn failed:`, err);
      return null;
    } finally {
      isSpawningRef.current = false;
    }
  }, [worldStore, generateCombinedMCF, startAgentClientLoop]);

  // Sync window generators
  useEffect(() => {
    (window as any).__SYNTHIA_GENERATE_COMBINED_MJCF__ = generateCombinedMCF;
    (window as any).synthia = {
      spawnAgent: async () => {
        return await spawnAgent();
      },
      getActiveAgentId: () => {
        return useAgentStore.getState().activeAgentId;
      },
      setActiveAgent: (id: string) => {
        useAgentStore.getState().setActiveAgentId(id);
      }
    };
    return () => {
      delete (window as any).__SYNTHIA_GENERATE_COMBINED_MJCF__;
      delete (window as any).synthia;
    };
  }, [generateCombinedMCF, spawnAgent]);

  // Sync connection store changes with active loops.
  // Per-agent runtime overrides (agentRuntimeStore) take precedence: loops with
  // an override for a given field keep their own value; everything else follows global.
  const connStore = useConnectionStore();
  const agentRuntimeConfigs = useAgentRuntimeStore((s) => s.configs);
  useEffect(() => {
    activeAgentLoopsRef.current.forEach((loop, agentId) => {
      const rt = useAgentRuntimeStore.getState();
      const runtime = rt.getConfig(agentId);
      const has = (key: any) => rt.hasOverride(agentId, key);

      const provider = has('provider') ? runtime.provider : connStore.provider;
      const endpoint = has('endpoint') ? runtime.endpoint : connStore.inferenceEndpoint;
      const apiKey = has('apiKey') ? runtime.apiKey : connStore.providerApiKey;
      const model = has('model') ? runtime.model : connStore.providerModel;
      loop.setProvider(provider, endpoint, apiKey, model);

      const supabaseUrl = has('supabaseUrl') ? runtime.supabaseUrl : connStore.supabaseUrl;
      const supabaseKey = has('supabaseKey') ? runtime.supabaseKey : connStore.supabaseKey;
      loop.updateSupabase(supabaseUrl, supabaseKey);

      const cycleMs = has('cycleMs') ? runtime.cycleMs : connStore.cycleMs;
      loop.setCycleMs(cycleMs || 2000);
    });
  }, [
    agentRuntimeConfigs,
    connStore.provider,
    connStore.inferenceEndpoint,
    connStore.providerApiKey,
    connStore.providerModel,
    connStore.supabaseUrl,
    connStore.supabaseKey,
    connStore.cycleMs,
  ]);

  useEffect(() => {
    const handlePush = (e: any) => {
      const { partName, impulse, agentId = 'agent_0' } = e.detail;
      const binder = humanoidPhysicsBindersRef.current.get(agentId);
      if (worldStore.bodyType === 'humanoid' && binder) {
        binder.push(partName, new THREE.Vector3(impulse.x, impulse.y, impulse.z));
      }
    };

    window.addEventListener('synthia:push', handlePush);
    return () => window.removeEventListener('synthia:push', handlePush);
  }, [worldStore.bodyType]);

  useEffect(() => {
    const handleAction = (e: any) => {
      const { jointOverrides, programSequence, sequence, activeGaitPhase, agentId = 'agent_0' } = e.detail;
      const binder = humanoidPhysicsBindersRef.current.get(agentId) as any;
      if (!binder) return;

      binder['timelineQueue'] = [];
      binder['timelineSequenceStart'] = null;

      Logger.info(`[ACTION_PIPELINE] useWorld handling action for ${agentId}: jointOverrides=${Object.keys(jointOverrides || {}).length} keys, sequence=${Array.isArray(sequence) ? sequence.length : 0}`);

      if (worldStore.bodyType === 'humanoid') {
        binder.setGaitActive(!!activeGaitPhase);
        try {
          const skeleton = binder['skeleton'];

          if (Array.isArray(sequence) && sequence.length > 0) {
            const validation = binder.validateAndApplyTimeline(skeleton, sequence, { activeGaitPhase: !!activeGaitPhase });
            for (const f of validation.appliedTimeline) {
              if (f.timeOffsetMs === 0) {
                binder.setMotorTargets(f.overrides as any);
              }
            }

            const loop = activeAgentLoopsRef.current.get(agentId);
            if (loop && validation.rejections.length > 0) {
              loop.recordActionFeedback(validation.rejections);
            }
          } else {
            const seq = [{ timeOffsetMs: 0, overrides: jointOverrides || {} }];
            const validation = binder.validateAndApplyTimeline(skeleton, seq, { activeGaitPhase: false });
            for (const f of validation.appliedTimeline) {
              if (f.timeOffsetMs === 0) binder.setMotorTargets(f.overrides as any);
            }

            const loop = activeAgentLoopsRef.current.get(agentId);
            if (loop && validation.rejections.length > 0) {
              loop.recordActionFeedback(validation.rejections);
            }
          }

          if (programSequence && Array.isArray(programSequence) && programSequence.length > 0) {
            binder.executeProgramSequence(programSequence);
          }
        } catch (err) {
          Logger.warn(`Action validation failed for ${agentId}`, err);
        }
      }
    };

    window.addEventListener('synthia:action', handleAction);
    return () => window.removeEventListener('synthia:action', handleAction);
  }, [worldStore.bodyType]);

  // ── Reset Pose Event Handler ─────────────────────────────────────────
  useEffect(() => {
    const handleResetPose = (e: Event) => {
      const targetId = (e as CustomEvent)?.detail?.agentId;
      humanoidPhysicsBindersRef.current.forEach((binder, id) => {
        if (targetId && id !== targetId) return;
        // Stand up in-place at the model's current XZ, don't teleport to origin.
        const capsuleBody = binder.getCapsuleBody();
        let x = 0;
        let z = 0;
        if (capsuleBody?.isValid()) {
          const t = capsuleBody.translation();
          x = t.x;
          z = t.z;
        }
        binder.resetPose(new THREE.Vector3(x, 0, z));
      });
    };
    window.addEventListener('synthia:resetPose', handleResetPose);
    return () => window.removeEventListener('synthia:resetPose', handleResetPose);
  }, []);

  // ── Agent Spoke perception event handler ─────────────────────────────
  useEffect(() => {
    const handleAgentSpoke = (e: Event) => {
      const { agentId, text } = (e as CustomEvent).detail;
      const binder = humanoidPhysicsBindersRef.current.get(agentId);
      const position = new THREE.Vector3(0, 1.6, 0); // fallback

      if (binder) {
        const headTransform = binder.getHeadTransform();
        if (headTransform && headTransform.position) {
          position.copy(headTransform.position);
        } else {
          const capBody = binder.getCapsuleBody();
          if (capBody && capBody.isValid()) {
            const t = capBody.translation();
            position.set(t.x, t.y + 1.6, t.z);
          }
        }
      }

      useSpeechStore.getState().addUtterance({
        id: Math.random().toString(36).substr(2, 9),
        speakerId: agentId,
        text,
        position: { x: position.x, y: position.y, z: position.z },
        timestamp: Date.now(),
        deliveredTo: [],
      });
    };

    window.addEventListener('synthia:agent_spoke', handleAgentSpoke);
    return () => window.removeEventListener('synthia:agent_spoke', handleAgentSpoke);
  }, []);

  // ── Agent-Specific Body Mode & Multi-Body PD Event Handlers ─────────
  useEffect(() => {
    const handleSetBodyMode = (e: Event) => {
      const { agentId, mode } = (e as CustomEvent).detail;
      const binder = humanoidPhysicsBindersRef.current.get(agentId);
      if (binder) {
        binder.setMode(mode);
      }
    };
    const handleToggleMultiBodyPD = (e: Event) => {
      const { agentId, enable } = (e as CustomEvent).detail;
      const binder = humanoidPhysicsBindersRef.current.get(agentId);
      if (binder) {
        if (enable) {
          binder.activateMultiBody();
        } else {
          binder.deactivateMultiBody();
        }
      }
    };
    window.addEventListener('synthia:setBodyMode', handleSetBodyMode);
    window.addEventListener('synthia:toggleMultiBodyPD', handleToggleMultiBodyPD);
    return () => {
      window.removeEventListener('synthia:setBodyMode', handleSetBodyMode);
      window.removeEventListener('synthia:toggleMultiBodyPD', handleToggleMultiBodyPD);
    };
  }, []);

  // ── Root Motion Event Handler ────────────────────────────────────────
  useEffect(() => {
    const handleRootMotion = (e: Event) => {
      const detail = (e as CustomEvent).detail || {};
      const { dx = 0, dz = 0, agentId = 'agent_0' } = detail;
      if (worldStore.bodyType !== 'humanoid') return;
      const binder = humanoidPhysicsBindersRef.current.get(agentId);
      if (!binder) return;
      const capsuleBody = binder.getCapsuleBody();
      if (!capsuleBody || !capsuleBody.isValid()) return;
      const t = capsuleBody.translation();
      capsuleBody.setTranslation({ x: t.x + dx, y: t.y, z: t.z + dz }, true);
    };
    window.addEventListener('synthia:rootMotion', handleRootMotion);
    return () => { window.removeEventListener('synthia:rootMotion', handleRootMotion); };
  }, [worldStore.bodyType]);

  useEffect(() => {
    if (isReady && humanoidPhysicsBindersRef.current.size === 0) {
      Logger.info("useWorld: Auto-spawning initial agent (agent_0) on load via spawnAgent()");
      spawnAgent();
    }
  }, [isReady, spawnAgent]);

  useEffect(() => {
    if (!isReady) return;

    // Day/night cycle removed — lighting stays bright permanently
  }, [isReady]);

  useEffect(() => {
    if (!worldEngineRef.current) return;
    // Set bright lighting immediately — no day/night transitions
    worldEngineRef.current.updateLighting('day', 1);
  }, []);

  // Escape to deselect + Delete to remove selected object
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === 'Escape') {
        useUIStore.getState().setSelectedEntityId(null);
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        const selectedId = useUIStore.getState().selectedEntityId;
        if (selectedId && window.confirm('Delete selected object? This cannot be undone.')) {
          window.dispatchEvent(new CustomEvent('synthia:deleteObject', { detail: { id: selectedId } }));
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const captureWorldState = useCallback(async () => {
    const activeId = useAgentStore.getState().activeAgentId || 'agent_0';
    return await captureWorldStateForAgent(activeId);
  }, [captureWorldStateForAgent]);

  const detectOutcomes = useCallback(() => {
    const outcomes = [...pendingOutcomesRef.current];
    pendingOutcomesRef.current = [];

    if (useAgentStore.getState().status === "falling") {
      outcomes.push({
        type: "outcome",
        data: { success: false, reward: -1.0, description: "Agent fell" },
        agentId: useAgentStore.getState().activeAgentId || 'agent_0',
      });
    }

    // Route outcomes into the correct agent's client-side cognitive loop so it can
    // finalize its pending memory cycle with the correct reward/outcome. Previously
    // this went to the coordinator's server loop via WorldViewport. Now each outcome
    // carries an agentId stamped at collection time (via ObjectManager.extractAgentIdFromPair).
    if (outcomes.length > 0) {
      for (const outcome of outcomes) {
        const targetId = outcome.agentId || useAgentStore.getState().activeAgentId || 'agent_0';
        const loop = activeAgentLoopsRef.current.get(targetId);
        if (loop) {
          loop.handleOutcome(outcome).catch((err) =>
            Logger.warn(`[useWorld] outcome routing to ${targetId} failed`, err)
          );
        } else {
          Logger.warn(`[useWorld] no loop found for agent ${targetId}, defaulting to active`);
          const activeId = useAgentStore.getState().activeAgentId || 'agent_0';
          const activeLoop = activeAgentLoopsRef.current.get(activeId);
          if (activeLoop) {
            activeLoop.handleOutcome(outcome).catch((e) =>
              Logger.warn(`[useWorld] outcome routing to active ${activeId} failed`, e)
            );
          }
        }
      }
    }

    return outcomes;
  }, []);

  return {
    isReady,
    getRagdoll: () => null,
    spawnObject: (presetId: string, pos: THREE.Vector3) => {
      const activeObjManager = objectManagerRef.current;
      return activeObjManager?.spawnObject(presetId, pos) || null;
    },
    deleteObject: (id: string) => {
      const activeObjManager = objectManagerRef.current;
      activeObjManager?.deleteObject(id);
    },
    push: (partName: string, impulse: THREE.Vector3) => {
      const activeId = useAgentStore.getState().activeAgentId || 'agent_0';
      const binder = humanoidPhysicsBindersRef.current.get(activeId);
      if (worldStore.bodyType === 'humanoid' && binder) {
        binder.push(partName, impulse);
      }
    },
    captureWorldState,
    detectOutcomes,
    pauseAgentClientLoop,
    resumeAgentClientLoop,
    sleepAllAgents,
    resumeAllAgents,
  };
};

// Expose pause/resume globally so settings modal can access them
declare global {
  interface Window {
    __synthia?: {
      pauseAgent?: (agentId: string) => void;
      resumeAgent?: (agentId: string) => void;
      sleepAllAgents?: () => number;
      resumeAllAgents?: () => number;
      updateAgentProvider?: (agentId: string, provider: string, endpoint: string, apiKey?: string, model?: string) => void;
      manualIdentityUpdate?: (agentId: string, update: any, reason: string) => Promise<{ ok: boolean; error?: string }>;
    };
  }
}
