Balancing a Language-Model-Directed Humanoid in the Browser: A Layered Real-Time Control Architecture, Failure-Mode Analysis, and a Validated Bridge to Robot-Learning Tooling

Greatness Okorie
Bells University of Technology, Ota, Ogun State, Nigeria
okorie.greatness@bellsuniversity.edu.ng
ORCID: 0009-0007-0940-123X

# Abstract

Large language models can issue semantic motor commands, but they are far too slow to keep a legged body upright. Something else has to handle balance. This report describes that something else: a four-layer, two-rate real-time controller for an approximately 80-DoF humanoid running entirely in a browser on MuJoCo compiled to WebAssembly. The layers are a root-capsule orientation torque balancer, per-joint PD servoing across 49 actuated joints, a physical 18 kg reaction mass sliding on two perpendicular rails, and a center-of-mass lean-and-step recovery layer built around the standard capture-point construction from the bipedal locomotion literature.

Two failure modes showed up during development and are still unresolved. First, a balance-authority flag exists in the code but is never actually read, so the root balancer fights commanded motion at full strength instead of yielding to it. Second, the stepping-recovery layer saturates its corrective lean before the swing leg can clear the ground. Both are reported at the mechanism level, with exact gains and code identifiers, not softened into a generic limitations paragraph. The only claim we treat as fully verified is narrower than either of those: trajectories recorded by the system, whether the balance controller succeeds or fails on a given episode, export cleanly into the Hugging Face `lerobot` schema and load into standard imitation-learning dataloaders without modification.

# 1. Introduction

Recent work has shown that large vision-language models can serve as the high-level reasoning core of a robot policy [1, 2, 3, 4]. Zawalski et al. [4] demonstrated that forcing such a model to produce step-by-step, visually grounded reasoning improves generalization on manipulation tasks. Most of that line of work assumes either that a competent low-level controller already exists, or that the model itself can be trained to emit actions at a rate the robot can actually use.

That assumption gets uncomfortable once the body in question is a free-standing humanoid instead of a fixed-base arm. An 80-DoF body that receives the instruction "step forward" still has to generate joint torques fast enough not to fall over while it figures out what "step forward" even means in terms of torque. Network LLM inference cannot run anywhere near that rate. The gap between semantic intent and physical stability is what this report is about.

The system, Synthia [17], runs the entire loop in a browser tab. Physics is MuJoCo compiled to WebAssembly, stepped at a fixed 500 Hz (2 ms timestep, implicit-fast integrator, 200 solver iterations). Rendering runs separately at 60 Hz. The browser deployment itself is not the contribution here, a companion report covers that question directly, so we mention it only briefly in Section 3. This paper is about the control architecture, the two failure modes that turned up while building it, and one compatibility result we've actually verified against existing tooling.

Contributions:

1. A four-layer, two-rate balance architecture that includes a physical reaction-mass system, along with the interface that lets a slow external LLM set motor intent without that intent overwriting the layers keeping the body upright.
2. A mechanistic account of two unresolved failure modes, including the exact gains and the specific code paths involved, offered as a diagnostic reference for anyone building something similar.
3. Confirmation that trajectories produced by the system load into the official Hugging Face `lerobot` library without modification, regardless of whether the episode that produced them was a balance success or a fall.

# 2. Related Work

**Balance and push-recovery control for legged robots.** Capture-point methods remain the standard reference for bipedal push recovery. Pratt et al. [5] introduced the construction $\text{CP} = x_{\text{COM}} + \dot{x}_{\text{COM}} \cdot \tau$, with $\tau = \sqrt{h/g}$ for pendulum height $h$, the point on the ground where a biped must step to bring its center of mass to rest. Later work turned this into a feedback law with proven exponential stability [6], and into model-predictive controllers that respect zero-moment-point constraints during recovery [7]. Our system's diagnostic recorder computes the same quantities, tau and capture point, from live center-of-mass state, so the link to this literature is direct rather than metaphorical. Section 5.2 covers where our simplified, heuristic recovery layer diverges from a full capture-point controller, and why that divergence currently fails.

**Physics simulation for robot learning.** MuJoCo [9] is the standard engine for contact-rich rigid-body control research and is what this work compiles to WebAssembly. PyBullet [10], Isaac Sim [11], Habitat [12], and PyRep [13] occupy adjacent roles in the simulation landscape, generally requiring local installation and, for the higher-fidelity options, a dedicated GPU.

**Vision-language-action models and embodied reasoning.** RT-1 [14] and RT-2 [1] showed that transformer policies fine-tuned on internet-scale vision-language data can control real robot arms from natural language instructions. OpenVLA [2] provided a fully open-source vision-language-action model on a Llama-2 backbone, which Zawalski et al. [4] extended with embodied chain-of-thought reasoning, interleaving sub-task planning with visually grounded predictions before the model commits to an action. Our system differs on two counts: the embodiment is a free-standing humanoid rather than a fixed-base arm, and the language model only sets high-level intent, a separate, conventionally engineered control stack, not the model itself, is what keeps the body upright.

**Standardized robot-learning data.** Open X-Embodiment [15] and DROID [16] aggregate large-scale real-world manipulation trajectories, and the Hugging Face LeRobot project [8] has become a widely used schema for organizing such trajectories for imitation learning. Section 6 reports a direct compatibility test against it.

# 3. System Overview

Synthia instantiates this architecture on a single embodiment: an articulated humanoid with 49 actuated joints (52 total, counting the 6-DOF root free joint and two inert toe-base passthroughs that carry no actuator), simulated with MuJoCo-on-WebAssembly and rendered with Three.js/WebGL, entirely client-side, no local installation step. The agent loop assembles an observation from physics state, proprioception, and an offscreen 448x448 perception render, sends it to an external, user-configured LLM over its provider API, parses the returned motor intent and clamps joint angles to $\pm\pi$, then hands that intent to the control architecture in Section 4. Every resulting transition is logged as a structured trajectory point for later export (Section 6). Multiple agents can share one physics world through a prefixed-namespace MJCF scheme (`agent_0_...`, `agent_1_...`); neither that multi-agent capability nor the scripted-controller fallback mode is examined here.

# 4. Layered Balance Control Architecture

Control runs at two rates. All four layers below execute at 500 Hz, inside the physics loop. A separate 60 Hz loop handles rendering and pose flushing. The split exists for a fairly blunt reason: balance is a fast problem, a 60 Hz correction loop cannot reject disturbances at the rate a humanoid's own contacts generate them, and the 60 Hz pose flush must never overwrite the reaction-mass actuators that the faster loop is writing to independently (Section 4.3).

## 4.1 Root-capsule orientation torque balancer

The pelvis is treated as a 15 kg capsule. A PD torque is applied to it through `xfrc_applied`, proportional to its tilt from upright plus a damping term on angular velocity: default gains $k_p = 800$, $k_d = 320$, with a hard torque cap of 120 N·m. This layer is meant to run at reduced authority while the body is executing deliberate, commanded motion. Section 5.1 explains why that reduction never actually happens.

## 4.2 Per-joint PD position servoing

Each of the 49 actuated joints has its own PD servo, and the gains are set per anatomical group rather than uniformly: spine at $k_p{=}700, k_v{=}130$ (stiff, to resist trunk sag under gravity); hips at $k_p{=}900, k_v{=}150$; knees at $k_p{=}1000, k_v{=}180$, the highest in the body, since they carry the most load; ankles at $k_p{=}600, k_v{=}100$; arms and shoulders somewhere in the $k_p{=}150\text{-}200$ range; and neck/head deliberately soft at $k_p{=}80, k_v{=}25$, a stiffer gain there just makes the head bobble. Targets pass through a three-stage clamp pipeline, joint coupling rules, anatomical limits, and a 20-step actuation ramp after spawn or reset so the body doesn't snap to target on the first physics step, before reaching the physics engine.

## 4.3 Reaction-mass balance system

This layer is physical, not purely computational, and it's the one part of the architecture that doesn't have an obvious analog in most contact-force controllers. An 18 kg non-colliding sphere rides on two perpendicular slide joints, $\pm 0.6$ m of travel on each. Every slide is driven by its own actuator ($k_p{=}1500, k_v{=}260$) with a paired shock-absorber term ($k_p{=}200, k_d{=}40$). The mass is written exclusively by a dedicated `ReactionMassController` at the full 500 Hz physics rate, and it's deliberately excluded from the actuator map the 60 Hz pose-flush routine uses, specifically so that routine can never zero it out from under the balance layer. The working principle is the same as a reaction wheel or counterweight in physical robotics, sliding mass opposite the direction of center-of-mass disturbance, just implemented as a linear slide instead of a rotating wheel. We're not aware of this exact mechanism, a slide-mounted reaction mass balancing a browser-simulated humanoid, having been documented elsewhere, though the underlying reaction-mass principle is well established.

## 4.4 Center-of-mass lean reflex and capture-step recovery

This layer continuously estimates horizontal center-of-mass offset and velocity and computes the capture-point quantities from Section 2 ($\tau = \sqrt{h/g}$, $\text{CP} = x_{\text{COM}} + \dot{x}_{\text{COM}}\tau$). Below a threshold offset, it injects a corrective lean at the upper spine (positive offset leans the torso backward, countering forward drift). Above that threshold, it triggers a capture step, a forced swing-leg motion meant to place a foot under the falling center of mass. This is where the second failure mode shows up.

## 4.5 Locomotion reference

Walking isn't generated from scratch by the layers above; it's driven by a hand-authored, four-phase gait table (right push/left swing, left touchdown/weight transfer, left push/right swing, right touchdown/cycle reset), expressed as per-phase joint-angle overrides on a Mixamo-convention skeleton, with an explicit design rationale attached to each phase. For example, a 2-degree lateral spine lean at push-off is there specifically to shift the center of mass over the stance foot for swing-leg clearance, and the arms counter-swing to conserve angular momentum. The table supplies nominal joint targets; the layers above correct deviations from it as they come up. It's documented internally as a starting suggestion meant to be adapted dynamically, not a fixed ground truth, and it isn't derived from motion-capture data.

## 4.6 LLM-to-controller interface

The language model supplies discrete, semantic intent, not 500 Hz joint commands, since network inference latency makes that rate structurally out of reach. That intent gets held fixed across many physics steps while the layers above execute against it. This is the same rate-separation problem Zawalski et al. [4] deal with for chain-of-thought token generation during VLA inference; we solve it by splitting reasoning rate from control rate at the architecture level, since our controller, unlike a VLA, isn't the thing doing the reasoning.

# 5. Failure-Mode Analysis

Both failures below are reported exactly as they appear in the project's own internal engineering notes, without softening either one to make the system look more finished than it is.

## 5.1 Balance-authority arbitration defect

The root balancer (Section 4.1) is supposed to run at reduced torque authority while the body is executing intentional, commanded motion, the reasoning being that a balancer tuned to resist disturbance will also resist deliberate movement unless something temporarily backs it off. A scaling constant for exactly this purpose, `MotorController.GAIT_BALANCE_SCALE`, exists in the codebase, gated behind a `gaitActive` flag. Per the project's own debugging documentation, that flag "exist[s] but [is] never activated by any caller." Even the intended scale value is inconsistent across the project's internal docs, one source lists 0.5, another lists 0.15, and nobody has bothered reconciling the two, because neither has ever actually run. The practical result: the balancer sits at full authority ($k_p{=}800, k_d{=}320$) at all times, including during deliberate motion, and it actively resists commanded movement instead of only correcting unintended drift.

This is a wiring bug, not a flaw in the layering itself. We report it in this much detail because anyone building a similar stacked controller should expect, and explicitly test for, exactly this class of defect: an authority-reduction path that's written but never actually wired into the control flow that's supposed to invoke it.

## 5.2 Capture-step structural failure

This one is architectural rather than a simple wiring mistake. When center-of-mass offset crosses the lean-correction threshold, the corrective lean commanded through the upper-spine injection point saturates before the body has actually stabilized, and the torso tips past the point where recovery is still possible. Meanwhile the swing leg's knee extensor, gain $k_p{=}1000, k_v{=}180$, the stiffest joint in the body, but still bounded by the same clamp pipeline as every other joint (Section 4.2), isn't strong enough to lift the foot clear of the ground against full body weight in the time available before the torso has already committed. The capture step triggers correctly under the intended conditions. It essentially never lands. The timing budget for the lean saturation and the torque budget for the swing-leg lift simply aren't compatible with each other as currently tuned.

The system's own 300-frame fall-diagnosis ring buffer (tilt angle, root height, per-foot contact state, center-of-mass position, applied `xfrc` torques, per-joint state) is what let us characterize this failure in the first place, and it's the same tooling we'd recommend for the controlled, repeated-trial study this report doesn't itself contain (Section 8). Fixing it will likely mean either a higher knee-extensor torque budget specifically during the swing phase, a lower lean-saturation limit that triggers the step earlier while more recovery margin is still available, or both. We leave the actual resolution, and confirmation of how much of this failure traces back to Section 5.1's defect versus the torque budget on its own, to future work.

# 6. Data Export and Validated Compatibility

Every transition gets logged as a structured trajectory point regardless of whether the balance controller succeeds on that episode, and it can be exported as JSONL, CSV, Parquet, HDF5, or a schema compatible with Hugging Face LeRobot [8]. To check this against real downstream tooling rather than just our own schema definition, we recorded a 50-episode session, exported it in the LeRobot v2.0 schema entirely client-side, and loaded the result in a clean Python 3.10 environment using the official `lerobot` library. It passed the library's validation with no structural modification and no manual field casting: proprioceptive states and motor targets mapped correctly into `torch.Tensor` objects of shape `[N, 80]` for the joint-state dimension, and episode boundaries, session identifiers, and termination flags indexed correctly for a standard `torch.utils.data.DataLoader`. This is the one claim in this report we're fully confident stating without qualification.

# 7. Discussion

In the terms Zawalski et al. [4] use, the architecture in Section 4 is an instance of splitting "thinking carefully" from "looking carefully and acting correctly": the LLM handles semantic intent, and a conventionally engineered, non-learned control stack, four layers deep, including a physical reaction-mass mechanism, is what actually keeps that intent physically viable. Section 5 is, in effect, the evidence for why it would be unsafe to assume a language model alone could produce that stability: even with a perfectly correct semantic instruction, an incorrectly arbitrated or under-budgeted low-level layer is enough to sink the whole task, regardless of how good the reasoning above it was.

We also think the specificity in Section 5 matters beyond this particular system. Naming the exact flag, the exact gains, and the exact recorder used to catch the failure isn't just thoroughness for its own sake, it's a shortcut for the next person. Someone building a similar layered stack for a different body will very plausibly hit an equivalent authority-arbitration bug, because the general failure pattern, a corrective layer that's never actually disabled during voluntary motion, is easy to introduce and easy to miss until deliberate motion under load is actually attempted for the first time.

# 8. Limitations

- Both failure modes in Section 5 are unresolved as of this writing. Nothing here should be read as a claim that the stepping-recovery controller works.
- The architecture has been built and exercised on exactly one embodiment. Whether the four-layer structure holds up on a differently proportioned or actuated body is untested.
- There's no controlled, repeated-trial benchmark of balance success rate, recovery latency, or robustness under systematically varied disturbance in this report, even though the tooling to run one already exists in the codebase (Section 5.2). Section 5's account is a mechanistic diagnosis from development observation and internal documentation, not a statistical characterization, and shouldn't be read as one.
- The LLM-to-controller rate separation in Section 4.6 is a fixed hold-and-execute scheme. Nothing adaptive, varying the hold duration with observed inference latency, has been implemented or tested.
- The compatibility result in Section 6 is about data format and schema validity, not about the quality, diversity, or downstream training utility of the trajectories themselves.
- The intended gait-authority reduction value is inconsistently documented internally (0.5 vs. 0.15, Section 5.1). Since the code path is unreachable, this has no runtime effect right now, but it should get resolved before the arbitration defect itself is fixed.

# 9. Future Work

The most immediate step is closing the authority-arbitration defect in Section 5.1, a scoped fix rather than an open research question, followed by a controlled evaluation, using the fall-diagnosis ring buffer already built into the system, of whether that fix changes anything about the capture-step failure in Section 5.2. That evaluation should report, at minimum, capture-step trigger rate, landing success rate, and fall latency across a fixed number of repeated trials under a few controlled disturbance magnitudes, which the existing `com_pendulum_recorder` and `diagnose_fall_quick` tooling already support without modification. Past that, replacing the fixed hold-and-execute LLM interface (Section 4.6) with something that adapts to observed inference latency, and testing the architecture on a second embodiment to see how much of its structure is actually load-bearing versus specific to this particular humanoid, are the directions we think are most likely to teach us something.

# 10. Conclusion

This report documented a four-layer, two-rate real-time balance control architecture, including a physical reaction-mass balancing mechanism, for an LLM-directed, high-degree-of-freedom humanoid running client-side in a browser, and it reported two of that architecture's failure modes with exact gains and code-level identifiers rather than glossing over them. The view underneath all of this is that an honest, specific account of why a control stack currently fails is worth as much to the field as an account of one that currently succeeds. Independent of the balance controller's current limitations, the system's trajectory export was validated against existing robot-learning tooling without modification, and that's the one fully verified claim in this document.

# Appendix A: Actuator and Control Parameters

| Bone group | $k_p$ | $k_v$ | Approx. damping ratio $\zeta$ | Notes |
|---|---|---|---|---|
| Fingers/thumbs | 5 | 1 | ~0.16 | Underdamped, tendon-synergy driven |
| Neck/head | 80 | 25 | ~0.49 | Deliberately soft to avoid oscillation |
| Shoulders | 150 | 30 | ~0.39 | |
| Wrists/hands | 150 | 30 | ~0.39 | |
| Arms/forearms | 200 | 40 | ~0.41 | |
| Spine (x3 joints) | 700 | 130 | ~0.35 | Stiff to resist trunk sag |
| Ankles/feet | 600 | 100 | ~0.32 | Balance-critical |
| Hips/upper legs | 900 | 150 | ~0.28 | High for upright stabilization |
| Knees | 1000 | 180 | ~0.28 | Highest in the body |
| Root-capsule torque balancer | 800 | 320 | n/a | Torque cap 120 N·m; intended gait-time reduction never executes (Section 5.1) |
| Reaction-mass slides (x2) | 1500 | 260 | n/a | Paired shock-absorber gain $k_p{=}200, k_d{=}40$ |

Physics timestep: 2 ms (500 Hz), implicit-fast integrator, 200 solver iterations. Actuation ramp: $\min(1.0, \text{stepCount}/20)$ over the first 20 physics steps after spawn or reset.

# Code and Data Availability

Live platform: `https://runsynthia.online`
Source code: `https://github.com/Greatness0123/synthia`

# References

[1] A. Brohan et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In *Conference on Robot Learning*, 2023.

[2] M. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. OpenVLA: An open-source vision-language-action model. In *Proceedings of the 8th Conference on Robot Learning (CoRL)*, PMLR 270:2679-2713, 2024.

[3] W. Huang et al. Inner monologue: Embodied reasoning through planning with language models. 2022.

[4] M. Zawalski, W. Chen, K. Pertsch, O. Mees, C. Finn, and S. Levine. Robotic control via embodied chain-of-thought reasoning. arXiv:2407.08693, 2024.

[5] J. Pratt, J. Carff, S. Drakunov, and A. Goswami. Capture point: A step toward humanoid push recovery. In *2006 6th IEEE-RAS International Conference on Humanoid Robots*, pp. 200-207, 2006.

[6] J. Englsberger, C. Ott, M. A. Roa, A. Albu-Schäffer, and G. Hirzinger. Bipedal walking control based on capture point dynamics. In *2011 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pp. 4420-4427, 2011.

[7] M. Krause, J. Englsberger, P.-B. Wieber, and C. Ott. Stabilization of the capture point dynamics for bipedal walking based on model predictive control. *IFAC Proceedings Volumes*, 45(22), pp. 165-171, 2012.

[8] Hugging Face. LeRobot: State-of-the-art machine learning for real-world robotics. 2024.

[9] E. Todorov, T. Erez, and Y. Tassa. MuJoCo: A physics engine for model-based control. In *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2012.

[10] E. Coumans and R. Bai. PyBullet: A Python module for physics simulation for games, robotics, and machine learning. 2016.

[11] NVIDIA. Isaac Sim: Robotics simulation and synthetic data generation. 2023.

[12] M. Savva et al. Habitat: A platform for embodied AI research. In *IEEE/CVF International Conference on Computer Vision*, 2019.

[13] S. James et al. PyRep: Bringing the V-REP simulator to researchers. In *Conference on Robot Learning*, 2019.

[14] A. Brohan et al. RT-1: Robotics transformer for real-world control at scale. In *Proceedings of Robotics: Science and Systems (RSS)*, 2023.

[15] Open X-Embodiment Collaboration. Open X-Embodiment: Robotic learning datasets and RT-X models. arXiv:2310.08864, 2023.

[16] A. Khazatsky et al. DROID: A large-scale in-the-wild robot manipulation dataset. arXiv:2403.12945, 2024.

[17] G. Okorie. Synthia: A browser-native, LLM-directed embodied AI platform. Open-source software, v1.5.1, 2026. `https://github.com/Greatness0123/synthia`