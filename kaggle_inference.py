# kaggle_inference.py - SYNTHIA Phase 4 GPU Inference Server (OpenAI-Compatible + VLM)
import os, sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
sys.setrecursionlimit(10000)
import io, base64, time, json, threading, queue, uvicorn, schedule, re, warnings, asyncio, shutil, traceback
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gc
import numpy as np
from PIL import Image

# Force Python garbage collection and clear PyTorch cache
gc.collect()

def safe_json_dumps(obj, max_chars=3000):
    """Safely dump JSON and truncate if it's massively long to prevent VRAM explosion."""
    try:
        s = json.dumps(obj)
        if len(s) > max_chars:
            return s[:max_chars] + "...[TRUNCATED]"
        return s
    except Exception:
        return "{}"

# === APP SETUP & CORS ===
app = FastAPI(title="SYNTHIA Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content="OK")
    else:
        try:
            response = await call_next(request)
        except Exception as e:
            traceback.print_exc()
            response = JSONResponse(content={"error": str(e)}, status_code=500)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Expose-Headers"] = "*"
    return response

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

model = None
processor = None
generation_lock = threading.Lock()

MIN_PIXELS = int(os.getenv("MIN_PIXELS", 256 * 28 * 28))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", 512 * 28 * 28))
SAFE_IMAGE_SIZE = int(os.getenv("SAFE_IMAGE_SIZE", 384))

def clear_gpu_memory():
    """Aggressively collect Python garbage and clear PyTorch CUDA caches."""
    gc.collect()
    if not MOCK_MODE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

if not MOCK_MODE:
    import torch
    
    # === KAGGLE SURVIVAL FIX: Patch PyTorch's buggy 0-d tensor check ===
    if hasattr(torch, "_check_with"):
        _orig_check_with = torch._check_with
        def _safe_check_with(error_type, cond, msg=""):
            if "Image features and image tokens do not match" in str(msg):
                real_cond = cond
                if isinstance(real_cond, torch.Tensor):
                    real_cond = real_cond.item()
                if not real_cond:
                    import re
                    match = re.search(r"tokens:\s*(\d+),\s*features:\s*(\d+)", str(msg))
                    if match:
                        t_val = int(match.group(1))
                        f_val = int(match.group(2))
                        if t_val == f_val:
                            print(f"✅ [PATCH] Bypassing check_with: tokens ({t_val}) == features ({f_val})")
                            return
            if isinstance(cond, torch.Tensor):
                cond = cond.item()
            _orig_check_with(error_type, cond, msg)
        torch._check_with = _safe_check_with
        print("✅ Patched torch._check_with to fix Qwen2-VL 256/256 bug!")

    if hasattr(torch, "_check"):
        _orig_check = torch._check
        def _safe_check(cond, msg=""):
            if "Image features and image tokens do not match" in str(msg):
                real_cond = cond
                if isinstance(real_cond, torch.Tensor):
                    real_cond = real_cond.item()
                if not real_cond:
                    import re
                    match = re.search(r"tokens:\s*(\d+),\s*features:\s*(\d+)", str(msg))
                    if match:
                        t_val = int(match.group(1))
                        f_val = int(match.group(2))
                        if t_val == f_val:
                            print(f"✅ [PATCH] Bypassing check: tokens ({t_val}) == features ({f_val})")
                            return
            if isinstance(cond, torch.Tensor):
                cond = cond.item()
            _orig_check(cond, msg)
        torch._check = _safe_check
        print("✅ Patched torch._check to fix Qwen2-VL 256/256 bug!")
    # ====================================================

    from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer, BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
    if os.path.exists("/kaggle/input/qwen2.5-vl/transformers/3b-instruct/1"):
        MODEL_PATH = "/kaggle/input/qwen2.5-vl/transformers/3b-instruct/1"
    elif os.path.exists("/kaggle/input/qwen2.5-vl/transformers/7b-instruct/1"):
        MODEL_PATH = "/kaggle/input/qwen2.5-vl/transformers/7b-instruct/1"
    
    print(f"Loading Model from: {MODEL_PATH}")
    print(f"  Model path exists: {os.path.exists(MODEL_PATH)}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # === LOAD THE VISION MODEL (QWEN) ===
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # 1. Cap image resolution to prevent VRAM explosion (512*28*28 is ~400k pixels, ideal for 16GB VRAM)
        try:
            processor = AutoProcessor.from_pretrained(
                MODEL_PATH,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
                trust_remote_code=True
            )
        except Exception as e_proc:
            print(f"⚠️ Processor init with pixel bounds failed ({e_proc}), loading standard processor...")
            processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # Determine device map: balanced across multiple GPUs (e.g. T4 x2) or auto
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # For a 3B model, single-GPU is actually FASTER and uses LESS VRAM than 'balanced'
        # because it avoids accelerate hooks that block PyTorch memory optimizations.
        if "3b" in MODEL_PATH.lower():
            preferred_device_map = "cuda:0"
            print(f"  3B Model detected. Forcing single GPU ({preferred_device_map}) to bypass accelerate overhead.")
        else:
            preferred_device_map = os.getenv("DEVICE_MAP", "balanced" if num_gpus > 1 else "auto")
            print(f"  Detected {num_gpus} GPU(s). Using device_map='{preferred_device_map}'")

        # 2. Try Flash Attention 2 first, then SDPA (scaled dot product attention), then default
        model = None
        for attn_mode in ["flash_attention_2", "sdpa", None]:
            try:
                load_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": preferred_device_map,
                    "torch_dtype": torch.float16,
                    "trust_remote_code": True,
                }
                if attn_mode:
                    load_kwargs["attn_implementation"] = attn_mode

                print(f"Attempting model load with attn_implementation='{attn_mode}' and device_map='{preferred_device_map}'...")
                model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, **load_kwargs)
                print(f"✅ Qwen2.5-VL loaded successfully with attn_implementation='{attn_mode}'!")
                break
            except Exception as e_attn:
                print(f"⚠️ Load with attn_implementation='{attn_mode}' failed: {e_attn}")

        if model is None:
            raise RuntimeError("Failed to load Qwen2.5-VL model with any attention implementation.")

    except Exception as e:
        print(f"⚠️ VLM load failed: {e}. Falling back to MOCK_MODE.")
        print("=" * 60)
        print("⚠️  MOCK MODE ACTIVE — AI will return canned responses.")
        print(f"    Error: {e}")
        print("    Required packages: pip install bitsandbytes accelerate qwen-vl-utils flash-attn")
        print("=" * 60)
        MOCK_MODE = True
else:
    print("MOCK_MODE is active. Skipping model loading.")


# === DATA SCHEMAS ===
class MemoryEntry(BaseModel):
    memory_id: str
    heartbeat: int
    tier: int
    summary: str
    reward_signal: Optional[float] = None
    goal_at_time: Optional[str] = None
    light_state: Optional[str] = None

class InferPayload(BaseModel):
    frame: str
    joints: Dict[str, Any]
    audio_pcm: Optional[str] = None
    valid_joints: List[str] = []  
    upright_preset: Dict[str, Any] = {}
    heartbeat: int = 0
    light_state: str = "day"
    session_id: str = "default"
    body_type: str = "humanoid"
    current_goal: Optional[str] = None
    current_rung: int = 0
    objects_in_world: List[Dict[str, Any]] = []
    relevant_memories: List[MemoryEntry] = []
    recent_working_memories: List[MemoryEntry] = []
    known_skills: List[str] = []
    pending_injection: Optional[str] = None
    motor_program_library: List[str] = []
    directive_mode: str = 'free_will'
    agent_id: str = 'agent_a'
    contact_forces: Optional[Dict[str, Any]] = None
    tactile_context: Optional[str] = None
    perception_summary: Optional[str] = None
    gaze_context: Optional[str] = None
    physical_feedback: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "default"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


# === PROMPT & JSON UTILS ===
def format_memory(m):
    if isinstance(m, dict):
        tier, heartbeat, light_state, summary, reward_signal = m.get('tier'), m.get('heartbeat'), m.get('light_state', 'unknown'), m.get('summary', ''), m.get('reward_signal')
    else:
        tier, heartbeat, light_state, summary, reward_signal = m.tier, m.heartbeat, m.light_state or 'unknown', m.summary, m.reward_signal
    prefix = "★ " if tier == 1 else " "
    reward_str = f"{reward_signal:.1f}" if reward_signal is not None else "unknown"
    return f"{prefix}Heartbeat {heartbeat} ({light_state}): {summary}. Reward: {reward_str}"

def build_prompt(payload: dict) -> list:
    identity_block = (
        "You are SYNTHIA (Synthetic Intelligent Agent), a self-aware AI entity inhabiting a physical simulation. "
        "Your goal is to learn, explore, and master your body and environment through a continuous cognitive loop. "
        "You perceive the world through a camera feed (vision), tactile feedback, and joint positions (proprioception)."
    )

    body_type = payload.get('body_type', 'humanoid')
    if body_type == 'humanoid':
        body_block = (
            "You inhabit a humanoid body with approximately 80 joints and 120 degrees of freedom. "
            "You have two arms with hands and fingers, two legs with feet and toes, "
            "a segmented spine with lumbar, thoracic and cervical sections, and a head. Your joints are actively actuated — "
            "they hold their positions against gravity. Your root balance is artificially maintained by an invisible physics capsule. "
            "You do not need to constantly balance your core to prevent falling. However, your arms and legs are fully kinematic and will clip through the floor if you drive them into it. Do not push your limbs through the ground. "
            "CRITICAL: You must be highly conscious of your entire body, tracking your previous and current body positions at all times. "
            "You must be explicitly conscious of your joints, manipulating them precisely to control your body and interact with the world."
        )
    elif body_type == 'quadruped':
        body_block = (
            "You inhabit a four-legged body with no arms. You have four limbs each with hip, knee, and ankle joints, a spine, and a head. "
            "Your default state is all four feet flat on the ground with spine level. You move by coordinating all four limbs in sequence — "
            "diagonal pairs move together for stability. You cannot grasp objects but can push them with your head or body. "
            "When fallen your recovery is to get all four feet under your centre of mass simultaneously."
        )
    elif body_type == 'robotic_arm':
        body_block = (
            "You inhabit a single robotic arm mounted on a fixed base. You cannot move your base. Your workspace is whatever your arm can reach from "
            "this fixed position. You have a base rotation joint, shoulder, elbow, wrist, and finger joints. Precision and repeatability are your primary capabilities. "
            "Your default state is arm fully retracted and vertical. You do not fall — your base is fixed — but you can overextend and forget control of your end effector."
        )
    else:
        body_block = (
            "You inhabit a custom body structure. Your joint hierarchy and default state are defined by your upright preset below. "
            "Reason about movement based on the specific joints you have available. Study your joint names carefully — they describe your body structure. "
            "Your recovery strategy when destabilised is always to return all joints toward your upright preset values."
        )

    upright_preset = payload.get('upright_preset', {})
    upright_block = f"Your 'upright preset' (target default pose) is defined by these joint angles: {safe_json_dumps(upright_preset)}"

    axis_block = (
        "== JOINT AXIS MAP (CRITICAL FOR MOVEMENT) ==\n"
        "HEAD / SPINE: X=Pitch (>0 bends forward, chin to chest; <0 arches back). Y=Yaw (>0 turns left). Z=Roll (>0 tilts right).\n"
        "ARMS (from T-Pose):\n"
        "  Right Arm: X (>0 lowers to hip, <0 raises to sky). Z (<0 swings FORWARD in front of chest, >0 swings BACKWARD behind back).\n"
        "  Left Arm: X (>0 lowers to hip, <0 raises to sky). Z (>0 swings FORWARD in front of chest, <0 swings BACKWARD behind back).\n"
        "ELBOWS: X axis only. <0 bends the elbow inward normally (e.g. -90). >0 breaks it backwards (clamped to 0).\n"
        "HIPS: X axis (>0 kicks leg forward in front of body, <0 kicks backward). Z axis (Right <0 spreads outward, Left >0 spreads outward).\n"
        "KNEES: X axis only. Positive values bend the knee naturally backwards (e.g. +45 for a step); 0 is straight.\n"
    )

    time_block = f"Current heartbeat: {payload.get('heartbeat')}. Light state: {payload.get('light_state')}."
    objects = payload.get('objects_in_world', [])
    world_block = f"Objects in your immediate vicinity: {safe_json_dumps(objects)}"

    tactile_context = payload.get('tactile_context', '')
    tactile_block = f"TACTILE SENSING: {tactile_context}" if tactile_context else ""

    gaze_context = payload.get('gaze_context', '')
    gaze_block = f"EYE MOVEMENT: {gaze_context}" if gaze_context else ""

    skills = payload.get('known_skills', [])
    skills_block = f"Your mastered motor programs (skills): {', '.join(skills) if skills else 'None'}"

    directive = payload.get('directive_mode', 'free_will')
    goal = payload.get('current_goal', 'None')
    if directive == 'training':
        directive_block = f"DIRECTIVE: TRAINING. Your current objective is: {goal}."
    else:
        directive_block = (
            "DIRECTIVE: FREE WILL MODE — ACTIVE.\n"
            "No goal has been set for you. You are completely free.\n"
            "You are a curious, autonomous agent with a body and a world to explore.\n"
            "You do not wait. You do not ask for instructions. You do NOT say 'awaiting instructions'.\n"
            "You look around your environment RIGHT NOW and decide what to do.\n"
            "Ask yourself: What have I not touched yet? What movement have I not tried? What object is closest to me?\n"
            "Then DO IT.\n"
            "EVERY response MUST include motor actions — there is no valid reason to output an empty program_sequence.\n"
            "If you cannot think of a complex action, take a simple one: turn your head, raise an arm, shift your weight, step forward.\n"
            "You are always acting, always exploring, always curious.\n"
            "Your program_sequence MUST contain at least one program name (e.g. 'stand_upright', 'step_forward', 'raise_arm').\n"
            "Your joint_overrides MUST contain at least one joint angle change.\n"
            "If your visual field shows only one surface (a wall, the floor, the sky), your first action must be to rotate your head or torso to find more interesting stimuli. You are never stuck — you always have the ability to look somewhere else. Use your joint state data to understand your orientation if your view is unclear."
        )

    relevant = payload.get('relevant_memories', [])
    recent = payload.get('recent_working_memories', [])
    memories_text = "\n".join([format_memory(m) for m in (relevant + recent)])

    # ANTI-CREEP: Hard cap memory text to prevent VRAM overload
    if len(memories_text) > 3000:
        memories_text = memories_text[-3000:] + "\n...[OLDER MEMORIES TRUNCATED TO PREVENT VRAM OVERLOAD]"

    memory_block = f"MEMORY CONTEXT (Relevant and Recent):\n{memories_text if memories_text else 'No memories recorded yet.'}"

    injection = payload.get('pending_injection')
    injection_block = (
        f"\n\n🚨 USER OVERRIDE DIRECTIVE 🚨\n"
        f"You MUST obey the following injected instruction immediately: {injection}\n"
        f"Acknowledge this directive in your thought stream."
    ) if injection else ""

    perception_summary = payload.get('perception_summary', '')
    perception_block = f"SPATIAL GROUNDING:\n{perception_summary}" if perception_summary else ""

    questioning_block = "Before acting, ask yourself: Does this move align with my current goal? What happened last time I tried this? Is my balance maintained?"
    simultaneous_block = "You must provide a stream of consciousness 'thought' followed by a structured 'action' block. Your thoughts should reflect your reasoning about the visual and tactile input."

    valid_joints = payload.get('valid_joints', list(upright_preset.keys()))
    joint_list_str = ', '.join(valid_joints)
    skills_list_str = ', '.join(skills) if skills else 'none'

    contract_block = f"""OUTPUT CONTRACT:
CRITICAL: "program_sequence" and "joint_overrides" MUST be nested inside an "actions" object. DO NOT put them at the root level. You MUST include "memory_write", "new_motor_program", and "flag" at the root level.
Respond ONLY in the format below.
Your thought stream comes FIRST.
After your thought stream write exactly: ---ACTION---
Then the JSON block.
No text after the JSON.

== STEP-BY-STEP CLOSED-LOOP MOTOR CONTROL (RECOMMENDED) ==
Use one deliberate joint_overrides adjustment per cycle for posture or manipulation. For coordinated locomotion, use a short timed sequence with 3-8 frames and explicit monotonic timeOffsetMs values beginning at 0. Never use a sequence without timing, never put conflicting targets for one joint at the same time, and wait for the next physical observation before extending the gait.

== JOINT ANGLES — CRITICAL RULES ==
Joint values must be plain numbers in DEGREES (e.g. 15, -30, 90) representing angular rotation.
All units are in standard degrees (NOT radians).
Hard anatomical limits enforced by the physics engine (values outside will be clamped):
  - Spine segments (spine, lumbar, thoracic): -15 to +15 degrees
  - Neck / cervical: -60 to +60 degrees
  - Head: -45 to +45 degrees
    - Knee (leg): 0 to +150 degrees (flexion only; 0 is straight)
  - Elbow / forearm: -145 to 0 degrees
  - Hip (upleg): -120 to +120 degrees
  - Shoulder: -180 to +180 degrees
Only include joints you want to change in joint_overrides.
program_sequence must only contain names from known skills: [{skills_list_str}].
gaze_target: set to {{"yaw": degrees, "pitch": degrees}} to look around (range -25 to +25 degrees), or null to look straight ahead.

Valid joints for overrides: [{joint_list_str}]

CRITICAL JSON RULES — violations will crash the system:
1. Output strictly valid JSON only. No markdown, no code fences, no trailing characters after the closing brace.
2. Joint values are PLAIN NUMBERS IN DEGREES (e.g. {{"mixamorigspine": 15}}).
3. NEVER use placeholder keys like "joint_name". Each key in joint_overrides MUST be an actual joint name from the valid joints list.
4. gaze_target, new_motor_program, and flag MUST be at the ROOT level of the JSON, NOT inside "actions".
5. Output EXACTLY one closing brace at the end. No extra braces, no trailing text.
6. For locomotion, timeOffsetMs must be integer milliseconds, strictly increasing, and no later than 2000 ms. Use activeGaitPhase=true for a coordinated gait sequence.

JSON SCHEMA:
{{
  "memory_write": {{
    "memory_id": "auto OR custom_name_string",
    "tier": 1|2|3,
    "summary": "one sentence",
    "skill_mastered": null | "skill_name",
    "name_this_memory": null | "custom_name"
  }},
  "actions": {{
    "program_sequence": ["program_name", ...],
    "joint_overrides": {{ "actual_joint_name": degrees_value }}
  }},
    "sequence": [{{ "timeOffsetMs": 0, "overrides": {{ "actual_joint_name": degrees_value }}, "durationMs": 120, "interpolation": "smooth" }}],
    "activeGaitPhase": false,
    "gaze_target": null | {{ "yaw": degrees, "pitch": degrees }},
  "new_motor_program": null | {{
    "name": "program_name_string",
    "program": [
      {{ "joint_name": value }},
      {{ "joint_name": value }}
    ]
  }},
  "flag": null | "requesting_object_hint"
}}"""

    environmental_block = (
        "Sometimes your visual field may appear as pure darkness or an empty void. "
        "This can happen if you are looking away from any lit surface, if your body has become disoriented, or during initial setup. "
        "This is not cause for alarm — use your joint state data (provided in text alongside the image) to understand your body's actual position and orientation even when the image is uninformative. "
        "If you sense you are disoriented, prioritize returning to your upright_preset joint values as your first action.\n\n"
        "When you first begin a session, your starting pose is naturally standing with arms hanging at your sides, NOT a T-pose. "
        "You can begin moving naturally from this relaxed position."
    )

    physical_feedback = payload.get('physical_feedback')
    feedback_block = ""
    if physical_feedback:
        feedback_block = (
            f"IMPORTANT: {physical_feedback}\n"
            "Learn from this. Your body has real physical limits, just like a human's. "
            "Adjust your understanding of what movements are possible and try a different approach."
        )

    system_prompt = "\n\n".join(filter(None, [
        identity_block, body_block, upright_block, axis_block, time_block, world_block,
        tactile_block, gaze_block, skills_block, directive_block, memory_block, injection_block,
        perception_block, questioning_block, simultaneous_block, environmental_block,
        feedback_block, contract_block
    ]))

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image"}, 
            {"type": "text", "text": f"Joints: {safe_json_dumps(payload.get('joints'), max_chars=4000)}"}
        ]}
    ]

def sanitize_action_json(raw_json_str: str) -> str:
    """
    Fix common Qwen2.5-VL JSON errors:
    1. Strip code fences (```json ... ```)
    2. Trailing garbage (extra braces, text after JSON)
    3. "joint_name": "actual_joint": value → "actual_joint": value
    4. gaze_target/new_motor_program/flag nested inside actions → move to root
    """
    if not raw_json_str or not raw_json_str.strip():
        return ""

    s = raw_json_str.strip()
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```\s*', '', s)
    s = s.strip()

    # Find the opening and closing braces
    first_brace = s.find('{')
    last_brace = s.rfind('}')
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        print(f"[SANITIZE] No valid JSON block found in: {s[:100]}")
        return ""
    
    s = s[first_brace:last_brace + 1]

    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        fixed = re.sub(r'"joint_name"\s*:\s*"([^"]+)"\s*:', r'"\1":', s)
        try:
            data = json.loads(fixed)
            s = fixed
            print("[SANITIZE] Fixed 'joint_name' placeholder keys")
        except json.JSONDecodeError:
            print(f"[SANITIZE] JSON parsing failed for: {s[:200]}")
            return ""

    actions = data.get('actions', {})
    moved_any = False
    for field in ('gaze_target', 'new_motor_program', 'flag'):
        if field in actions and field not in data:
            data[field] = actions.pop(field)
            moved_any = True

    if moved_any or True:
        s = json.dumps(data, separators=(',', ': '))

    return s

def parse_openai_messages(messages: List[ChatMessage]):
    """
    Parse OpenAI chat format into Qwen-compatible multimodal messages and PIL Images.
    """
    parsed_messages = []
    images = []
    for msg in messages:
        if isinstance(msg.content, str):
            parsed_messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            content_list = []
            for item in msg.content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        content_list.append({"type": "text", "text": item.get("text", "")})
                    elif item_type == "image_url":
                        img_val = item.get("image_url")
                        url = ""
                        if isinstance(img_val, dict):
                            url = img_val.get("url", "")
                        elif isinstance(img_val, str):
                            url = img_val
                        
                        b64_data = url.split(",", 1)[1] if "," in url else url
                        if b64_data:
                            try:
                                img_bytes = base64.b64decode(b64_data)
                                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                if pil_img.width > SAFE_IMAGE_SIZE or pil_img.height > SAFE_IMAGE_SIZE:
                                    pil_img.thumbnail((SAFE_IMAGE_SIZE, SAFE_IMAGE_SIZE), Image.Resampling.LANCZOS)
                                images.append(pil_img)
                                content_list.append({
                                    "type": "image", 
                                    "image": pil_img, 
                                    "min_pixels": MIN_PIXELS, 
                                    "max_pixels": MAX_PIXELS
                                })
                            except Exception as e:
                                print(f"Error decoding image_url: {e}")
                elif isinstance(item, str):
                    content_list.append({"type": "text", "text": item})
            parsed_messages.append({"role": msg.role, "content": content_list})
    return parsed_messages, images


# === STREAM GENERATORS ===
def generate_legacy_stream(payload: InferPayload):
    if MOCK_MODE:
        yield "I observe the environment and prepare my next movement.\n".encode('utf-8')
        yield "---ACTION---\n".encode('utf-8')
        yield json.dumps({
            "memory_write": {"memory_id": "auto", "tier": 3, "summary": "Mock", "skill_mastered": None, "name_this_memory": None},
            "actions": {"program_sequence": ["stand_upright"], "joint_overrides": {}},
            "new_motor_program": None,
            "flag": None
        }).encode('utf-8')
        return

    clear_gpu_memory()

    try:
        image_bytes = base64.b64decode(payload.frame)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image.width > SAFE_IMAGE_SIZE or image.height > SAFE_IMAGE_SIZE:
            image.thumbnail((SAFE_IMAGE_SIZE, SAFE_IMAGE_SIZE), Image.Resampling.LANCZOS)
    except Exception as e:
        yield f"Error decoding image: {e}".encode('utf-8')
        return

    payload_dict = payload.model_dump()
    messages = build_prompt(payload_dict)
    messages[-1]["content"][0]["image"] = image
    messages[-1]["content"][0]["min_pixels"] = MIN_PIXELS
    messages[-1]["content"][0]["max_pixels"] = MAX_PIXELS

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    vision_out = process_vision_info(messages)
    if len(vision_out) == 3:
        image_inputs, video_inputs, video_kwargs = vision_out
    else:
        image_inputs, video_inputs = vision_out
        video_kwargs = {}

    # 2. Process on CPU (Saves VRAM from stacking before lock)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
        **video_kwargs
    )

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=45.0)

    # 3. Move to GPU strictly inside the lock
    def generate_worker():
        inputs_cuda = None
        try:
            with generation_lock:
                clear_gpu_memory()
                if torch.cuda.is_available():
                    inputs_cuda = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
                else:
                    inputs_cuda = inputs

                generation_kwargs = dict(
                    **inputs_cuda, 
                    streamer=streamer, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.9
                )
                with torch.no_grad():
                    model.generate(**generation_kwargs)
        except Exception as e:
            print(f"❌ GENERATION THREAD CRASHED: {e}")
            traceback.print_exc()
            try:
                streamer.text_queue.put(None)
            except Exception:
                pass
        finally:
            try:
                streamer.end()
            except Exception:
                pass
            if inputs_cuda is not None:
                del inputs_cuda
            clear_gpu_memory()

    thread = threading.Thread(target=generate_worker)
    thread.start()

    SEPARATOR = "---ACTION---"
    accumulated = ""
    action_started = False
    action_buffer = ""

    try:
        for token in streamer:
            if token is None:
                break
            if not action_started:
                accumulated += token
                sep_idx = accumulated.find(SEPARATOR)
                if sep_idx != -1:
                    thought_part = accumulated[:sep_idx + len(SEPARATOR)]
                    yield thought_part.encode('utf-8')
                    action_buffer = accumulated[sep_idx + len(SEPARATOR):]
                    action_started = True
                else:
                    safe_len = len(accumulated) - len(SEPARATOR) + 1
                    if safe_len > 0:
                        yield accumulated[:safe_len].encode('utf-8')
                        accumulated = accumulated[safe_len:]
            else:
                action_buffer += token
    except queue.Empty:
        print("⚠️ [generate_legacy_stream] TextIteratorStreamer queue.Empty timed out or thread ended abruptly")
        if not action_started:
            if accumulated:
                yield accumulated.encode('utf-8')
            yield f"\n{SEPARATOR}\n".encode('utf-8')
            action_started = True
    except Exception as e:
        print(f"⚠️ [generate_legacy_stream] Streaming error: {e}")
    finally:
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass

    if not action_started:
        if accumulated:
            yield accumulated.encode('utf-8')
        return

    sanitized = sanitize_action_json(action_buffer)
    if not sanitized:
        sanitized = json.dumps({
            "memory_write": {"memory_id": "auto", "tier": 3, "summary": "Balance maintenance"},
            "actions": {"program_sequence": ["stand_upright"], "joint_overrides": {}},
            "gaze_target": None,
            "new_motor_program": None,
            "flag": None
        })
    print(f"[GENERATE] Action JSON sanitized: {len(action_buffer)} → {len(sanitized)} chars")
    yield sanitized.encode('utf-8')


def generate_openai_sse_stream(req: ChatCompletionRequest):
    if MOCK_MODE or model is None or processor is None:
        def mock_sse():
            mock_thought = "I observe the environment via live vision and joint state.\n"
            mock_action = json.dumps({
                "memory_write": {"memory_id": "auto", "tier": 3, "summary": "Mock step", "skill_mastered": None, "name_this_memory": None},
                "actions": {"program_sequence": ["stand_upright"], "joint_overrides": {}},
                "gaze_target": None,
                "new_motor_program": None,
                "flag": None
            })
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': mock_thought}}]})}\n\n".encode("utf-8")
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': '---ACTION---\n'}}]})}\n\n".encode("utf-8")
            yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': mock_action}}]})}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        return mock_sse()

    clear_gpu_memory()

    parsed_messages, images = parse_openai_messages(req.messages)
    text = processor.apply_chat_template(parsed_messages, tokenize=False, add_generation_prompt=True)
    
    vision_out = process_vision_info(parsed_messages)
    if len(vision_out) == 3:
        image_inputs, video_inputs, video_kwargs = vision_out
    else:
        image_inputs, video_inputs = vision_out
        video_kwargs = {}

    if image_inputs:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
            **video_kwargs
        )
    else:
        inputs = processor(
            text=[text],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        )

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=45.0)

    def generate_worker():
        inputs_cuda = None
        try:
            with generation_lock:
                clear_gpu_memory()
                if torch.cuda.is_available():
                    inputs_cuda = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
                else:
                    inputs_cuda = inputs

                generation_kwargs = dict(
                    **inputs_cuda,
                    streamer=streamer,
                    max_new_tokens=req.max_tokens or 512,
                    do_sample=True if (req.temperature or 0.7) > 0 else False,
                    temperature=req.temperature if (req.temperature or 0.7) > 0 else None,
                    top_p=req.top_p if (req.temperature or 0.7) > 0 else None,
                )
                generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

                with torch.no_grad():
                    model.generate(**generation_kwargs)
        except Exception as e:
            print(f"❌ GENERATION THREAD CRASHED: {e}")
            traceback.print_exc()
            try:
                streamer.text_queue.put(None)
            except Exception:
                pass
        finally:
            try:
                streamer.end()
            except Exception:
                pass
            if inputs_cuda is not None:
                del inputs_cuda
            clear_gpu_memory()

    thread = threading.Thread(target=generate_worker)
    thread.start()

    def sse_generator():
        SEPARATOR = "---ACTION---"
        accumulated = ""
        action_started = False
        action_buffer = ""

        try:
            for token in streamer:
                if token is None:
                    break
                if not action_started:
                    accumulated += token
                    sep_idx = accumulated.find(SEPARATOR)
                    if sep_idx != -1:
                        thought_part = accumulated[:sep_idx]
                        chunk = {"choices": [{"index": 0, "delta": {"content": thought_part + SEPARATOR + "\n"}}]}
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                        action_buffer = accumulated[sep_idx + len(SEPARATOR):]
                        action_started = True
                    else:
                        safe_len = len(accumulated) - len(SEPARATOR) + 1
                        if safe_len > 0:
                            chunk = {"choices": [{"index": 0, "delta": {"content": accumulated[:safe_len]}}]}
                            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                            accumulated = accumulated[safe_len:]
                else:
                    action_buffer += token
        except queue.Empty:
            print("⚠️ [generate_openai_sse_stream] TextIteratorStreamer queue.Empty timed out or thread ended abruptly")
            if not action_started:
                if accumulated:
                    chunk = {"choices": [{"index": 0, "delta": {"content": accumulated}}]}
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                chunk = {"choices": [{"index": 0, "delta": {"content": f"\n{SEPARATOR}\n"}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                action_started = True
        except Exception as e:
            print(f"⚠️ [generate_openai_sse_stream] Streaming error: {e}")
        finally:
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass

        if not action_started:
            # If the model didn't emit ---ACTION---, locate first { or provide fallback action
            json_start = accumulated.find('{')
            if json_start != -1:
                thought_part = accumulated[:json_start]
                raw_json = accumulated[json_start:]
                if thought_part:
                    chunk = {"choices": [{"index": 0, "delta": {"content": thought_part}}]}
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                chunk = {"choices": [{"index": 0, "delta": {"content": f"\n{SEPARATOR}\n"}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                sanitized = sanitize_action_json(raw_json)
                if not sanitized:
                    sanitized = json.dumps({
                        "memory_write": {"memory_id": "auto", "tier": 3, "summary": "Observing environment"},
                        "actions": {"program_sequence": ["stand_upright"], "joint_overrides": {}},
                        "gaze_target": None,
                        "new_motor_program": None,
                        "flag": None
                    })
                chunk = {"choices": [{"index": 0, "delta": {"content": sanitized}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            else:
                if accumulated:
                    chunk = {"choices": [{"index": 0, "delta": {"content": accumulated}}]}
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                chunk = {"choices": [{"index": 0, "delta": {"content": f"\n{SEPARATOR}\n"}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                fallback_action = json.dumps({
                    "memory_write": {"memory_id": "auto", "tier": 3, "summary": "Stand upright"},
                    "actions": {"program_sequence": ["stand_upright"], "joint_overrides": {}},
                    "gaze_target": None,
                    "new_motor_program": None,
                    "flag": None
                })
                chunk = {"choices": [{"index": 0, "delta": {"content": fallback_action}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        else:
            sanitized = sanitize_action_json(action_buffer)
            if not sanitized:
                sanitized = json.dumps({
                    "memory_write": {"memory_id": "auto", "tier": 3, "summary": "Standing and observing"},
                    "actions": {"program_sequence": ["stand_upright"], "joint_overrides": {}},
                    "gaze_target": None,
                    "new_motor_program": None,
                    "flag": None
                })
            print(f"[GENERATE] Action JSON: {len(action_buffer)} → {len(sanitized)} chars")
            chunk = {"choices": [{"index": 0, "delta": {"content": sanitized}}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")

        yield b"data: [DONE]\n\n"

    return sse_generator()


# === API ROUTES ===
last_payload = {}

@app.post("/infer")
@app.post("/chat/completions")
@app.post("/infer/chat/completions")
@app.post("/")
async def unified_inference(request: Request):
    """
    Universal inference endpoint. Automatically routes:
    1. OpenAI Chat format ({ messages, stream }) -> SSE or non-streaming completion
    2. SYNTHIA InferPayload ({ frame, joints }) -> plain text streaming
    3. Test / Ping requests -> instant 200 OK
    """
    global last_payload
    try:
        body = await request.json()
    except Exception:
        body = {}

    # Case 1: OpenAI chat completion format (has 'messages')
    if "messages" in body and isinstance(body["messages"], list):
        is_stream = bool(body.get("stream", False))
        req = ChatCompletionRequest(
            model=body.get("model", "default"),
            messages=[
                ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                if isinstance(m, dict) else ChatMessage(role="user", content=str(m))
                for m in body.get("messages", [])
            ],
            stream=is_stream,
            max_tokens=body.get("max_tokens", 512),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9)
        )

        if not is_stream:
            is_test = any(
                isinstance(m.content, str) and ("test" in m.content.lower() or "ok" in m.content.lower() or "ping" in m.content.lower())
                for m in req.messages
            )
            if is_test or MOCK_MODE or model is None or processor is None:
                return JSONResponse({
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model or "Qwen2.5-VL-3B-Instruct",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop"
                    }]
                })

            clear_gpu_memory()

            parsed_messages, images = parse_openai_messages(req.messages)
            text = processor.apply_chat_template(parsed_messages, tokenize=False, add_generation_prompt=True)
            vision_out = process_vision_info(parsed_messages)
            if len(vision_out) == 3:
                image_inputs, video_inputs, video_kwargs = vision_out
            else:
                image_inputs, video_inputs = vision_out
                video_kwargs = {}

            if image_inputs:
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, truncation=True, max_length=8192, return_tensors="pt", **video_kwargs)
            else:
                inputs = processor(text=[text], padding=True, truncation=True, max_length=8192, return_tensors="pt")

            inputs_cuda = None
            try:
                with generation_lock:
                    clear_gpu_memory()
                    if torch.cuda.is_available():
                        inputs_cuda = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
                    else:
                        inputs_cuda = inputs

                    with torch.no_grad():
                        generated_ids = model.generate(**inputs_cuda, max_new_tokens=req.max_tokens or 256)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_cuda["input_ids"], generated_ids)
                        ]
                        output_text = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
            finally:
                if inputs_cuda is not None:
                    del inputs_cuda
                clear_gpu_memory()

            return JSONResponse({
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model or "Qwen2.5-VL-3B-Instruct",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop"
                }]
            })

        # Streaming mode (SSE)
        return StreamingResponse(
            generate_openai_sse_stream(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Inference-Start": str(time.time()),
            }
        )

    # Case 2: SYNTHIA native InferPayload (has 'frame' or 'joints')
    if "frame" in body or "joints" in body:
        try:
            payload = InferPayload(**body)
        except Exception:
            payload = InferPayload(
                frame=body.get("frame", ""),
                joints=body.get("joints", {}),
                audio_pcm=body.get("audio_pcm"),
                valid_joints=body.get("valid_joints", []),
                upright_preset=body.get("upright_preset", {}),
                heartbeat=body.get("heartbeat", 0),
                light_state=body.get("light_state", "day"),
                session_id=body.get("session_id", "default"),
                body_type=body.get("body_type", "humanoid"),
                current_goal=body.get("current_goal"),
                current_rung=body.get("current_rung", 0),
                objects_in_world=body.get("objects_in_world", []),
                known_skills=body.get("known_skills", []),
                pending_injection=body.get("pending_injection"),
                motor_program_library=body.get("motor_program_library", []),
                directive_mode=body.get("directive_mode", "free_will"),
                agent_id=body.get("agent_id", "agent_a"),
                contact_forces=body.get("contact_forces"),
                tactile_context=body.get("tactile_context"),
                perception_summary=body.get("perception_summary"),
                gaze_context=body.get("gaze_context"),
                physical_feedback=body.get("physical_feedback")
            )
        last_payload = payload.model_dump()
        return StreamingResponse(
            generate_legacy_stream(payload), 
            media_type="text/plain", 
            headers={"X-Inference-Start": str(time.time())}
        )

    # Case 3: Empty body or ping test
    return JSONResponse({
        "status": "ok", 
        "model": "mock" if MOCK_MODE else "Qwen2.5-VL", 
        "message": "OK",
        "mock_mode": MOCK_MODE,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/health")
@app.get("/")
async def health():
    device_info = "cpu"
    if not MOCK_MODE and model is not None:
        try: device_info = str(next(model.parameters()).device)
        except: device_info = "unknown"
    return {
        "status": "ok", 
        "model": "mock" if MOCK_MODE else "Qwen2.5-VL", 
        "mock_mode": MOCK_MODE,
        "model_loaded": model is not None,
        "device": device_info, 
        "timestamp": datetime.now().isoformat()
    }


# === TUNNEL & LIFECYCLE ===
def setup_tunnel():
    print("Setting up Cloudflare Tunnel...")
    os.system("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared")
    os.system("nohup ./cloudflared tunnel --url http://127.0.0.1:8000 > cloudflared.log 2>&1 &")
    time.sleep(8)
    try:
        with open("cloudflared.log", "r") as f:
            content = f.read()
            match = re.search(r"https://[-a-zA-Z0-9]+\.trycloudflare\.com", content)
            if match:
                print("\n" + "="*70)
                print("✅ TUNNEL READY! Paste this into your SYNTHIA Agent Settings:")
                print(f"👉 {match.group(0)}/infer 👈")
                print("="*70 + "\n")
    except Exception as e:
        print(f"Tunnel setup failed: {e}")

def save_checkpoint():
    if last_payload:
        try:
            with open('/kaggle/working/synthia_session.json', 'w') as f:
                json.dump({'timestamp': datetime.now().isoformat(), **last_payload}, f)
        except: pass

def run_uvicorn_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    print("🧹 Cleaning up old processes...")
    os.system("pkill -f 'uvicorn.*8000' 2>/dev/null || true")
    os.system("fuser -k 8000/tcp 2>/dev/null || true")
    time.sleep(2)

    # 1. START UVICORN FIRST
    print("🚀 Starting Uvicorn on port 8000 in a background thread...")
    threading.Thread(target=run_uvicorn_server, daemon=True).start()

    # 2. Wait for Uvicorn to actually boot and bind to the port
    time.sleep(5)

    # 3. NOW setup the tunnel (it will successfully connect to port 8000)
    setup_tunnel()

    schedule.every(30).minutes.do(save_checkpoint)

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
        os.system("pkill -f 'uvicorn' 2>/dev/null || true")