import os
import json
import subprocess

# 1. SETUP LOGICAL ABSOLUTE PATH RESOLUTION
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# If running inside 'textgrad' folder, step up. If running from root, keep root.
if os.path.basename(CURRENT_SCRIPT_DIR) == "textgrad":
    COSMOS_WORKSPACE_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
else:
    COSMOS_WORKSPACE_ROOT = CURRENT_SCRIPT_DIR

# 2. LOAD YOUR OPTIMIZED SYSTEM PROMPT STRUCT
# Adjust this path if you saved the prompt text file somewhere else
prompt_file_path = os.path.join(COSMOS_WORKSPACE_ROOT, "textgrad", "optimized_system_prompt.txt")
with open(prompt_file_path, "r") as f:
    optimized_system_prompt = f.read().strip()

# 3. READ THE BASELINE MAP DESCRIPTION INSTANCE
caption_file = os.path.join(
    COSMOS_WORKSPACE_ROOT, 
    "assets/example/captions/2d23a1f4-c269-46aa-8e7d-1bb595d1e421_2445376400000_2445396400000.txt"
)
with open(caption_file, "r") as f:
    hdmap_layout_instance = f.read().strip()

print("Formulating final scenario recipe instructions...")
# Combine your optimized prompt system architecture with your map asset layout instance
final_prompt_string = f"{optimized_system_prompt}\n\nInput Layout Context:\n{hdmap_layout_instance}"

# 4. EXPORT JSON FOR COSMOS GENERATION WORKSPACE
captions_dir = os.path.join(COSMOS_WORKSPACE_ROOT, "outputs/captions")
os.makedirs(captions_dir, exist_ok=True)

caption_file_path = os.path.join(captions_dir, "2d23a1f4-c269-46aa-8e7d-1bb595d1e421_2445376400000_2445396400000.json")
with open(caption_file_path, "w") as f:
    json.dump({"caption": final_prompt_string}, f)

# 5. ENVIRONMENT SETUP FOR INDEPENDENT SUBPROCESS
env = os.environ.copy()
env["PYTHONPATH"] = f"{COSMOS_WORKSPACE_ROOT}:{os.path.join(COSMOS_WORKSPACE_ROOT, 'cosmos-transfer1')}:{env.get('PYTHONPATH', '')}"

print("Generating final long-tail risk video utilizing optimal instructions...")
print("Launching Cosmos Video Generator (Frames: 121)...")

# Trigger production loop execution pass with full environment VRAM headroom
subprocess.run([
    "python", "scripts/generate_video_single_view.py",
    "--caption_path", "outputs/captions",
    "--input_path", "outputs",
    "--video_save_folder", "outputs/single_view",
    "--checkpoint_dir", "checkpoints/",
    "--is_av_sample",
    "--controlnet_specs", "assets/sample_av_hdmap_spec.json",

    "--offload_text_encoder_model",
    "--offload_guardrail_models"
], cwd=COSMOS_WORKSPACE_ROOT, env=env)

print("\nPipeline run complete. Proceed to Step 4 for surrounding multiview expansion.")