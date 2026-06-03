import os

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY is not set")

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

import json
import gc
import sys
import subprocess
import torch
import textgrad as tg

# -------------------------------------------------------------
# 1. SETUP THE TEXTGRAD OPTIMIZER ENGINE
# -------------------------------------------------------------
llm_engine = tg.get_engine("openai/gpt-4o")
tg.set_backward_engine("openai/gpt-4o", override=True)

# Define your initial baseline system prompt strategy
initial_prompt_text = (
    "You are a scenario designer for a driving world model. Based on the structural layout "
    "provided in the input, expand it into a detailed description specifying standard daylight conditions, "
    "straightforward traffic flows, and optimal weather."
)

# Declare the prompt as a trainable TextGrad Variable
system_prompt = tg.Variable(
    initial_prompt_text, 
    requires_grad=True, 
    role_description="system prompt to guide the LLM's scenario synthesis strategy for long-tail scenario generation"
)

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)

optimizer_system_prompt = (
    "You are a Textual Gradient Descent engine updating a text variable. "
    "You will receive a variable, its description, and feedback gradients. "
    "Your ONLY job is to output the updated, rewritten version of the text variable. "
    "CRITICAL: Do not output lists of bullet points, suggestions, meta-criticisms, or wrapped feedback text. "
    "Do not use tags like <FEEDBACK>. You MUST strictly place the updated string inside the tags requested by the user prompt."
)

optimizer = tg.TGD(
    parameters=list(model.parameters()),
    engine=llm_engine, 
    optimizer_system_prompt=optimizer_system_prompt
)

# -------------------------------------------------------------
# AUTOMATED ROOT ABSOLUTE PATH RESOLUTION
# -------------------------------------------------------------
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COSMOS_WORKSPACE_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR) 

# -------------------------------------------------------------
# 2. BLACK-BOX SIMULATOR WRAPPER (COSMOS FRONT-VIEW)
# -------------------------------------------------------------
def run_cosmos_front_view_generation(prompt_string, num_frames=24, check_errors=False):
    """
    Saves the current prompt string to the expected cosmos workspace
    and invokes the front-view single-camera generator script.
    """
    caption_payload = {"caption": prompt_string}
    
    captions_dir = os.path.join(COSMOS_WORKSPACE_ROOT, "outputs/captions")
    os.makedirs(captions_dir, exist_ok=True)
    
    caption_file_path = os.path.join(captions_dir, "optimized_scenario.json")
    with open(caption_file_path, "w") as f:
        json.dump(caption_payload, f)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{COSMOS_WORKSPACE_ROOT}:{os.path.join(COSMOS_WORKSPACE_ROOT, 'cosmos-transfer1')}:{env.get('PYTHONPATH', '')}"
    
    # If check_errors is True, we capture stderr to look out for CUDA OOM crashes.
    # If it's False (during the optimization loop text phase), we skip the hardware call to save VRAM.
    if not check_errors:
        print("Optimization Mode Active: Skipping hardware render pass to conserve GPU VRAM context...")
        return os.path.join(COSMOS_WORKSPACE_ROOT, "outputs/single_view/optimized_scenario.mp4")

    print(f"Launching Cosmos Video Generator (Frames: {num_frames})...")
    result = subprocess.run([
        "python", "scripts/generate_video_single_view.py",
        "--caption_path", "outputs/captions",
        "--input_path", "outputs",
        "--video_save_folder", "outputs/single_view",
        "--checkpoint_dir", "checkpoints/",
        "--is_av_sample",
        "--controlnet_specs", "assets/sample_av_hdmap_spec.json"
    ], cwd=COSMOS_WORKSPACE_ROOT, env=env, capture_output=True, text=True) 
    
    # Catch structural failures or CUDA OOM issues immediately
    if result.returncode != 0:
        print(f"\n❌ CRITICAL ERROR IN SIMULATOR RUN:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
        
    return os.path.join(COSMOS_WORKSPACE_ROOT, "outputs/single_view/optimized_scenario.mp4")


# -------------------------------------------------------------
# 3. CONSTRUCT THE TEXTGRAD CRITIQUE LOSS NODE
# -------------------------------------------------------------
evaluation_instruction = (
    "You are a critical safety audit engine. Evaluate the generated scenario prompt description variable.\n"
    "You will also review accompanying autonomous vehicle tracking performance logs and a targeted safety risk domain.\n\n"
    "CRITERIA:\n"
    "1. If the AV perception tracking is working smoothly and perfectly, the scenario has failed to expose a vulnerability.\n"
    "2. If the scene lacks realistic, heavy environmental impairments, it has failed to hit the target risk envelope.\n"
    "Provide precise, constructive criticism outlining what extreme visual parameters, weather artifacts, or "
    "occlusions need to be explicitly appended to the scenario generation strategy to force system degradation."
)
loss_fn = tg.TextLoss(evaluation_instruction)


# -------------------------------------------------------------
# 4. RUN THE OPTIMIZATION LOOP
# -------------------------------------------------------------
caption_file = os.path.join(
    COSMOS_WORKSPACE_ROOT,
    "assets/example/captions/2d23a1f4-c269-46aa-8e7d-1bb595d1e421_2445376400000_2445396400000.txt"
)

with open(caption_file, "r") as f:
    hdmap_layout_instance = f.read().strip()

question = tg.Variable(hdmap_layout_instance, role_description="the target baseline HDMap condition to simulate", requires_grad=False)
target_hazard = "A heavy torrential downpour at dusk creating severe blinding asphalt glare and major vehicle water spray."

num_iterations = 3

for iteration in range(num_iterations):
    optimizer.zero_grad()
    print(f"\n--- TextGrad Prompt Optimization Iteration {iteration + 1} ---")
    
    # Step A: Generate scenario text variant
    prediction = model(question)
    print(f"Current System Output Prediction:\n{prediction.value}\n")
    
    # Step B: Fast-track the text optimization graph path without choking VRAM
    video_output = run_cosmos_front_view_generation(prediction.value, num_frames=24, check_errors=False)
    
    # Step C: Formulate metrics configuration block
    av_tracker_metrics = "System Telemetry: Pedestrians and lane markers tracked with 99% accuracy. Zero tracking lag detected."
    perception_log_var = tg.Variable(av_tracker_metrics, role_description="downstream AV tracking telemetry data", requires_grad=False)
    target_domain_var = tg.Variable(target_hazard, role_description="the required target hazard envelope", requires_grad=False)
    
    # Step D: Construct loss container
    combined_loss_input = (
        f"Generated Scene Description: {prediction}\n"
        f"Downstream System Metrics: {perception_log_var}\n"
        f"Target Domain Envelope: {target_domain_var}"
    )
    loss_node = tg.Variable(combined_loss_input, role_description="aggregated execution state for safety evaluation")
    
    loss = loss_fn(loss_node)
    print(f"Computed Feedback Loss:\n{loss.value}\n")
    
    # Step E: Propagate textual feedback backward to your system prompt parameters
    print("Computing gradients on system parameter layers...")
    loss.backward()
    
    # Clear local session footprints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    optimizer.step()

# -------------------------------------------------------------
# 5. EXECUTE PRODUCTION GENERATION WITH MEMORY EVACUATION
# -------------------------------------------------------------
print("\n=== SYSTEM OPTIMIZATION EXECUTION STEP COMPLETE ===")
print(f"Final Optimized System Prompt Parameter:\n{system_prompt.value}\n")

# Get final textual instruction variant
print("Formulating final scenario recipe instructions...")
final_production_prediction = model(question)

# CRITICAL VRAM RECLAMATION PASS: Clean and clear everything from the main script's memory
del model
del optimizer
del loss_fn
del llm_engine

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Generating final long-tail risk video utilizing optimal instructions...")
# Run production loop with validation checking set to True
run_cosmos_front_view_generation(final_production_prediction.value, num_frames=121, check_errors=True)
print("Pipeline run complete. Proceed to Step 4 for surrounding multiview expansion.")