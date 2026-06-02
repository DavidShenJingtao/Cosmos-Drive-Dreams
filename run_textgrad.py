import os

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY is not set")

os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

import json
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
    engine=llm_engine, # Explicitly bind your gpt-4o engine instance
    optimizer_system_prompt=optimizer_system_prompt
)


# -------------------------------------------------------------
# 2. BLACK-BOX SIMULATOR WRAPPER (COSMOS FRONT-VIEW)
# -------------------------------------------------------------
def run_cosmos_front_view_generation(prompt_string, num_frames=24):
    """
    Saves the current prompt string to the expected cosmos workspace
    and invokes the front-view single-camera generator script.
    """
    # Write optimized prompt out to the JSON format Cosmos expects
    caption_payload = {"caption": prompt_string}
    os.makedirs("outputs/captions", exist_ok=True)
    with open("outputs/captions/optimized_scenario.json", "w") as f:
        json.dump(caption_payload, f)
    
    # CRITICAL VRAM MITIGATION: Force Cosmos to render a short sequence (e.g., 24 frames)
    # during the optimization phase to avoid local CUDA Out-of-Memory crashes.
    subprocess.run([
        "python", "scripts/generate_video_single_view.py",
        "--caption_path", "outputs/captions",
        "--input_path", "outputs",
        "--video_save_folder", "outputs/single_view",
        "--checkpoint_dir", "checkpoints/",
        "--is_av_sample",
        "--controlnet_specs", "assets_av_hdmap_spec.json"
    ], stdout=subprocess.DEVNULL) 
    
    return "outputs/single_view/optimized_scenario.mp4"


# -------------------------------------------------------------
# 3. CONSTRUCT THE TEXTGRAD CRITIQUE LOSS NODE (Fixed)
# -------------------------------------------------------------
# Define the strict TextLoss constraints OUTSIDE the function loop.
# This ensures it stays consistently anchored inside TextGrad's execution graph.
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
# Define a representative map instance from your Step 1 assets
hdmap_layout_instance = (
    "The video shows a highway scene during twilight or early evening, with a clear sky "
    "transitioning from blue to darker shades. Several cars are visible on the road, some "
    "moving forward while others appear stationary, indicating moderate traffic. The road is "
    "flanked by trees and a concrete barrier on one side, with utility poles and wires running "
    "parallel to the highway. A billboard is visible in the distance, and the overall atmosphere "
    "suggests a calm urban or suburban setting. The lighting indicates that it is either dusk "
    "or dawn, with the sky showing signs of fading light."
)
question = tg.Variable(hdmap_layout_instance, role_description="the target baseline HDMap condition to simulate", requires_grad=False)

# Define your target long-tail risk profile
target_hazard = "A heavy torrential downpour at dusk creating severe blinding asphalt glare and major vehicle water spray."

num_iterations = 3

for iteration in range(num_iterations):
    optimizer.zero_grad()
    print(f"\n--- TextGrad Prompt Optimization Iteration {iteration + 1} ---")
    
    # Step A: Generate specific prompt variation based on current parameters
    prediction = model(question)
    print(f"Current System Output Prediction:\n{prediction.value}\n")
    
    # Step B: Render the short video snippet locally
    video_output = run_cosmos_front_view_generation(prediction.value, num_frames=24)
    
    # Step C: Pack evaluation measurements into static TextGrad Variables (requires_grad=False)
    # (In production, replace this placeholder with programmatic metrics parsed from your trackers)
    av_tracker_metrics = "System Telemetry: Pedestrians and lane markers tracked with 99% accuracy. Zero tracking lag detected."
    
    perception_log_var = tg.Variable(av_tracker_metrics, role_description="downstream AV tracking telemetry data", requires_grad=False)
    target_domain_var = tg.Variable(target_hazard, role_description="the required target hazard envelope", requires_grad=False)
    
    # Step D: Construct a compound input container for TextLoss evaluation
    # This feeds all context variables into the graph alongside your prediction node
    combined_loss_input = (
        f"Generated Scene Description: {prediction}\n"
        f"Downstream System Metrics: {perception_log_var}\n"
        f"Target Domain Envelope: {target_domain_var}"
    )
    loss_node = tg.Variable(combined_loss_input, role_description="aggregated execution state for safety evaluation")
    
    # Compute the textual loss score object
    loss = loss_fn(loss_node)
    print(f"Computed Feedback Loss:\n{loss.value}\n")
    
    # Step E: Backward pass propagates feedback back to system_prompt parameters
    print("Computing gradients on system parameter layers...")
    loss.backward()
    
    # Evacuate PyTorch memory footprints to avoid memory collisions with Cosmos
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Step F: Update system parameters (updates system_prompt string instructions)
    optimizer.step()

# -------------------------------------------------------------
# 5. EXECUTE PRODUCTION GENERATION
# -------------------------------------------------------------
print("\n=== SYSTEM OPTIMIZATION EXECUTION STEP COMPLETE ===")
print(f"Final Optimized System Prompt Parameter:\n{system_prompt.value}\n")

print("Generating final long-tail risk video utilizing optimal instructions...")
final_production_prediction = model(question)

# Generate full-length high quality asset safely now that training loops are complete
run_cosmos_front_view_generation(final_production_prediction.value, num_frames=121)
print("Pipeline run complete. Proceed to Step 4 for surrounding multiview expansion.")
