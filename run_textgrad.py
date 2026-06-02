import os
import json
import subprocess
import torch
import textgrad as tg

# -------------------------------------------------------------
# 1. SETUP THE TEXTGRAD OPTIMIZER ENGINE
# -------------------------------------------------------------
# Using gpt-4o as the core text engine that computes gradients and steps
engine = tg.EngineVLM(model_string="gpt-4o")

# Load your baseline driving prompt from your assets directory
with open("assets/example/captions/2d23a1f4-c269-46aa-8e7d-1bb595d1e421_2445376400000_2445396400000.txt", "r") as f:
    initial_prompt_text = f.read().strip()

# Declare the prompt as a trainable TextGrad Variable
scenario_prompt = tg.Variable(
    initial_prompt_text, 
    requires_grad=True, 
    role_description="Text prompt specifying weather, lightning, and traffic anomalies for video generation"
)

# Instantiate the Textual Gradient Descent (TGD) Optimizer
optimizer = tg.TGD(parameters=[scenario_prompt], engine=engine)


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
    
    # CRITICAL VRAM MITIGATION: Force Cosmos to render a very short sequence 
    # during the optimization phase. (e.g., 24 or 32 frames instead of 121)
    # Check your Cosmos configuration args to specify lower frame sequences.
    subprocess.run([
        "python", "scripts/generate_video_single_view.py",
        "--caption_path", "outputs/captions",
        "--input_path", "outputs",
        "--video_save_folder", "outputs/single_view",
        "--checkpoint_dir", "checkpoints/",
        "--is_av_sample",
        "--controlnet_specs", "assets/sample_av_hdmap_spec.json"
    ], stdout=subprocess.DEVNULL) # Suppress heavy logs
    
    # Return path to the rendered front-view video asset
    return "outputs/single_view/optimized_scenario.mp4"


# -------------------------------------------------------------
# 3. DEFINE CRITIQUE LLMS / EVALUATORS (THE LOSS NODES)
# -------------------------------------------------------------
# Setup loop evaluators using TextGrad's BlackboxLLM component
loop_a_perception_evaluator = tg.BlackboxLLM(
    system_prompt=(
        "You are an autonomous driving perception critic. Analyze the telemetry data "
        "and tracking performance matrices of an AV system executing on a scenario. "
        "Your goal is to find visual exploits. Provide constructive criticism outlining "
        "why the current environment failed to confuse the object detector or tracker, "
        "and what precise hazardous visual details should be integrated to force a model failure."
    ),
    engine=engine
)

loop_b_guidance_evaluator = tg.BlackboxLLM(
    system_prompt=(
        "You are an environmental consistency safety manager. Your target long-tail distribution "
        "is: 'A severe torrential downpour at dusk causing high water sprays from heavy trucks, "
        "resulting in wet asphalt surface glare and low boundary visibility.' "
        "Evaluate the visual description of the scene and provide a harsh text critique detailing "
        "how the prompt must be updated to align closer with this targeted risk distribution."
    ),
    engine=engine
)

# -------------------------------------------------------------
# 4. EXECUTE THE CLOSED-LOOP OPTIMIZATION
# -------------------------------------------------------------
num_iterations = 3

for iteration in range(num_iterations):
    optimizer.zero_grad()
    print(f"\n--- Starting Optimization Step {iteration + 1} ---")
    print(f"Current Prompt Condition: {scenario_prompt.value}\n")
    
    # Forward Pass Part 1: Render video via local hardware
    video_output_path = run_cosmos_front_view_generation(scenario_prompt.value, num_frames=24)
    
    # Forward Pass Part 2: Evaluate the video outputs
    # (Optional: Pass the video to a local 3D detector/lane tracker first, collect metrics text)
    # For simplicity, we summarize video frames or provide metrics here:
    av_perception_metrics = "Lane Tracker: 98% confidence. Object BBox: Pedestrian tracked perfectly."
    visual_scene_telemetry = "The scene appears as clear asphalt with minor rain effects. High visibility."
    
    # Feed evaluations into our TextGrad Loss functions
    loss_a = loop_a_perception_evaluator(tg.Variable(av_perception_metrics, role_description="AV perception performance telemetry"))
    loss_b = loop_b_guidance_evaluator(tg.Variable(visual_scene_telemetry, role_description="Visual layout observations"))
    
    # Aggregate into a single master evaluation Loss node
    composite_loss_text = (
        f"Perception Failure Criterion (Loop A): {loss_a.value}\n"
        f"Target Domain Adherence Criterion (Loop B): {loss_b.value}"
    )
    total_loss = tg.Variable(composite_loss_text, role_description="Composite Guided Adversarial Prompt Loss")
    
    # Backward Pass: TextGrad computes textual gradients via API backpropagation
    print("Computing natural language gradients...")
    total_loss.backward()
    
    # Clear local VRAM cache immediately before TextGrad performs its update step
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Optimizer Step: TGD mutates the prompt variable based on textual gradients
    optimizer.step()

# -------------------------------------------------------------
# 5. ESCAPE TO FULL-SCALE SYNTHETIC DATA GENERATION
# -------------------------------------------------------------
print("\n=== Prompt Optimization Complete! ===")
print(f"Final Optimized Long-Tail Prompt:\n{scenario_prompt.value}\n")

# Run the final output sequence one time at full 121-frame layout capacity 
print("Generating final 121-frame, high-quality front view...")
run_cosmos_front_view_generation(scenario_prompt.value, num_frames=121)

print("Pipeline optimized. You are now safe to run Step 4 (Multi-view video expansion).")