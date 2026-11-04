# =============================================================================
# 02 — CHECKPOINT vs ARCHITECTURE: THE H3 LESSON
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine you have a LEGO instruction manual (architecture = the class definition)
# and a box of LEGO bricks (checkpoint = the .safetensors weights).
# You need BOTH. And the manual must match the bricks.
# If you use the wrong manual for the bricks → pieces don't fit.
#
# THE H3 LESSON (from your investigation):
# MiniMax-H3 looks like an AnimateDiff checkpoint BUT uses a custom architecture.
# DiffusionPipeline.from_pretrained("minimax/H3") would fail because:
#   - It reads model_index.json
#   - Sees "_class_name": "MinimaxH3Pipeline" (or similar custom name)
#   - That class is NOT in diffusers — it's in the model's OWN code
#   - You need trust_remote_code=True, or clone their repo
#
# FOUR WAYS A MODEL CAN SHIP:
# 1. Standard class  — "unet": ["diffusers", "UNet2DConditionModel"] → works with pip diffusers
# 2. Custom class in repo  — "unet": [".", "CustomUNet"] → needs trust_remote_code
# 3. No pipeline, just weights  — just model.safetensors + config.json → need manual load
# 4. GGUF / ONNX  — not in diffusers at all → different loader (llama.cpp, onnxruntime)
#
# =============================================================================

import json
import torch
import torch.nn as nn

print("=" * 60)
print("SAME WEIGHTS, DIFFERENT LOADER CLASS → DIFFERENT RESULT")
print("=" * 60)

# Demonstrate: the same state dict loaded into two different architectures
class ArchitectureA(nn.Module):
    """The correct architecture for these weights."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class ArchitectureB(nn.Module):
    """Wrong architecture — same parameter names but different sizes."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)   # <- wider than A
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Save weights from A
model_a = ArchitectureA()
torch.save(model_a.state_dict(), "C:/AI-Lab/Learning/13_model_investigation/arch_a_weights.pt")

# Try loading into B (wrong architecture)
model_b = ArchitectureB()
try:
    model_b.load_state_dict(
        torch.load("C:/AI-Lab/Learning/13_model_investigation/arch_a_weights.pt",
                   weights_only=True),
        strict=True
    )
    print("Load into wrong arch: succeeded (unexpected)")
except RuntimeError as e:
    print(f"Load into wrong arch FAILED:")
    print(f"  {str(e)[:200]}")
    print()
    print("This is what happens when you use the wrong loader class.")
    print("The .safetensors has fc1 of shape (8,4) but ArchitectureB expects (16,4).")

print()
# Load into correct arch
model_a2 = ArchitectureA()
model_a2.load_state_dict(
    torch.load("C:/AI-Lab/Learning/13_model_investigation/arch_a_weights.pt", weights_only=True)
)
print("Load into correct arch: OK")

print()
print("=" * 60)
print("HOW DiffusionPipeline.from_pretrained DECIDES WHAT CLASS TO USE")
print("=" * 60)

print("""
Step 1: Download model_index.json
Step 2: Read "_class_name" → that is the Pipeline class to instantiate
Step 3: For each component: read its [library, class_name]
Step 4: Import that class from the library
Step 5: Call ComponentClass.from_pretrained(repo, subfolder=<component>)
Step 6: Assemble all components into the Pipeline

If step 4 fails (class not in the installed library):
  → ImportError: "Cannot find class 'WanT2VPipeline' in diffusers"

Fix A: pip install --upgrade diffusers   (if it was added in a newer version)
Fix B: trust_remote_code=True            (runs Python code from the repo)
Fix C: Clone the repo, use its own loader script

Example of a custom _class_name in model_index.json:
""")

custom_index = {
    "_class_name": "WanImageToVideoPipeline",   # added in diffusers 0.33+
    "_diffusers_version": "0.33.0",
    "image_encoder": ["transformers", "CLIPVisionModel"],
    "text_encoder": ["transformers", "UMT5EncoderModel"],
    "tokenizer": ["transformers", "AutoTokenizer"],
    "transformer": ["diffusers", "WanTransformer3DModel"],   # DiT, not UNet
    "vae": ["diffusers", "AutoencoderKLWan"],
    "scheduler": ["diffusers", "UniPCMultistepScheduler"],
}
print(json.dumps(custom_index, indent=2))
print()
print("Notice: 'transformer' instead of 'unet' → this is a Diffusion Transformer (DiT).")
print("And the VAE class is 'AutoencoderKLWan' (Wan-specific), not standard 'AutoencoderKL'.")
print("→ You need diffusers >= 0.33.0 OR trust_remote_code=True")

print()
print("=" * 60)
print("trust_remote_code: WHAT IT DOES AND THE RISK")
print("=" * 60)
print("""
When trust_remote_code=True:
  1. HuggingFace downloads Python files from the repo (modeling_*.py, pipeline_*.py)
  2. Those files are EXECUTED on your machine (they run arbitrary Python)
  3. The custom classes in those files are used to load the model

RISK:
  A malicious repo could include code that:
  - Reads your files, SSH keys, tokens
  - Downloads and runs more code
  - Makes network requests
  - Corrupts files

WHEN IT IS SAFE:
  - Official org accounts (minimax-ai, Wan-AI, stabilityai, meta-llama)
  - Repos with many downloads, public scrutiny
  - You have REVIEWED the modeling_*.py files yourself

WHEN TO REFUSE:
  - Random user repo with 0 stars and no README
  - Any repo that asks you to run its install script
  - Any repo where the .py files touch os.system() or subprocess

HOW TO REVIEW BEFORE TRUSTING:
  import huggingface_hub
  api = huggingface_hub.HfApi()
  files = api.list_repo_files("owner/model")
  # Download and read each .py file before passing trust_remote_code=True
""")

print("=" * 60)
print("THE 'STRICT=FALSE' ESCAPE HATCH")
print("=" * 60)
print("""
Sometimes a checkpoint has EXTRA keys (fine-tuning added adapters)
or MISSING keys (a component was separated).

load_state_dict(state, strict=False) ignores:
  - Missing keys: weights not in the checkpoint (stay as random init)
  - Unexpected keys: weights in checkpoint not in the model (silently ignored)

Use with CAUTION: a silent mismatch means parts of your model are random.

Example: loading a base SD 1.5 checkpoint into an AnimateDiff UNet.
The motion module weights are NOT in the SD 1.5 checkpoint.
  model.load_state_dict(sd15_weights, strict=False)
  # Motion module layers stay random — then load motion adapter separately.
""")

# =============================================================================
# EXPECTED OUTPUT:
# Load into wrong arch FAILED:
#   size mismatch for fc1.weight: copying a param with shape (8, 4) from checkpoint,
#   the shape in current model is torch.Size([16, 4]).
# Load into correct arch: OK
# =============================================================================
