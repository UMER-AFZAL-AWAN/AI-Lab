# =============================================================================
# 04 — INSPECTING A MODEL: SEEING INSIDE THE BLACK BOX
# =============================================================================
#
# LIKE YOU ARE 5:
# When you download a model from HuggingFace, it is a black box.
# "What is inside it? How many layers? How big are the weights?"
# This file shows you how to look inside any PyTorch model.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("CREATING A MODEL TO INSPECT")
print("=" * 60)

class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings   = nn.Embedding(1000, 128)
        self.encoder      = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.attn         = nn.MultiheadAttention(128, 8, batch_first=True)
        self.norm         = nn.LayerNorm(128)
        self.output_head  = nn.Linear(128, 10)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.encoder(x)
        x, _ = self.attn(x, x, x)
        x = self.norm(x[:, 0])
        return self.output_head(x)

model = ExampleModel()
print(model)   # prints the full module tree

print()
print("=" * 60)
print("COUNTING PARAMETERS")
print("=" * 60)

total      = sum(p.numel() for p in model.parameters())
trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen     = total - trainable
print(f"Total parameters:     {total:>12,}")
print(f"Trainable:            {trainable:>12,}")
print(f"Frozen:               {frozen:>12,}")
print()

# Break down by submodule
print("Per-module breakdown:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    pct    = params / total * 100
    print(f"  {name:<20} {params:>10,} params  ({pct:.1f}%)")

print()
print("=" * 60)
print("INSPECTING THE STATE DICT")
print("=" * 60)

state = model.state_dict()
print(f"Number of tensors in state dict: {len(state)}")
print()
print(f"{'Parameter name':<45} {'Shape':>25}  {'dtype':>8}  {'MB':>6}")
print("-" * 90)
total_mb = 0
for name, tensor in state.items():
    mb = tensor.element_size() * tensor.nelement() / 1024**2
    total_mb += mb
    print(f"  {name:<43} {str(tensor.shape):>25}  {str(tensor.dtype):>8}  {mb:>6.3f}")
print(f"{'TOTAL':>80}  {total_mb:>6.3f}")

print()
print("=" * 60)
print("INSPECTING A SPECIFIC LAYER")
print("=" * 60)

layer = model.encoder[0]   # first Linear layer in encoder
print(f"Layer: {layer}")
print(f"  Weight: {layer.weight.shape}  min={layer.weight.min():.4f}  max={layer.weight.max():.4f}")
print(f"  Bias:   {layer.bias.shape}    mean={layer.bias.mean():.4f}")
print(f"  requires_grad: {layer.weight.requires_grad}")

print()
print("=" * 60)
print("INSPECTING THE DIFFUSION UNET (conceptual)")
print("=" * 60)
#
# When you load a real AnimateDiff UNet, you can inspect it the same way.
# Here's what you'd find:

print("""
# Example: loading and inspecting AnimateDiff UNet
from diffusers import UNet2DConditionModel

unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    torch_dtype=torch.float16
)

# Count parameters
total = sum(p.numel() for p in unet.parameters())
print(f"UNet parameters: {total / 1e9:.2f} B")
# -> ~860M parameters

# See the structure
print(unet)
# -> UNet2DConditionModel(
#     conv_in: Conv2d(4, 320, kernel_size=3, padding=1)
#     down_blocks: ModuleList of 4 DownBlock2D / CrossAttnDownBlock2D
#     mid_block: UNetMidBlock2DCrossAttn
#     up_blocks: ModuleList of 4 UpBlock2D / CrossAttnUpBlock2D
#     conv_out: Conv2d(320, 4, kernel_size=3, padding=1)
#    )

# Inspect the first down block
block = unet.down_blocks[0]
print(block)
# -> CrossAttnDownBlock2D with:
#     resnets: ResnetBlock2D ×2
#     attentions: Transformer2DModel ×2
#     downsamplers: Downsample2D ×1

# See all attention layers
for name, module in unet.named_modules():
    if isinstance(module, nn.MultiheadAttention):
        print(f"{name}: embed_dim={module.embed_dim}, num_heads={module.num_heads}")
""")

print()
print("=" * 60)
print("FORWARD HOOKS: INTERCEPTING ACTIVATIONS")
print("=" * 60)
#
# Hooks let you intercept any layer's input/output during a forward pass.
# Useful for:
# - Debugging: "what does this layer actually output?"
# - Feature extraction: "I want the intermediate features, not the final output"
# - Visualisation: "show me the attention weights"

activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on each child module
handles = []
for name, module in model.named_children():
    h = module.register_forward_hook(hook_fn(name))
    handles.append(h)

# Run a forward pass
ids = torch.randint(0, 1000, (2, 10))
with torch.no_grad():
    out = model(ids)

print("Captured activations:")
for name, act in activations.items():
    print(f"  {name:<20}: {act.shape}")

# Remove hooks after use (important! hooks cause memory leaks if not removed)
for h in handles:
    h.remove()
print("\nHooks removed.")

# =============================================================================
# EXPECTED OUTPUT:
# Total parameters: 426,890 (approximate)
# embeddings: ~128,000 params
# Captured activations for each submodule
# =============================================================================
