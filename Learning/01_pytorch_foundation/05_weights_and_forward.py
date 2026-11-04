# =============================================================================
# 05 — WEIGHTS AND THE FORWARD PASS
# =============================================================================
#
# LIKE YOU ARE 5:
# Imagine a machine that takes a photo and tells you "cat" or "dog".
# Inside the machine are thousands of dials (weights).
# When the machine was built, all dials were set randomly. It got everything wrong.
# During TRAINING, someone showed it millions of labelled photos.
# After each wrong answer, the dials were nudged slightly to do better next time.
# After millions of nudges, the dials are set just right. That is a trained model.
#
# When you DOWNLOAD a pretrained model (like RIFE or AnimateDiff), you are
# downloading the final dial settings — the weights. The architecture
# (the machine's structure) is in the Python code. The weights are in the file.
#
# A FORWARD PASS is running one input through the machine to get an output.
# The dials do NOT change during inference. Only during training.
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("WHAT WEIGHTS LOOK LIKE")
print("=" * 60)

layer = nn.Linear(3, 2)
print("Fresh layer (randomly initialised weights):")
print(f"  Weight:\n{layer.weight.data}")
print(f"  Bias:   {layer.bias.data}")

# Manually set weights (like loading a checkpoint would do)
layer.weight.data = torch.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0]])
layer.bias.data   = torch.tensor([0.5, -0.5])
print("\nAfter manually setting weights (simulating a checkpoint load):")
print(f"  Weight:\n{layer.weight.data}")
print(f"  Bias:   {layer.bias.data}")

# Forward pass
x = torch.tensor([3.0, 7.0, 0.0])
out = layer(x)
print(f"\nInput:  {x}")
print(f"Output: {out}")
# Row 0: [1,0,0] @ [3,7,0] + 0.5 = 3.0 + 0.5 = 3.5
# Row 1: [0,1,0] @ [3,7,0] - 0.5 = 7.0 - 0.5 = 6.5

print()
print("=" * 60)
print("SAVING AND LOADING WEIGHTS (state_dict)")
print("=" * 60)
#
# A "state dict" is a Python dictionary mapping parameter names to tensors.
# It is what gets saved to a .pt or .safetensors checkpoint file.
# C# analogy: it is like serialising your model's private fields to a Dictionary<string, float[]>.

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

# Save to file
torch.save(model.state_dict(), "C:/AI-Lab/Learning/01_pytorch_foundation/tiny_weights.pt")
print("Saved weights to tiny_weights.pt")

# Inspect the state dict
state = model.state_dict()
print("\nKeys in state dict:")
for key, tensor in state.items():
    print(f"  {key:25s}  shape={tensor.shape}")

# Load weights into a fresh model
model2 = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
model2.load_state_dict(torch.load("C:/AI-Lab/Learning/01_pytorch_foundation/tiny_weights.pt", weights_only=True))
print("\nLoaded weights into model2. Verifying they match:")
for key in state:
    match = torch.allclose(state[key], model2.state_dict()[key])
    print(f"  {key:25s}  match={match}")

print()
print("=" * 60)
print("INFERENCE MODE  (no gradient tracking)")
print("=" * 60)
#
# During TRAINING, PyTorch tracks every operation to compute gradients.
# That uses memory and time. During INFERENCE you don't need gradients.
# torch.no_grad() turns off that tracking — faster and less VRAM.
# model.eval() also disables dropout and batch norm training behaviour.
#
# C# analogy: it is like disabling an expensive diagnostic observer
# that you only need during development (training), not production (inference).

model.eval()  # set to evaluation mode

x = torch.randn(1, 4)  # batch of 1, 4 features
with torch.no_grad():  # no gradient tracking
    output = model(x)
print(f"Inference output: {output}")
print(f"requires_grad:    {output.requires_grad}")  # False — no gradient tracking

print()
print("=" * 60)
print("TRAINABLE vs FROZEN PARAMETERS")
print("=" * 60)
#
# In fine-tuning (LoRA, adapter training) you often freeze most of the model
# and only train a small part. requires_grad=False means that parameter
# is NOT updated during training — it is frozen.

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Trainable parameters: {trainable}")
print(f"Frozen parameters:    {frozen}")

# Unfreeze just the last layer
for param in model[-1].parameters():  # model[-1] = second Linear
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nAfter unfreezing last layer:")
print(f"Trainable parameters: {trainable}  (only the last Linear layer)")

# =============================================================================
# EXPECTED OUTPUT:
# Keys in state dict:
#   0.weight                   shape=torch.Size([8, 4])
#   0.bias                     shape=torch.Size([8])
#   2.weight                   shape=torch.Size([2, 8])
#   2.bias                     shape=torch.Size([2])
# match=True for all keys
# requires_grad: False
# Frozen parameters: 50
# After unfreezing: Trainable parameters: 18
# =============================================================================
