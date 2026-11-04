# =============================================================================
# 04 — nn.Module: THE BUILDING BLOCK OF EVERY NEURAL NETWORK
# =============================================================================
#
# LIKE YOU ARE 5:
# Every neural network in PyTorch is a class that inherits from nn.Module.
# It has two things:
#   1. Parameters (weights) — numbers that were learned during training.
#   2. forward() — what to do with an input to produce an output.
#
# C# ANALOGY:
# nn.Module is like an abstract base class with one required method: forward().
# Parameters are like private fields, but PyTorch tracks them automatically.
# When you call model(x), PyTorch calls model.forward(x) for you.
#
#   public abstract class NeuralLayer
#   {
#       protected float[] weights;  // learned parameters
#       public abstract float[] Forward(float[] input);
#   }
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("THE SIMPLEST NEURAL NETWORK: ONE LINEAR LAYER")
print("=" * 60)
#
# A Linear layer does one thing: output = input @ weights.T + bias
# It is exactly the matrix multiply we saw in 02_operations.py.
#
# nn.Linear(in_features, out_features)
#   in_features:  how many numbers come IN
#   out_features: how many numbers come OUT
# It automatically creates weight and bias tensors and fills them randomly.

layer = nn.Linear(in_features=3, out_features=2)
print(f"Layer: {layer}")
print(f"Weight shape: {layer.weight.shape}")  # (out=2, in=3)
print(f"Bias shape:   {layer.bias.shape}")    # (out=2,)
print(f"Weight values:\n{layer.weight}")
print(f"Bias values:  {layer.bias}")

# Run an input through it
x = torch.tensor([1.0, 2.0, 3.0])  # 3 input features
out = layer(x)                       # calls forward() internally
print(f"\nInput:  {x}  shape={x.shape}")
print(f"Output: {out}  shape={out.shape}")
# Output: 2 numbers (one for each output feature)

print()
print("=" * 60)
print("BUILDING YOUR OWN MODULE")
print("=" * 60)
#
# Every model you encounter (RIFE, AnimateDiff, Llama) is built this way.
# IFBlock in RIFE is an nn.Module. IFNet contains multiple IFBlocks.
# They are just classes with weights and a forward() method.

class TinyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()           # always call super().__init__()
        # Define the layers (these hold the learned weights)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.relu   = nn.ReLU()      # activation function (no weights, just math)

    def forward(self, x):
        # This runs every time you call model(x)
        x = self.layer1(x)   # matrix multiply + bias
        x = self.relu(x)     # apply ReLU (zero out negatives)
        x = self.layer2(x)   # matrix multiply + bias
        return x

model = TinyNetwork(input_size=4, hidden_size=8, output_size=2)
print(f"Model:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params}")
# layer1: (4*8 weights) + (8 bias) = 40
# layer2: (8*2 weights) + (2 bias) = 18
# total: 58

# Forward pass
x = torch.randn(4)         # random 4-feature input
output = model(x)
print(f"Input:  {x.round(decimals=2)}")
print(f"Output: {output.round(decimals=4)}")

print()
print("=" * 60)
print("PARAMETERS vs BUFFERS")
print("=" * 60)
#
# Parameters: learned during training. Stored with the model. Affect gradients.
# Buffers: fixed values stored with the model. NOT learned. NOT gradients.
# Example: batch normalisation running mean is a buffer, not a parameter.

for name, param in model.named_parameters():
    print(f"  param: {name:20s}  shape={str(param.shape):20s}  requires_grad={param.requires_grad}")

print()
print("=" * 60)
print("DEVICE MANAGEMENT")
print("=" * 60)
#
# Move the whole model to GPU with one call.
# Every parameter and buffer moves together.

model_gpu = model.to("cuda")
x_gpu = x.to("cuda")
output_gpu = model_gpu(x_gpu)
print(f"Output on GPU: {output_gpu}")
print(f"Output device: {output_gpu.device}")

# Move back to CPU for inspection
output_cpu = output_gpu.cpu()
print(f"Output on CPU: {output_cpu}")

# =============================================================================
# EXPECTED OUTPUT:
# Layer: Linear(in_features=3, out_features=2, bias=True)
# Weight shape: torch.Size([2, 3])
# Total parameters: 58
#   param: layer1.weight         shape=torch.Size([8, 4])  requires_grad=True
#   param: layer1.bias           shape=torch.Size([8])     requires_grad=True
#   param: layer2.weight         shape=torch.Size([2, 8])  requires_grad=True
#   param: layer2.bias           shape=torch.Size([2])     requires_grad=True
# =============================================================================
