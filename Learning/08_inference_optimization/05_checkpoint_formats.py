# =============================================================================
# 05 — CHECKPOINT FORMATS: .pt, .safetensors, .onnx, .gguf
# =============================================================================
#
# LIKE YOU ARE 5:
# A model checkpoint is a saved file containing the weights.
# Different formats have different:
# - Safety (can the file run arbitrary code?)
# - Size (is it compressed?)
# - Compatibility (which runtimes can load it?)
#
# .pt / .pth      — PyTorch native. Can contain arbitrary Python objects (security risk!).
# .safetensors    — Safe format by HuggingFace. Only tensors, no code execution.
#                   Fast memory-mapped loading. PREFERRED for sharing.
# .ckpt           — Older Stable Diffusion format. Same as .pt but different convention.
# .onnx           — Open Neural Network Exchange. Runs on ONNX Runtime (CPU/GPU/edge).
# .gguf           — llama.cpp format. Quantised (INT4/INT8). Runs on CPU. Fast.
# .bin            — Generic binary. HuggingFace sometimes uses it for sharded models.
#
# =============================================================================

import torch
import torch.nn as nn
import os

print("=" * 60)
print("PYTORCH NATIVE FORMAT (.pt / .pth)")
print("=" * 60)
#
# torch.save() can save ANYTHING: a model, a dict, a list, a custom class.
# It uses pickle internally.
# torch.load() uses pickle to deserialise — can execute arbitrary code.
# NEVER load a .pt file from an untrusted source!

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

model = TinyModel()
save_path = "C:/AI-Lab/Learning/08_inference_optimization/tiny_model.pt"

# Save the state dict (PREFERRED — only saves the weights, not the class)
torch.save(model.state_dict(), save_path)
print(f"Saved state_dict to {save_path}")

# Load it back
loaded = TinyModel()
loaded.load_state_dict(torch.load(save_path, weights_only=True))
print(f"Loaded. Parameters match: {all(torch.allclose(a, b) for a, b in zip(model.parameters(), loaded.parameters()))}")
print()
print("weights_only=True: only loads tensors, refuses to unpickle arbitrary objects.")
print("Always use weights_only=True when loading from files you didn't create yourself.")

print()
print("=" * 60)
print("SAFETENSORS FORMAT")
print("=" * 60)

try:
    from safetensors.torch import save_file, load_file

    state = model.state_dict()
    safe_path = "C:/AI-Lab/Learning/08_inference_optimization/tiny_model.safetensors"
    save_file(state, safe_path)
    print(f"Saved safetensors to {safe_path}")

    loaded_state = load_file(safe_path)
    print(f"Keys: {list(loaded_state.keys())}")
    for key, tensor in loaded_state.items():
        print(f"  {key}: {tensor.shape}  dtype={tensor.dtype}")

    print()
    print("Advantages of safetensors:")
    print("  - No arbitrary code execution (100% safe to load from internet)")
    print("  - Memory-mapped loading (loads only what you need, instantly)")
    print("  - 2-5x faster to load than .pt for large models")
    print("  - AnimateDiff motion module files use .safetensors")

except ImportError:
    print("safetensors not installed. Run: pip install safetensors")
    print()
    print("Usage:")
    print("  from safetensors.torch import save_file, load_file")
    print("  save_file(model.state_dict(), 'model.safetensors')")
    print("  state = load_file('model.safetensors')")

print()
print("=" * 60)
print("ONNX FORMAT (.onnx)")
print("=" * 60)
#
# ONNX (Open Neural Network Exchange):
# - Language-agnostic graph format (can run from C++, C#, Java, etc.)
# - Works with ONNX Runtime (CPU/GPU/mobile/edge)
# - Export: torch.onnx.export() — traces the model and saves the graph
# - Load: onnxruntime.InferenceSession()
# - Use case: production deployment, edge devices, non-Python runtimes

try:
    import io
    dummy_input = torch.randn(1, 4)
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        dummy_input,
        buf,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print(f"ONNX export succeeded. Serialised size: {buf.tell() / 1024:.1f} KB")
    print("Run with: onnxruntime.InferenceSession('model.onnx')")
except Exception as e:
    print(f"ONNX export: {e}")

print()
print("=" * 60)
print("GGUF FORMAT (.gguf) — for llama.cpp")
print("=" * 60)
print("""
GGUF (GPT-Generated Unified Format) is used by llama.cpp:
- Stores quantised (INT4/INT8) weights of LLMs
- Can run on CPU with reasonable speed (no GPU needed)
- Format: a metadata header + tensor blobs
- Variants: Q4_K_M (4-bit), Q5_K_M (5-bit), Q8_0 (8-bit), F16 (16-bit)

How to get GGUF files:
  HuggingFace search: "[model name] GGUF TheBloke"
  Direct URL pattern: huggingface.co/TheBloke/Llama-3-8B-GGUF

Load in Python with llama-cpp-python:
  from llama_cpp import Llama
  llm = Llama(model_path="llama-3-8b.Q4_K_M.gguf", n_ctx=4096)
  output = llm("Once upon a time", max_tokens=100)
""")

print()
print("=" * 60)
print("WHICH FORMAT FOR WHICH TASK")
print("=" * 60)
print(f"""
{'Format':<15} {'When to use':<45} {'Safety'}
{'-'*70}
{'.safetensors':<15} {'Normal HuggingFace/diffusers loading':<45} {'Safe ✓'}
{'.pt/.pth':<15} {'Your own models, training checkpoints':<45} {'Only trust own files'}
{'.ckpt':<15} {'Older SD models from CivitAI etc.':<45} {'Risky — use safetensors instead'}
{'.onnx':<15} {'Production/non-Python deployment':<45} {'Safe ✓'}
{'.gguf':<15} {'LLMs on CPU (llama.cpp)':<45} {'Safe ✓'}
""")

# =============================================================================
# EXPECTED OUTPUT:
# Loaded. Parameters match: True
# Keys: ['fc.weight', 'fc.bias']
# ONNX export succeeded
# =============================================================================
