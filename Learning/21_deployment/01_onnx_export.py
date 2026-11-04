# =============================================================================
# 21_deployment / 01 — ONNX EXPORT AND OPTIMIZATION
# =============================================================================
#
# LIKE YOU ARE 5:
# ONNX is like a universal power adapter for neural networks.
# A model trained in PyTorch can be exported to ONNX format,
# and then run in C#, Java, C++, mobile, edge devices — anything.
# ONNX also makes inference FASTER by optimising the computation graph.
#
# C# ANALOGY:
# Like compiling C# to IL (Intermediate Language):
#   .cs source → .dll (IL) → runs on any .NET runtime
#   PyTorch model → .onnx (graph) → runs on any ONNX runtime
# Microsoft.ML.OnnxRuntime is the C# ONNX runtime.
#
# =============================================================================

import torch
import torch.nn as nn
import os

torch.manual_seed(42)

print("=" * 60)
print("PART 1: WHAT IS ONNX")
print("=" * 60)
print("""
ONNX (Open Neural Network Exchange):
  An open format for representing machine learning models.
  Defines a common set of operators (Conv, MatMul, Relu, Softmax, ...)
  and a file format for storing the graph.

Why export to ONNX?
  1. Deployment: run with ONNX Runtime (faster than PyTorch for inference)
  2. Cross-platform: Python, C++, C#, Java, mobile, edge
  3. Optimisation: graph-level fusion (Conv+BN+ReLU → single op)
  4. Hardware: TensorRT (Nvidia), OpenVINO (Intel), CoreML (Apple)

ONNX Runtime:
  Microsoft's runtime for ONNX models.
  Typically 1.5×–4× faster than PyTorch for CPU inference.
  With TensorRT backend: even faster on NVIDIA GPUs.

Install:
  pip install onnx onnxruntime         ← CPU inference
  pip install onnxruntime-gpu          ← GPU inference (CUDA)
""")

print("=" * 60)
print("PART 2: EXPORTING A MODEL TO ONNX")
print("=" * 60)

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
    def forward(self, x):
        return self.net(x)

model = TinyClassifier().eval()   # always .eval() before export

# Dummy input — same shape as real inputs
dummy_input = torch.randn(1, 4)   # batch_size=1, 4 features

onnx_path = "tiny_classifier.onnx"

# torch.onnx.export traces the model's forward pass
# and records every operation into a computation graph
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17,         # ONNX opset (17 = current standard)
    input_names=["input"],    # names for input nodes in the graph
    output_names=["output"],  # names for output nodes
    dynamic_axes={            # allow variable batch size
        "input":  {0: "batch_size"},
        "output": {0: "batch_size"},
    },
    verbose=False,
)

print(f"Exported model to: {onnx_path}")
print(f"File size: {os.path.getsize(onnx_path) / 1024:.1f} KB")

# Verify the ONNX model is valid
try:
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("✓ ONNX model is valid")
    # Print graph nodes
    print(f"  Nodes: {len(model_onnx.graph.node)}")
    for node in model_onnx.graph.node:
        print(f"    {node.op_type}")
except ImportError:
    print("(onnx not installed — skipping validation)")

print()
print("=" * 60)
print("PART 3: RUNNING WITH ONNX RUNTIME")
print("=" * 60)
print("""
Once exported, inference with ONNX Runtime is simple:
""")

try:
    import onnxruntime as ort
    import numpy as np

    # Create inference session (auto-selects best execution provider)
    session = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # try GPU, fall back to CPU
    )

    # Check which provider was selected
    print(f"  Execution provider: {session.get_providers()}")

    # Print input/output info
    for inp in session.get_inputs():
        print(f"  Input:  name='{inp.name}', shape={inp.shape}, dtype={inp.type}")
    for out in session.get_outputs():
        print(f"  Output: name='{out.name}', shape={out.shape}, dtype={out.type}")

    # Run inference (input must be a numpy array, not a torch tensor)
    batch = np.random.randn(4, 4).astype(np.float32)   # batch_size=4
    outputs = session.run(
        output_names=["output"],
        input_feed={"input": batch}
    )
    result = outputs[0]
    print(f"\n  Input shape:  {batch.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Predicted classes: {result.argmax(axis=1)}")

    # Compare PyTorch vs ONNX Runtime results
    with torch.no_grad():
        pt_out = model(torch.from_numpy(batch)).numpy()
    max_diff = abs(pt_out - result).max()
    print(f"\n  Max difference PyTorch vs ONNX: {max_diff:.6f}  (should be ~0)")

except ImportError:
    print("""  (onnxruntime not installed)

  Install: pip install onnxruntime  (CPU)
           pip install onnxruntime-gpu  (GPU, requires CUDA)

  Usage:
    import onnxruntime as ort
    import numpy as np

    session = ort.InferenceSession("model.onnx")
    result  = session.run(["output"], {"input": np.array(...)})
""")

print()
print("=" * 60)
print("PART 4: DYNAMIC AXES (variable batch size / sequence length)")
print("=" * 60)
print("""
By default, ONNX fixes the batch size at export time.
Use dynamic_axes to allow variable sizes:

  torch.onnx.export(
      model, dummy_input, "model.onnx",
      dynamic_axes={
          "input":    {0: "batch_size"},   # dim 0 is variable
          "output":   {0: "batch_size"},
      }
  )

For NLP models with variable sequence length:
  dynamic_axes={
      "input_ids":      {0: "batch_size", 1: "sequence_length"},
      "attention_mask": {0: "batch_size", 1: "sequence_length"},
      "logits":         {0: "batch_size", 1: "sequence_length"},
  }
""")

print("=" * 60)
print("PART 5: ONNX GRAPH OPTIMIZATION")
print("=" * 60)
print("""
ONNX Runtime applies graph optimisations automatically:
  1. Constant folding: pre-compute operations with static inputs
  2. Node fusion: combine adjacent operations (e.g., LayerNorm = 5 ops → 1)
  3. Memory planning: optimise tensor memory reuse

Manual optimisation with onnxruntime:
  from onnxruntime.transformers import optimizer

  # Optimise for a specific transformer model type
  optimized = optimizer.optimize_model(
      "bert_model.onnx",
      model_type="bert",
      num_heads=12,
      hidden_size=768,
  )
  optimized.save_model_to_file("bert_optimized.onnx")

TensorRT export (maximum GPU speed):
  import torch_tensorrt
  trt_model = torch_tensorrt.compile(
      model,
      inputs=[torch_tensorrt.Input((1, 4))],
      enabled_precisions={torch.float16},   # FP16 for speed
  )
  torch.jit.save(trt_model, "model_trt.ts")

  Typical speedup: 2-4× over PyTorch, 1.5-2× over ONNX Runtime
  Requires: Nvidia GPU, TensorRT installed with CUDA
""")

print("=" * 60)
print("PART 6: C# USAGE")
print("=" * 60)
print("""
Using the exported model from C# (the reason you might export to ONNX):

  // NuGet: Microsoft.ML.OnnxRuntime
  using Microsoft.ML.OnnxRuntime;
  using Microsoft.ML.OnnxRuntime.Tensors;

  var session = new InferenceSession("tiny_classifier.onnx");

  // Create input tensor
  float[] data  = { 0.1f, -0.2f, 0.5f, 0.8f };
  var tensor    = new DenseTensor<float>(data, new[] { 1, 4 });
  var inputs    = new List<NamedOnnxValue> {
      NamedOnnxValue.CreateFromTensor("input", tensor)
  };

  // Run inference
  using var results = session.Run(inputs);
  var output = results.First().AsEnumerable<float>().ToArray();
  // output: [logit_class0, logit_class1, logit_class2]
  int predicted_class = output.ToList().IndexOf(output.Max());
  Console.WriteLine($"Predicted: class {predicted_class}");
""")

print("=" * 60)
print("QUICK REFERENCE: EXPORT CHECKLIST")
print("=" * 60)
print("""
Before exporting:
  ☐ model.eval()  (disables dropout/batchnorm train mode)
  ☐ dummy input has the EXACT shape the model expects
  ☐ dynamic_axes set for variable batch/sequence dims
  ☐ opset_version=17 (or higher — check ONNX docs for latest)

After exporting:
  ☐ onnx.checker.check_model(onnx.load("model.onnx"))
  ☐ Compare PyTorch vs ONNX outputs (max_diff < 1e-5)
  ☐ Benchmark inference time before claiming speedup

Common issues:
  Error: "Unsupported op" → upgrade opset_version
  Error: "dynamic control flow" → model has if/for depending on tensor values
    → use torch.jit.script() instead of torch.onnx.export()
  NaN outputs → check normalisation; model in .eval() mode?
""")

# Cleanup
if os.path.exists(onnx_path):
    os.remove(onnx_path)
    print(f"\n(Cleaned up {onnx_path})")

# =============================================================================
# EXPECTED OUTPUT:
# Exported to tiny_classifier.onnx
# ONNX validation: 3 nodes (Gemm, Relu, Gemm)
# If onnxruntime installed: PyTorch vs ONNX max_diff < 1e-5
# =============================================================================
