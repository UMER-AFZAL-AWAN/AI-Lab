# =============================================================================
# 21_deployment / 02 — API SERVING WITH FASTAPI
# =============================================================================
#
# LIKE YOU ARE 5:
# You've trained a model and it works on your machine.
# Now you want other people (or your own app) to use it.
# An API server is a process that:
#   1. Listens for requests on a port (e.g., http://localhost:8000)
#   2. Receives input data (JSON)
#   3. Runs your model
#   4. Sends back the result (JSON)
#
# C# ANALOGY:
# This is exactly what a .NET Web API controller does:
#   [HttpPost("/predict")]
#   public ActionResult<PredictionResult> Predict(InputData input) { ... }
# In Python, FastAPI does the same thing.
#
# =============================================================================

import torch
import torch.nn as nn
import json

print("=" * 60)
print("PART 1: FASTAPI OVERVIEW")
print("=" * 60)
print("""
FastAPI is a modern Python web framework for building APIs.
  pip install fastapi uvicorn

Features:
  - Automatic JSON serialisation/deserialisation (like ASP.NET model binding)
  - Automatic request validation (Pydantic models)
  - Auto-generated Swagger UI at /docs
  - Async support (non-blocking I/O)
  - Very fast (comparable to Node.js, Go)

Minimal server:
  from fastapi import FastAPI
  import uvicorn

  app = FastAPI()

  @app.get("/")
  def hello():
      return {"message": "hello world"}

  if __name__ == "__main__":
      uvicorn.run(app, host="0.0.0.0", port=8000)
""")

print("=" * 60)
print("PART 2: A COMPLETE MODEL SERVING API")
print("=" * 60)

server_code = '''
# server.py — run with: uvicorn server:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List

# ── Pydantic models (request/response schemas) ────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float]              # [f1, f2, f3, f4]
    return_probabilities: bool = False

class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: List[float] | None = None

# ── Model definition ──────────────────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
    def forward(self, x):
        return self.net(x)

# ── Load model ONCE at startup (not per request!) ────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = Classifier().to(device)

# In production: load saved weights
# model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()   # MUST be in eval mode for inference

print(f"Model loaded on {device}")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="ML Model API", version="1.0")

@app.get("/health")
def health():
    """Health check endpoint — load balancer pings this."""
    return {"status": "healthy", "device": device}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run inference on a single sample."""
    if len(req.features) != 4:
        raise HTTPException(status_code=400, detail="Expected exactly 4 features")

    # Convert to tensor and move to device
    x = torch.tensor(req.features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)              # (1, 3)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)   # (3,)

    pred_class  = probs.argmax().item()
    confidence  = probs.max().item()

    return PredictResponse(
        predicted_class=pred_class,
        confidence=round(confidence, 4),
        probabilities=probs.tolist() if req.return_probabilities else None,
    )

@app.post("/predict/batch")
def predict_batch(samples: List[PredictRequest]):
    """Run inference on a batch of samples (more efficient than N single calls)."""
    batch = torch.stack([
        torch.tensor(s.features, dtype=torch.float32)
        for s in samples
    ]).to(device)   # (N, 4)

    with torch.no_grad():
        logits = model(batch)
        probs  = torch.softmax(logits, dim=-1)   # (N, 3)

    return [
        {"predicted_class": probs[i].argmax().item(),
         "confidence": round(probs[i].max().item(), 4)}
        for i in range(len(samples))
    ]
'''

print(server_code)

print("=" * 60)
print("PART 3: TESTING THE API")
print("=" * 60)
print("""
Using requests (Python client):
  import requests

  # Single prediction
  r = requests.post("http://localhost:8000/predict", json={
      "features": [0.1, -0.2, 0.5, 0.8],
      "return_probabilities": True,
  })
  print(r.json())
  # → {"predicted_class": 1, "confidence": 0.72, "probabilities": [0.12, 0.72, 0.16]}

  # Batch prediction
  r = requests.post("http://localhost:8000/predict/batch", json=[
      {"features": [0.1, -0.2, 0.5, 0.8]},
      {"features": [-1.0, 0.3, -0.1, 0.4]},
  ])
  print(r.json())

Using curl:
  curl -X POST http://localhost:8000/predict \\
    -H "Content-Type: application/json" \\
    -d '{"features": [0.1, -0.2, 0.5, 0.8]}'

Swagger UI (auto-generated):
  Open http://localhost:8000/docs in browser
  → Interactive API documentation with "Try it out" buttons
""")

# Demonstrate the core logic without an actual server
print("=" * 60)
print("PART 4: THE MODEL SERVING LOGIC (demonstrated locally)")
print("=" * 60)

# Build the model locally and demonstrate the inference path
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 3))
    def forward(self, x): return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = Classifier().to(device).eval()

def predict(features: list) -> dict:
    """The core inference function called by the API endpoint."""
    if len(features) != 4:
        return {"error": "Expected exactly 4 features"}
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=-1).squeeze(0)
    return {
        "predicted_class": probs.argmax().item(),
        "confidence": round(probs.max().item(), 4),
        "probabilities": [round(p, 4) for p in probs.tolist()],
    }

# Test
samples = [
    [0.1, -0.2, 0.5, 0.8],
    [-1.0, 0.3, -0.1, 0.4],
    [2.0, 1.5, -0.5, 0.0],
]
for sample in samples:
    result = predict(sample)
    print(f"  Input: {sample}")
    print(f"  → class={result['predicted_class']}, conf={result['confidence']}, "
          f"probs={result['probabilities']}")
    print()

print("=" * 60)
print("PART 5: PRODUCTION CONSIDERATIONS")
print("=" * 60)
print("""
Concurrency:
  One model per process — load it at startup, reuse across requests.
  For high throughput: run multiple worker processes:
    uvicorn server:app --workers 4

  Or use dynamic batching: collect requests for 50ms, then batch them.

GPU utilisation:
  Single requests use <1% GPU — wasteful.
  Batch requests: collect multiple, run as one forward pass.

  Dynamic batching libraries:
    triton-inference-server (NVIDIA): full-featured, complex
    ray-serve: Python-native, simpler
    bitsandbytes: quantization for lower memory

Memory management:
  Release CUDA cache between requests (optional):
    torch.cuda.empty_cache()

  For streaming outputs (like LLM generation):
    Use async generators + Server-Sent Events (SSE)

Containerisation:
  Dockerfile for a FastAPI + PyTorch app:
    FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

  Build + run:
    docker build -t ml-api .
    docker run --gpus all -p 8000:8000 ml-api
""")

print("=" * 60)
print("QUICK REFERENCE: RUNNING YOUR OWN SERVING STACK")
print("=" * 60)
print("""
Minimal stack (this file, works for any model):
  pip install fastapi uvicorn
  uvicorn server:app --host 0.0.0.0 --port 8000

For LLMs (text generation):
  ollama serve   ← pull and run any GGUF model locally
  vllm serve model_name   ← high-throughput LLM server with paged attention

For diffusion models (images/video):
  A1111 (Stable Diffusion Web UI)  → localhost UI + API
  ComfyUI                          → node-based workflow + API

For model registries + versioning:
  MLflow: mlflow models serve -m "runs:/<run_id>/model" -p 5000
  BentoML: bentoml serve service:svc --port 3000
""")

# =============================================================================
# EXPECTED OUTPUT:
# 3 prediction results (class 0-2, confidence 0.3-0.9, probs sum to 1.0)
# No server started (just the core logic demonstrated)
# =============================================================================
