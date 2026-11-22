# Character Animation — Model Downloads

Run these commands once from `C:\AI-Lab` with the venv active.

---

## 1. IP-Adapter FaceID

Injects face identity from a reference photo into the generation pipeline.

```bash
mkdir -p Models/IP-Adapter-FaceID
cd Models/IP-Adapter-FaceID

# Main weights
curl -L "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin" -o ip-adapter-faceid_sd15.bin

# LORA weights (improves face consistency)
curl -L "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15_lora.safetensors" -o ip-adapter-faceid_sd15_lora.safetensors
```

## 2. ControlNet OpenPose

Drives body pose frame-by-frame from a skeleton image.
Only needed if you pass `pose_image=` to AnimationRequest.

```bash
huggingface-cli download lllyasviel/control_v11p_sd15_openpose \
    --local-dir Models/ControlNet-OpenPose
```

## 3. InsightFace (Python package)

Used to extract face embeddings from the reference photo.

```bash
pip install insightface onnxruntime-gpu
```

InsightFace downloads the `buffalo_l` model automatically on first use
(~300 MB, saved to `~/.insightface/models/`).

---

## Verify layout

```
Models/
  IP-Adapter-FaceID/
    ip-adapter-faceid_sd15.bin
    ip-adapter-faceid_sd15_lora.safetensors
  ControlNet-OpenPose/
    diffusion_pytorch_model.safetensors
    config.json
  AnimateDiff-MotionAdapter/   ← already present
  AnimateDiff-BaseModel/       ← already present
```
