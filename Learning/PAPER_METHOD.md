# PAPER_METHOD — How to Investigate a Research Paper

> Use this checklist every time you encounter a new model paper (Wan, MiniMax-H3, AnimateDiff, etc.).
> Work through it in order. Skip nothing on a first read.

---

## 1. What problem does it solve?
- Read the **abstract** and **introduction** (first two paragraphs only).
- Write one sentence: *"This paper solves __ by __."*
- Ask: does this problem matter for my use case (video generation on 8 GB VRAM)?

## 2. What is the core claim?
- Find the **contributions** list (usually a bullet list in the introduction).
- Identify the ONE key insight that makes this model different.
- Example: "Uses a Transformer (DiT) instead of a UNet for the denoiser."

## 3. What architecture does it use?
- Find the **method/architecture** section (Section 3 or 4 typically).
- Draw a simple box diagram: Input → Block A → Block B → Output.
- Note: UNet vs DiT? CNN vs Transformer? Encoder-decoder vs decoder-only?
- Compare to what you already know (AnimateDiff uses UNet + motion module).

## 4. What is the training objective?
- Find the **loss function(s)** in the method section.
- Is it DDPM noise prediction? Flow Matching? Adversarial? Reconstruction?
- Note the scheduler type — this tells you what `pipe.scheduler` will be.

## 5. What dataset was it trained on?
- Find the **training data** section or **experiments** section.
- Note: size, sources (internet video, synthetic, curated), content types.
- Biases in training data = biases in output.

## 6. What is the model size?
- Find: number of parameters, number of layers, hidden dimension.
- Look for a **model variants table** (small/base/large versions).
- Apply the VRAM formula: `params_B × 2 GB (FP16)` for weight memory.
  - See `08_inference_optimization/01_precision_dtypes.py`
  - See `13_model_investigation/03_choosing_a_checkpoint.py`

## 7. How is it evaluated? (metrics)
- Find the **evaluation metrics** in the experiments section.
- Common video metrics:
  - **FVD** (Fréchet Video Distance) — lower is better; measures video quality vs a reference
  - **FID** (Fréchet Inception Distance) — per-frame image quality
  - **CLIP score** — text-to-image/video alignment (0–1, higher is better)
  - **PSNR/SSIM** — pixel-level reconstruction quality
  - **User study score** — human preference percentage
- Ask: are the metrics meaningful for your task (aesthetics vs alignment vs motion)?

## 8. What do the results actually show?
- Find **Table 1** (the main results table) and **Figure 3-5** (qualitative examples).
- Does it beat the baseline? By how much?
- Look for a **failure cases** section or supplementary video — what does it get wrong?

## 9. What are the limitations?
- Find the **limitations** section (sometimes buried in the conclusion).
- Common limitations for video models:
  - Maximum clip length (e.g. 5 seconds at 24fps)
  - Resolution cap (e.g. 720p max without tiling)
  - Motion quality vs texture quality tradeoff
  - Content it fails on (hands, text, physics)

## 10. Can I reproduce this?
- Look for a **GitHub link** in the abstract or conclusion.
- Check: is the full model released, or just a demo API?
- Check: is the training code released, or inference-only?
- Read the LICENSE file before using in any project.

## 11. What checkpoints are available?
- Go to **huggingface.co** → search the paper title or org name.
- Apply the full investigation checklist from `13_model_investigation/01_reading_a_hf_repo.py`.
- Look for community-quantised versions (search: `model_name gguf` or `model_name awq`).

## 12. Does it use custom code?
- Check `model_index.json` for `"."` as the library source → `trust_remote_code=True` needed.
- Read the custom `.py` files before trusting.
- See `13_model_investigation/02_checkpoint_vs_architecture.py`.

## 13. What is the runtime cost on my hardware?
- Run `13_model_investigation/05_benchmark_an_inference.py` after loading.
- Record: load time, inference time, peak VRAM, output shape.
- Compare to the VRAM budget from `13_model_investigation/03_choosing_a_checkpoint.py`.

## 14. How does the architecture connect to what I already know?
- Match each component to a learning file:
  | Component | Learning file |
  |---|---|
  | Denoiser is a UNet | `04_diffusion/01..06`, `02_cnn/` |
  | Denoiser is a DiT | `04_diffusion/07_dit_and_flow_matching.py` |
  | Flow Matching scheduler | `04_diffusion/07_dit_and_flow_matching.py` |
  | Text encoder is CLIP | `05_transformer/01_embeddings.py`, `14_hf_libraries/02_transformers.py` |
  | Text encoder is T5/UMT5 | `14_hf_libraries/02_transformers.py` |
  | Motion module | `04_diffusion/06_animatediff_video.py` |
  | VAE / Autoencoder | `03_autoencoder_vae/` |
  | MoE layers | `05_transformer/09_moe.py` |
  | KV cache (LLM) | `05_transformer/08_kv_cache.py` |

## 15. What would integrating this into LocalVideoStudio require?
- What pipeline class? (from `14_hf_libraries/01_diffusers.py`)
- What scheduler? (swap pattern in `14_hf_libraries/01_diffusers.py`)
- What VRAM tricks? (from `08_inference_optimization/03_cpu_offloading.py`)
- What input/output format? (compare to your existing `inject_scene` and worker prompts)
- What is the minimum code change to try it?

---

## Quick Reference: Paper Sections → Learning Files

| Paper section | Key concept | Learning file |
|---|---|---|
| Architecture | DiT vs UNet | `04_diffusion/07_dit_and_flow_matching.py` |
| Loss / training | DDPM vs Flow Matching | `04_diffusion/07_dit_and_flow_matching.py` |
| Model size | VRAM budget | `13_model_investigation/03_choosing_a_checkpoint.py` |
| Inference | Scheduler, steps, CFG | `04_diffusion/02..04_scheduler_guidance.py` |
| Text conditioning | CLIP/T5 encoding | `05_transformer/07_cross_attention.py` |
| Video latent | 5D tensors | `04_diffusion/06_animatediff_video.py` |
| VAE | Encode/decode | `03_autoencoder_vae/03_vae.py` |
| MoE scaling | Expert routing | `05_transformer/09_moe.py` |
| Quantisation | Fit on 8 GB | `08_inference_optimization/02_quantization.py` |
