# AI Learning Folder
## Your personal revision guide — open any file, run it, read it.

All files are self-contained Python scripts. Run with:
```
cd C:\AI-Lab
python -m Learning.00_the_map
python -m Learning.01_pytorch_foundation.01_tensors
# etc.
```
Or open in VS Code and press F5 (uses your .venv Python).

---

## Map

```
00_the_map.py              ← Start here. The whole landscape on one page.

01_pytorch_foundation/
  01_tensors.py            ← What a tensor is, shapes, dtypes
  02_operations.py         ← Math on tensors: +, @, broadcasting, activations
  03_gpu_cuda.py           ← Moving tensors to GPU, FP16, speed comparison
  04_nn_module.py          ← nn.Module: every neural network's base class
  05_weights_and_forward.py ← State dicts, loading checkpoints, eval mode

02_cnn/
  01_convolution.py        ← How a filter slides over an image
  02_feature_maps.py       ← What the CNN learns to see at each layer
  03_pooling_strides_padding.py ← Shrinking spatial size, keeping it the same
  04_residual_connections.py    ← Skip connections: why deep networks work
  05_rife_connection.py    ← How RIFE (the model you've run) uses all this

03_autoencoder_vae/
  01_autoencoder.py        ← Compress → reconstruct
  02_latent_space.py       ← The compressed world: interpolation, similarity
  03_vae.py                ← Reparameterisation trick, KL divergence, generation
  04_vqvae.py              ← Discrete codes (tokens) for images
  05_in_animatediff.py     ← How the VAE fits into AnimateDiff generation

04_diffusion/
  01_adding_noise.py       ← The forward process: clean → noisy
  02_removing_noise.py     ← The reverse process: denoising step by step
  03_scheduler.py          ← DDIMScheduler: how the noise steps are managed
  04_guidance.py           ← Classifier-free guidance, negative prompts
  05_latent_diffusion.py   ← Why work in latent space (48x compression)
  06_animatediff_video.py  ← 5D tensors, motion module, ping-pong loop
  07_dit_and_flow_matching.py ← DiT (Wan/H3 denoiser) + Flow Matching vs DDPM

05_transformer/
  01_embeddings.py         ← Tokens → vectors, CLIP tokenisation
  02_attention.py          ← Q, K, V: the core attention mechanism
  03_multi_head_attention.py ← H heads in parallel, cross-attention
  04_positional_encoding.py  ← Sinusoidal PE, learned PE, RoPE
  05_encoder_decoder.py    ← Transformer blocks, BERT vs GPT
  06_vision_transformer.py ← Patches as tokens, CLS token, CLIP ViT
  07_cross_attention.py    ← How text prompts steer image generation
  08_kv_cache.py           ← KV cache: why LLMs are fast after prefill
  09_moe.py                ← Mixture of Experts: 57B total, 14B active

06_gan/
  01_generator_discriminator.py ← The forger vs the detective
  02_adversarial_training.py    ← Training loop, mode collapse, DCGAN
  03_conditional_gan.py        ← Class-conditional, pix2pix, PatchGAN

07_other_architectures/
  01_rnn_lstm_gru.py       ← Sequential memory, LSTM gates, GRU
  02_state_space_mamba.py  ← SSM, Mamba's selectivity, complexity table
  03_graph_neural_networks.py  ← Nodes, edges, message passing, GCN/GAT

08_inference_optimization/
  01_precision_dtypes.py   ← FP32/FP16/BF16/INT8/INT4, the VRAM table
  02_quantization.py       ← INT8 quantisation, bitsandbytes, GGUF
  03_cpu_offloading.py     ← Moving models between CPU RAM and GPU VRAM
  04_memory_budget.py      ← What uses VRAM, measuring peak usage
  05_checkpoint_formats.py ← .pt, .safetensors, .onnx, .gguf

09_the_stack/
  01_architecture_vs_weights_vs_pipeline.py ← The 4 things people confuse
  02_huggingface_ecosystem.py ← transformers, diffusers, Hub, from_pretrained
  03_inspect_gpu.py        ← CUDA info, VRAM status, Tensor Core benchmark
  04_inspect_model.py      ← Counting params, state dict, forward hooks

13_model_investigation/    ← PRIMARY GOAL: running unfamiliar models
  01_reading_a_hf_repo.py  ← Anatomy of a HF repo before downloading
  02_checkpoint_vs_architecture.py ← Why wrong loader class → crash (the H3 lesson)
  03_choosing_a_checkpoint.py ← VRAM formula; can 10Eros-Max fit in 8 GB?
  04_load_and_run.py       ← from_pretrained patterns, device_map, trust_remote_code
  05_benchmark_an_inference.py ← Measure load time, VRAM, inference time

14_hf_libraries/
  01_diffusers.py          ← DiffusionPipeline, scheduler swap, memory helpers
  02_transformers.py       ← AutoModel, AutoTokenizer, pipeline(), CLIP/T5
  03_accelerate.py         ← device_map, cpu_offload, init_empty_weights

PAPER_METHOD.md            ← 15-point checklist for investigating any research paper
```

---

## Suggested revision order
1. `00_the_map.py` — get the overview
2. `01_pytorch_foundation/` — all 5 files (the foundation everything else needs)
3. `02_cnn/` — especially `05_rife_connection.py` (you've run RIFE!)
4. `03_autoencoder_vae/` — especially `05_in_animatediff.py`
5. `04_diffusion/` — especially `06_animatediff_video.py` and `07_dit_and_flow_matching.py`
6. `05_transformer/` — especially `07_cross_attention.py`, `08_kv_cache.py`, `09_moe.py`
7. `08_inference_optimization/` — especially `01_precision_dtypes.py`
8. `09_the_stack/` — the big picture
9. `06_gan/` and `07_other_architectures/` — for completeness
10. `13_model_investigation/` — **primary goal**: how to run any unfamiliar model
11. `14_hf_libraries/` — the HF toolkit in depth
12. `PAPER_METHOD.md` — before reading any new paper

---

## Quick reference: what your code uses
| File in LocalVideoStudio | AI concept | Learning file |
|---|---|---|
| RIFE / IFNet.py | CNN, optical flow, residual connections | `02_cnn/` |
| AnimateDiff pipeline | Diffusion, VAE, motion module | `03_autoencoder_vae/`, `04_diffusion/` |
| inject_scene() | No AI — ffmpeg + xfade | `00_the_map.py` |
| torch.float16 everywhere | FP16 precision | `08/01_precision_dtypes.py` |
| enable_model_cpu_offload | CPU offloading | `08/03_cpu_offloading.py` |
| prompt= parameter | CLIP text encoding, cross-attention | `05/01_embeddings.py`, `05/07_cross_attention.py` |
| Investigating a new model | HF repo anatomy, VRAM budget | `13_model_investigation/` |
| Swapping AnimateDiff → Wan | DiT, Flow Matching, pipeline swap | `04/07_dit_and_flow_matching.py`, `14_hf_libraries/01_diffusers.py` |
| Reading a paper | 15-point checklist | `PAPER_METHOD.md` |
