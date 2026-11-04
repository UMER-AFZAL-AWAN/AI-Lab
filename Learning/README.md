# AI Learning Folder
## Your personal revision guide — open any file, run it, read it.

All files are self-contained Python scripts. Run with:
```
cd C:\AI-Lab
python -m Learning.01_pytorch_foundation.01_tensors
# etc.
```
Or open in VS Code and press F5 (uses your .venv Python).

---

## Map

```
00_the_map.py              ← Start here. The whole landscape on one page.

00_math_foundations/       ← START HERE if maths feels shaky
  01_linear_algebra.py     ← Vectors, matrices, dot product (the soul of attention)
  02_calculus_and_gradients.py ← Chain rule, autograd, gradient descent
  03_probability_statistics.py ← Softmax, KL divergence, Gaussian, cross-entropy

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

03_unet/
  01_encoder_decoder.py    ← Compress then expand — the backbone of diffusion
  02_skip_connections.py   ← Why shortcuts matter for preserving detail
  03_cross_attention.py    ← How text ("a cat") steers the image

04_diffusion/
  01_noise_schedule.py     ← Adding and predicting noise step by step
  02_forward_process.py    ← q(x_t | x_0): the noising maths
  03_reverse_process.py    ← p(x_{t-1} | x_t): denoising = generating
  04_ddpm_scheduler.py     ← DDPM, DDIM, DPM++: different de-noising paths
  05_cfg.py                ← Classifier-Free Guidance: strength slider explained
  06_vae.py                ← Encode to 64×64 latent; decode back to image
  07_dit_and_flow_matching.py ← DiT patches, flow matching straight-line paths

05_transformer/
  01_self_attention.py     ← Q·Kᵀ/√d · V: the fundamental operation
  02_multi_head_attention.py ← 8 heads, each looking at different aspects
  03_positional_encoding.py ← How the model knows token order
  04_feed_forward.py       ← The expand-ReLU-shrink block after attention
  05_full_transformer.py   ← Putting all blocks together
  06_clip.py               ← Contrastive image-text alignment (the text encoder)
  07_text_to_embedding.py  ← Token IDs → embeddings → context vectors
  08_kv_cache.py           ← Why LLM inference is fast after the first token
  09_moe.py                ← Mixture of Experts: total vs active parameters

06_animatediff/
  01_motion_module.py      ← Temporal self-attention across frames
  02_full_pipeline.py      ← Base SD + motion module = video
  03_inference_params.py   ← Steps, guidance scale, seeds explained

07_wan/
  01_wan_architecture.py   ← VACE, Wan2.1 internals
  02_vfr_analysis.py       ← VFR detection and normalisation

08_lora/
  01_what_is_lora.py       ← Low-rank A·B delta, rank vs parameters
  02_load_and_use.py       ← load_lora_weights, set_adapters, fuse/unfuse

09_vae_deep/
  01_encoder_decoder.py    ← Detailed VAE internals

10_lora_and_fine_tuning/
  01_what_is_lora.py       ← LoRA from scratch: LoRALinear class
  02_load_and_use_lora.py  ← load_lora_weights, multi-LoRA composition

10_the_practice/
  01_animatediff_from_scratch.py ← Full 7-step inference loop without pipe()

11_controlnet_and_ip_adapter/
  01_controlnet.py         ← Residual injection into UNet decoder
  02_ip_adapter.py         ← Dual cross-attention: text + image prompting

12_video_metrics/
  01_measuring_quality.py  ← PSNR, SSIM, CLIP score, FVD explained + code

13_model_investigation/
  01_reading_a_paper.py    ← 15-point checklist for any new paper
  02_map_the_hf_ecosystem.py ← diffusers, transformers, accelerate
  03_reading_model_configs.py ← config.json, what each field means
  04_load_and_run.py       ← 4 from_pretrained patterns, device_map
  05_benchmark_an_inference.py ← Timer, VRAM peak, 8GB rule table

15_training_fundamentals/  ← CRITICAL: understanding how training actually works
  01_loss_functions.py     ← MSE, BCE, CrossEntropy, VAE loss, GAN loss
  02_backpropagation.py    ← Forward+backward side by side, gradient accumulation trap
  03_optimizers.py         ← SGD vs Adam vs AdamW, lr schedules
  04_training_loop.py      ← Complete loop, train/eval mode, overfitting demo
  05_regularization.py     ← Dropout, weight decay, BatchNorm, LayerNorm, augmentation
  06_mixed_precision_training.py ← FP16/BF16, GradScaler, autocast
  07_dataset_and_dataloader.py ← Dataset class, DataLoader, collate_fn, splits

16_classical_ml/           ← Foundations: understand what NNs improved upon
  01_linear_logistic_regression.py ← From scratch: linear reg, sigmoid, softmax
  02_decision_trees_random_forests.py ← Gini impurity, NMS, random forest ensembles
  03_pca_and_embeddings.py ← PCA via SVD, nn.Embedding, cosine similarity

17_data_pipelines/         ← Getting data ready for training
  01_image_transforms.py   ← torchvision transforms, normalisation constants
  02_video_frame_extraction.py ← cv2, ffmpeg, sampling strategies
  03_custom_dataset.py     ← Dataset class patterns, collate_fn, VideoClipDataset

18_experiment_tracking/    ← Knowing what's happening during training
  01_tensorboard_and_logging.py ← SummaryWriter, images, histograms, wandb
  02_debugging_training.py ← NaN detection, gradient checking, diagnosis

19_computer_vision/        ← Beyond generation: understanding, detecting, segmenting
  01_object_detection.py   ← YOLO, IoU, NMS, bounding box formats
  02_segmentation_sam.py   ← U-Net, SAM, semantic vs instance segmentation
  03_classification_finetuning.py ← ResNet/ViT fine-tuning, feature extraction

20_nlp_and_llms/           ← Language models from objectives to deployment
  01_language_model_objectives.py ← MLM (BERT) vs CLM (GPT), tokenisation, temperature
  02_llm_finetuning_sft.py ← SFT, chat templates, loss masking, LoRA+QLoRA
  03_rag_and_embeddings.py ← Vector databases, retrieval, chunking strategies

21_deployment/             ← Getting your model into production
  01_onnx_export.py        ← torch.onnx.export, ONNX Runtime, C# usage
  02_api_serving.py        ← FastAPI server, batching, production tips

PAPER_METHOD.md            ← 15-point checklist for reading any research paper
```

---

## Suggested revision order (complete course)

| Step | Topic | Files |
|------|-------|-------|
| 1 | Math foundations | `00_math_foundations/` |
| 2 | PyTorch basics | `01_pytorch_foundation/` |
| 3 | Training fundamentals | `15_training_fundamentals/` |
| 4 | CNNs | `02_cnn/` |
| 5 | UNet | `03_unet/` |
| 6 | Transformers | `05_transformer/` |
| 7 | Diffusion | `04_diffusion/` |
| 8 | Classical ML | `16_classical_ml/` |
| 9 | Data pipelines | `17_data_pipelines/` |
| 10 | Experiment tracking | `18_experiment_tracking/` |
| 11 | AnimateDiff / Wan | `06_animatediff/`, `07_wan/` |
| 12 | LoRA + ControlNet | `10_lora_and_fine_tuning/`, `11_controlnet_and_ip_adapter/` |
| 13 | Computer Vision | `19_computer_vision/` |
| 14 | NLP + LLMs | `20_nlp_and_llms/` |
| 15 | Model investigation | `13_model_investigation/` |
| 16 | Deployment | `21_deployment/` |

---

## Quick-reference table

| Concept | File |
|---------|------|
| What is a tensor? | `01_pytorch_foundation/01_tensors.py` |
| Matrix multiply shape rule | `00_math_foundations/01_linear_algebra.py` |
| Chain rule / autograd | `00_math_foundations/02_calculus_and_gradients.py` |
| Cross-entropy loss | `15_training_fundamentals/01_loss_functions.py` |
| Gradient accumulation bug | `15_training_fundamentals/02_backpropagation.py` |
| AdamW vs Adam | `15_training_fundamentals/03_optimizers.py` |
| Training loop boilerplate | `15_training_fundamentals/04_training_loop.py` |
| Dropout / BatchNorm | `15_training_fundamentals/05_regularization.py` |
| FP16 / mixed precision | `15_training_fundamentals/06_mixed_precision_training.py` |
| Custom Dataset class | `17_data_pipelines/03_custom_dataset.py` |
| Self-attention Q·K·V | `05_transformer/01_self_attention.py` |
| KV cache | `05_transformer/08_kv_cache.py` |
| Classifier-Free Guidance | `04_diffusion/05_cfg.py` |
| LoRA explained | `10_lora_and_fine_tuning/01_what_is_lora.py` |
| AnimateDiff full loop | `10_the_practice/01_animatediff_from_scratch.py` |
| ControlNet | `11_controlnet_and_ip_adapter/01_controlnet.py` |
| YOLO / NMS | `19_computer_vision/01_object_detection.py` |
| BERT vs GPT | `20_nlp_and_llms/01_language_model_objectives.py` |
| SFT / LoRA fine-tuning | `20_nlp_and_llms/02_llm_finetuning_sft.py` |
| RAG pipeline | `20_nlp_and_llms/03_rag_and_embeddings.py` |
| ONNX export | `21_deployment/01_onnx_export.py` |
| Serve a model as API | `21_deployment/02_api_serving.py` |
| NaN in training | `18_experiment_tracking/02_debugging_training.py` |

---

## Total: 83 files  
Every concept taught like you are 5. Every file runnable. No model downloads needed.
