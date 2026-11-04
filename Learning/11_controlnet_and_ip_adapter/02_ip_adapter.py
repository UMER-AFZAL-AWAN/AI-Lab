# =============================================================================
# 11_controlnet_and_ip_adapter / 02 — IP-ADAPTER
# =============================================================================
#
# LIKE YOU ARE 5:
# ControlNet says "follow THIS STRUCTURE."
# IP-Adapter says "follow THIS IMAGE'S STYLE/APPEARANCE."
# You give it a reference photo: "make the new generation look like this photo."
# It's like saying: "paint me a cat in the same artistic style as this painting."
#
# C# ANALOGY:
# Like passing a "theme" object to a UI renderer.
# Without IP-Adapter: renderer.Draw(prompt) → default style.
# With IP-Adapter:    renderer.Draw(prompt, theme=reference_image)
#   → output matches the reference image's visual language.
#
# REAL USE IN LOCALVIDEOSTUDIO:
# You have a "look" you want to maintain across all generated frames.
# Give IP-Adapter one reference frame and it keeps the visual style consistent.
# → Style-consistent video generation.
#
# HOW IT WORKS:
# IP-Adapter adds a second cross-attention to each UNet attention block.
# The first cross-attention: attends to text embeddings (the prompt).
# The second cross-attention: attends to IMAGE embeddings (the reference).
# The image encoder: CLIP ViT (the vision encoder, not the text encoder).
# A tiny projection network: maps CLIP image features → K, V for the new attention.
#
# COST: adds ~22M parameters (the projection network).
# The image encoder (CLIP ViT-H: ~300M) loads separately.
#
# =============================================================================

import torch
import torch.nn as nn
print("=" * 60)
print("IP-ADAPTER vs CONTROLNET: SIDE BY SIDE")
print("=" * 60)
print("""
                   ControlNet          IP-Adapter
──────────────────────────────────────────────────────────────
Controls           Structure/pose      Appearance/style
Input type         Preprocessed map    Any RGB image
Where it hooks     UNet decoder        UNet attention (K,V)
Main use           Pose, depth, edges  Style transfer, face
Composable?        Yes (+ LoRA)        Yes (+ LoRA + ControlNet)
File size          ~1.5 GB             ~100 MB
Can be combined?   ─────────────────── Yes, all three at once ──
""")

print("=" * 60)
print("USAGE PATTERN (diffusers)")
print("=" * 60)
print("""
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# Load IP-Adapter (adds ~100 MB projection weights)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)

# Set how strong the image influence is (0.0–1.0)
pipe.set_ip_adapter_scale(0.6)

# Your reference image (the "style donor")
ref_image = Image.open("my_reference_painting.png")

# Generate
out = pipe(
    prompt        = "a cat playing piano",
    ip_adapter_image = ref_image,   # <- this is the new argument
    num_inference_steps = 20,
    guidance_scale = 7.5,
)
result = out.images[0]
# → Will look like a cat playing piano, painted in the style of my_reference_painting
""")

print("=" * 60)
print("IP-ADAPTER FOR VIDEO (AnimateDiff)")
print("=" * 60)
print("""
pipe = AnimateDiffPipeline.from_pretrained(...)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)
pipe.set_ip_adapter_scale(0.5)

# Use one reference image for style consistency across all frames
ref_frame = Image.open("reference_frame.png")

out = pipe(
    prompt           = "a knight walking, cinematic lighting",
    ip_adapter_image = ref_frame,
    num_frames       = 16,
    num_inference_steps = 25,
)
# → All 16 frames maintain visual consistency with ref_frame
""")

print("=" * 60)
print("IP-ADAPTER VARIANTS")
print("=" * 60)
print("""
ip-adapter_sd15.bin          ← SD 1.5 base; general appearance transfer
ip-adapter_sd15_light.bin    ← lighter version, less dominant appearance
ip-adapter-plus_sd15.bin     ← stronger face/detail fidelity
ip-adapter-plus-face_sd15.bin ← specialised for face ID transfer

For SDXL:
ip-adapter_sdxl.bin          ← SDXL base (use with SDXL pipeline)
ip-adapter-plus_sdxl_vit-h.bin ← stronger, uses ViT-H encoder

IP-Adapter FaceID:           ← face identity specifically (uses ArcFace features)
  h94/IP-Adapter-FaceID

For Wan / newer models: IP-Adapter support is in development;
check the model's HF page or diffusers changelog.
""")

print("=" * 60)
print("HOW IP-ADAPTER HOOKS INTO THE UNET (conceptual demo)")
print("=" * 60)

class IPAdapterCrossAttention(nn.Module):
    """
    Conceptual: IP-Adapter adds a SECOND cross-attention that queries image features.
    Real implementation in diffusers: AttnProcessor2_0 is replaced with IPAttnProcessor2_0.
    """
    def __init__(self, query_dim, context_dim, embed_dim, num_heads=4):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        # Original cross-attention (attends to text)
        self.text_attn_q = nn.Linear(query_dim, embed_dim)
        self.text_attn_k = nn.Linear(context_dim, embed_dim)
        self.text_attn_v = nn.Linear(context_dim, embed_dim)

        # IP-Adapter cross-attention (attends to image features)
        # Separate K and V projections — this is the ~100 MB IP-Adapter file
        self.ip_attn_k = nn.Linear(context_dim, embed_dim)
        self.ip_attn_v = nn.Linear(context_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, query_dim)
        self.scale = self.head_dim ** -0.5

    def _attn(self, Q, K, V):
        B, Nq, _ = Q.shape
        Q = Q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) * self.scale
        weights = scores.softmax(dim=-1)
        out = weights @ V
        return out.transpose(1, 2).contiguous().view(B, Nq, -1)

    def forward(self, x, text_context, image_context, ip_scale=0.6):
        """
        x:             (B, Nx, query_dim)   — spatial/temporal features
        text_context:  (B, 77, context_dim) — CLIP text embeddings
        image_context: (B, 257, context_dim)— CLIP image patch embeddings (ViT)
        ip_scale:       float — strength of image influence
        """
        Q  = self.text_attn_q(x)

        # Text cross-attention (always runs)
        K_text = self.text_attn_k(text_context)
        V_text = self.text_attn_v(text_context)
        text_out = self._attn(Q, K_text, V_text)

        # Image cross-attention (the IP-Adapter branch)
        K_img = self.ip_attn_k(image_context)
        V_img = self.ip_attn_v(image_context)
        img_out = self._attn(Q, K_img, V_img)

        # Blend: text output + scaled image output
        combined = text_out + ip_scale * img_out
        return self.out_proj(combined)


query_dim   = 32
context_dim = 32
embed_dim   = 32
B, Nx       = 2, 16    # batch, spatial tokens
N_text      = 10       # text tokens (77 in real CLIP)
N_img       = 20       # image patch tokens (257 in real CLIP ViT-L)

attn          = IPAdapterCrossAttention(query_dim, context_dim, embed_dim)
x             = torch.randn(B, Nx, query_dim)
text_context  = torch.randn(B, N_text, context_dim)
image_context = torch.randn(B, N_img, context_dim)

out_text_only = attn(x, text_context, image_context, ip_scale=0.0)
out_with_ip   = attn(x, text_context, image_context, ip_scale=0.6)

diff = (out_with_ip - out_text_only).abs().mean().item()
print(f"Output shape:                  {list(out_with_ip.shape)}")
print(f"Mean change from IP-Adapter:   {diff:.4f}")
print(f"(IP scale=0.6 meaningfully steers the cross-attention output)")

print()
print("=" * 60)
print("COMBINING CONTROLNET + IP-ADAPTER + LORA")
print("=" * 60)
print("""
# All three stack on top of the same base model simultaneously:

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,          # structural guidance
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# Add LoRA (style fine-tuning)
pipe.load_lora_weights("ghibli_style.safetensors")
pipe.set_adapters(["ghibli"], adapter_weights=[0.7])

# Add IP-Adapter (reference image appearance)
pipe.load_ip_adapter("h94/IP-Adapter", ...)
pipe.set_ip_adapter_scale(0.5)

# Generate
out = pipe(
    prompt                = "a warrior in a forest",
    image                 = canny_map,         # ControlNet: edge structure
    ip_adapter_image      = ref_image,         # IP-Adapter: appearance
    num_inference_steps   = 25,
)
# Result:
#   Structure follows canny_map (ControlNet)
#   Appearance resembles ref_image (IP-Adapter)
#   Style is Ghibli-influenced (LoRA)
""")

# =============================================================================
# EXPECTED OUTPUT:
# Output shape:                  [2, 16, 32]
# Mean change from IP-Adapter:   ~0.X (nonzero, confirming image cross-attn fires)
# =============================================================================
