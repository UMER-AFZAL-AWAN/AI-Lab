# =============================================================================
# 03 — CONDITIONAL GAN: GENERATING WHAT YOU WANT
# =============================================================================
#
# LIKE YOU ARE 5:
# Normal GAN: "make me SOMETHING" — you get a random image, could be anything.
# Conditional GAN: "make me a CAT" — you tell it what to make.
#
# HOW: inject a condition (class label, text embedding, another image)
#      into BOTH the generator AND the discriminator.
# Generator:     z + condition -> specific image
# Discriminator: image + condition -> "is this a real (image, condition) pair?"
#
# EXAMPLES:
# - Class-conditional GAN: condition = class index (0=cat, 1=dog)
# - Text-to-image GAN: condition = text embedding (predecessor to Diffusion models)
# - Image-to-image (pix2pix): condition = input image (e.g. sketch -> photo)
# - Super-resolution GAN (SRGAN/ESRGAN): condition = low-res image -> high-res
#
# MODERN RELEVANCE:
# While GANs are mostly replaced by Diffusion models for text-to-image generation,
# they are still used for:
# - Discriminator-based metrics (FID score evaluates how GAN-like the generated images are)
# - Adversarial losses in VAE training (to make textures sharper)
# - Some real-time style transfer applications
# - The patchGAN discriminator in pix2pix and ESRGAN
#
# =============================================================================

import torch
import torch.nn as nn

print("=" * 60)
print("CLASS-CONDITIONAL GAN")
print("=" * 60)
#
# Condition = one-hot or embedding of a class label.
# A label embedding maps class_id -> embed_dim vector.
# This is concatenated with the noise z (or with intermediate features).

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, embed_dim=16, img_size=28):
        super().__init__()
        self.img_size   = img_size
        self.label_emb  = nn.Embedding(num_classes, embed_dim)   # class -> vector
        self.net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),  # noise + class embedding
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_size * img_size),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        # Embed the labels and concatenate with noise
        label_vecs = self.label_emb(labels)         # (B, embed_dim)
        z_cond     = torch.cat([z, label_vecs], dim=1)  # (B, latent_dim + embed_dim)
        out = self.net(z_cond)
        return out.view(-1, 1, self.img_size, self.img_size)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=16, img_size=28):
        super().__init__()
        self.img_size  = img_size
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(img_size * img_size + embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # Embed labels and concatenate with flattened image
        label_vecs = self.label_emb(labels)              # (B, embed_dim)
        img_flat   = img.flatten(1)                       # (B, 784)
        x          = torch.cat([img_flat, label_vecs], dim=1)
        return self.net(x)

G = ConditionalGenerator()
D = ConditionalDiscriminator()

# Generate: I want 4 images of class 3 (e.g. "the number 3" for MNIST)
z      = torch.randn(4, 100)
labels = torch.tensor([3, 3, 3, 3])   # all class 3
fakes  = G(z, labels)
preds  = D(fakes, labels)
print(f"Generated images: {fakes.shape}  (class 3)")
print(f"Discriminator:    {preds.T.data.round(decimals=3)}")

# Generate different classes
classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
z10     = torch.randn(10, 100)
all_fakes = G(z10, classes)
print(f"\nOne fake per class (10 classes): {all_fakes.shape}")

print()
print("=" * 60)
print("IMAGE-TO-IMAGE: PATCHGAN DISCRIMINATOR")
print("=" * 60)
#
# In pix2pix and ESRGAN, the discriminator doesn't look at the whole image.
# It looks at overlapping PATCHES (e.g. 70x70) and decides if each patch is real.
# This is a PatchGAN discriminator.
#
# Advantages:
# - Only the local texture needs to look realistic (textures are local).
# - Works on images of any size (fully convolutional).
# - Fewer parameters than a full-image discriminator.

class PatchGANDiscriminator(nn.Module):
    """
    Takes a concatenation of [input_image, target_image] and outputs
    a 2D map of "real/fake" predictions for each patch.
    """
    def __init__(self, in_channels=6):   # 3 (input) + 3 (target/fake)
        super().__init__()
        self.net = nn.Sequential(
            # (B, 6, H, W) -> (B, 64, H/2, W/2)
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 128, H/4, W/4)
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            # -> (B, 256, H/4-1, W/4-1) [stride=1 reduces by 1 on each side]
            nn.Conv2d(128, 256, 4, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            # Output map
            nn.Conv2d(256, 1, 4, stride=1, padding=1),  # real/fake per patch
            nn.Sigmoid(),
        )

    def forward(self, input_img, target_img):
        combined = torch.cat([input_img, target_img], dim=1)
        return self.net(combined)

patch_D    = PatchGANDiscriminator(in_channels=6)
input_img  = torch.randn(2, 3, 64, 64)
real_target= torch.randn(2, 3, 64, 64)
fake_target= torch.randn(2, 3, 64, 64)
pred_real  = patch_D(input_img, real_target)
pred_fake  = patch_D(input_img, fake_target)
print(f"PatchGAN: input {input_img.shape} + target {real_target.shape}")
print(f"Output map (real): {pred_real.shape}  — each cell = one patch's real/fake score")
print(f"Output map (fake): {pred_fake.shape}")
print()
print("Mean real score:", pred_real.mean().item():.3f)
print("Mean fake score:", pred_fake.mean().item():.3f)
print("(Both random since untrained — but after training, real > 0.5, fake < 0.5)")

# =============================================================================
# EXPECTED OUTPUT:
# Generated images: torch.Size([4, 1, 28, 28])  (class 3)
# Output map: torch.Size([2, 1, N, N]) — depends on input resolution
# =============================================================================
