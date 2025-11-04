# Diffusion-Model-attempt

First attempt building a diffusion model from scratch in PyTorch. Trained on CelebA at **64×64** resolution.

- **Architecture:** UNet with time embeddings, skip connections, GroupNorm, bottleneck self-attention.
- **Objective:** **v-prediction** (predict v from x_t and t), with optional SNR weighting.
- **Noise schedule:** cosine β schedule (T=1000) with precomputed α, ᾱ.
- **Training:** AMP (autocast + GradScaler), AdamW, EMA optional.
- **Sampling:** DDPM ancestral steps (DDIM available), correct v→ε/x₀ conversions.
- **Status:** unconditional generation (no text/label conditioning).

> If you’re reading an older commit: earlier versions used ε-prediction; recent commits switched to v-prediction for improved stability/sharpness at low noise.
