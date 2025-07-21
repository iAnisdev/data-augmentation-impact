---
license: mit
tags:
  - vqvae
  - image-generation
  - unsupervised-learning
  - pytorch
  - cifar10
  - generative-model
datasets:
  - cifar10
library_name: pytorch
model-index:
  - name: VQ-VAE-CIFAR10
    results: []
---

# VQ-VAE for CIFAR-10

This is a **Vector Quantized Variational Autoencoder (VQ-VAE)** trained on the CIFAR-10 dataset using PyTorch. It is part of an image augmentation pipeline for generative modeling and unsupervised learning research.

## 🧠 Model Details

- **Model Type**: VQ-VAE
- **Dataset**: CIFAR-10
- **Epochs**: 35  
- **Latent Space**: Discrete (quantized vectors)
- **Input Size**: 64×64  
- **Reconstruction Loss**: MSE-based  
- **Implementation**: Custom PyTorch, 3-layer Conv Encoder/Decoder  
- **FID Score**: **71.11**  
- **Loss Curve**: [`loss_curve.png`](./loss_curve.png)

> This model was trained to learn a compact representation of CIFAR-10 images via vector quantization and used for downstream data augmentation.

## 📁 Files

- `generator.pt`: Trained VQ-VAE model weights.
- `loss_curve.png`: Visual plot of training loss across 35 epochs.
- `fid_score.json`: Stored Fréchet Inception Distance (FID) evaluation result.
- `fid_real/` and `fid_fake/`: 1000 real and generated samples used for FID computation.

## 📦 How to Use

```python
import torch
from models.vqvae.model import VQVAE

model = VQVAE()
model.load_state_dict(torch.load("generator.pt", map_location="cpu"))
model.eval()
