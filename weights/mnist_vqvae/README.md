---
license: mit
tags:
  - vqvae
  - image-generation
  - unsupervised-learning
  - pytorch
  - mnist
  - generative-model
datasets:
  - mnist
library_name: pytorch
model-index:
  - name: VQ-VAE-MNIST
    results:
      - task:
          type: image-generation
          name: Image Generation
        dataset:
          name: MNIST
          type: image-classification
        metrics:
          - name: FID
            type: frechet-inception-distance
            value: 53.21
---

# VQ-VAE for MNIST

This is a **Vector Quantized Variational Autoencoder (VQ-VAE)** trained on the MNIST dataset using PyTorch. The model compresses and reconstructs grayscale handwritten digits and is used as part of an image augmentation and generative modeling pipeline.

## 🧠 Model Details

- **Model Type**: VQ-VAE
- **Dataset**: MNIST
- **Epochs**: 35  
- **Latent Space**: Discrete (quantized vectors)
- **Input Size**: 64×64 (resized and converted to RGB)  
- **Reconstruction Loss**: MSE-based  
- **Implementation**: Custom PyTorch with 3-layer Conv Encoder/Decoder  
- **FID Score**: **53.21**  
- **Loss Curve**: [`loss_curve.png`](./loss_curve.png)

> This model learns compressed representations of digit images using vector quantization. The reconstructions can be used for augmentation or generative downstream tasks.

## 📁 Files

- `generator.pt`: Trained VQ-VAE model weights.
- `loss_curve.png`: Visual plot of training loss across 35 epochs.
- `fid_score.json`: Stored Fréchet Inception Distance (FID) evaluation result.
- `fid_real/` and `fid_fake/`: 1000 real and generated images used for FID computation.

## 📦 How to Use

```python
import torch
from models.vqvae.model import VQVAE

model = VQVAE()
model.load_state_dict(torch.load("generator.pt", map_location="cpu"))
model.eval()
