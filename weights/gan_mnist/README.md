---
library_name: pytorch
tags:
  - gan
  - mnist
  - image-generation
  - computer-vision
  - pytorch
license: mit
datasets:
  - mnist
model-index:
  - name: DCGAN-MNIST
    results: []
---

# GAN-MNIST

A simple yet effective **DCGAN** trained on the **MNIST** dataset using PyTorch, designed for **data augmentation** experiments.

## 🧠 Model Details

- **Architecture:** Deep Convolutional GAN (DCGAN)
- **Generator/Discriminator:** 3-layer CNN
- **Latent Dimension:** 100
- **Epochs Trained:** 20
- **Final FID Score:** 24.16
- **Image Size:** 28×28 grayscale

## 📦 Usage

```python
from torch import load
from models.gan.model import Generator

# Load model
model = Generator(latent_dim=100, img_channels=1)
model.load_state_dict(load("generator.pt"))
model.eval()
