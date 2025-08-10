---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

# Install Required Libraries

This cell ensures all required libraries (`matplotlib`, `torchvision`, `torch`, `transforms`) are installed before running the notebook.  
Skip if you're running in an environment where these are already available.


```python
from IPython.display import clear_output
```

```python
!pip install matplotlib torchvision torch transforms
# This will clear the *output* area of the cell on completion to keep things clean
clear_output()
```

# Add Project Source to Python Path

Add the `src/` folder to the Python path so it can be used to import augmentation and utility modules from our project codebase.

```python
import sys
sys.path.append('../src')
```

# Import Libraries

Import required modules from `torchvision` for dataset loading and `matplotlib` for visualization.

```python
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
```

# Pick One Image Per Class (CIFAR-10)

Load the CIFAR-10 dataset and select one representative image from each class.  
These samples will be used to demonstrate and visualize the effect of various augmentation techniques.

```python
dataset = datasets.CIFAR10(root='../.data', train=True, download=True, transform=transforms.ToTensor())
class_names = dataset.classes

# Pick one image per class
samples = {}  # label_id -> (PIL Image, label_name)
for img_tensor, label in dataset:
    if label not in samples:
        pil_img = transforms.ToPILImage()(img_tensor)
        samples[label] = (pil_img, class_names[label])
    if len(samples) == len(class_names):
        break

# For easy iteration, make a sorted list
picked_imgs = [samples[label][0] for label in sorted(samples.keys())]
picked_labels = [samples[label][1] for label in sorted(samples.keys())]
```

# Display Picked Images from CIFAR-10

Visualize one sample image from each class in the CIFAR-10 dataset.  
Each column represents a different class.

```python
fig, axes = plt.subplots(1, len(picked_imgs), figsize=(8, 2))

for ax, img, label in zip(axes, picked_imgs, picked_labels):
    ax.imshow(img)
    ax.set_title(label, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()
```

# Import Augmentation Techniques

In this section, first import the custom augmentation classes implemented for this demo. Each class below matches the exact transformations used in our code:

---

**1. Traditional Augmentation**

- `RandomHorizontalFlip` (always flips)
- `RandomRotation` (up to ±30 degrees)
- `ColorJitter` (brightness and contrast, up to 0.5)

These augmentations are standard in deep learning pipelines and introduce basic geometric and color variation to improve robustness.

---

**2. LSB Augmentation (Least Significant Bit Encoding)**

- Modifies the least significant bit of the red channel in every pixel using a fixed byte pattern.

This simulates steganographic manipulation, creating subtle, almost invisible changes. It is included for its novelty in the context of data augmentation.

---

**3. Mixup Augmentation (Standard Linear Blending)**

- Blends two images using a random ratio sampled from a Beta distribution (controls strength of mixing).
- Produces a new image that is a weighted sum of two input images (and, during training, also mixes their labels accordingly).

Mixup is a widely-used data augmentation technique that helps models learn smoother decision boundaries by exposing them to ambiguous, blended examples between classes.

---

**4. MiAMix Augmentation (Multi-stage Mixup Augmentation)**

- Random affine transformation (rotation, translation, etc.)
- Subtle color jitter
- Blends two augmented images (Either from different or same cwlasses) with equal weighting

This produces hybrid samples that help the model learn smoother transitions and boundaries between classes.

---

**5. Fusion Augmentation**

- Random horizontal flip and color jitter on both images
- Blends two augmented images with a 60/40 weighting

Fusion augmentation merges content from two images (can be same-class or different-class) to encourage more robust feature learning.

---

**6. VQ-VAE-based Augmentation**

- *(Mock implementation for demo purposes)*  
- Adds small, random perturbations to the pixel values to simulate the effect of encoding and reconstructing through a VQ-VAE model.

> **Note:**  
> Training and deploying a real VQ-VAE is computationally intensive and outside the scope of this initial demo.  
> **In future work**, Plan is to replace this mock with an actual pre-trained VQ-VAE model, which would generate more realistic, learned augmentations by reconstructing images through a discrete latent space.

---
### **Summary**

These techniques span from traditional hand-crafted augmentations to advanced multi-image blending and mock generative approaches.  
The **mock VQ-VAE** allows us to prototype our pipeline now and will be upgraded to a true generative augmentation as computational resources and time permit.


```python
from augmentations import (
    MiAMixAugmentation,
    MixupAugmentation,
    LSBAugmentation,
    VQVAEAugmentation,
    FusionAugmentation
)
```

# Visualize Basic Augmentations Across All Classes

This section demonstrates the effect of three fundamental augmentation techniques on the selected CIFAR-10 images.  
The grid below shows:

- **Columns:** One representative image from each CIFAR-10 class.
- **Rows:** Each augmentation method, applied to all classes.

**Augmentation methods shown:**
- **Original:** The raw, unaltered image.
- **Traditional:** Composite of random horizontal flip, random rotation (±30°), and color jitter for brightness/contrast.
- **LSB:** Least Significant Bit encoding, which subtly modifies the lowest bit of the red channel in each pixel (generally imperceptible to the human eye).

This visualization helps compare how each augmentation affects different classes and provides an intuitive, side-by-side reference for your report or discussions.


```python
augmentations_basic = [
    ("Original", lambda x: x),
    ("LSB", LSBAugmentation())
]
aug_names = [name for name, _ in augmentations_basic]

n_rows = len(augmentations_basic)
n_cols = len(picked_imgs)

fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(10, 4), gridspec_kw={'width_ratios': [0.15] + [1]*n_cols})

for row, (aug_name, aug) in enumerate(augmentations_basic):
    ax = axes[row, 0]
    ax.text(0.5, 0.5, aug_name, fontsize=6, fontweight='bold',
            va='center', ha='center', rotation=90, transform=ax.transAxes)
    ax.axis('off')
    for col, (img, label) in enumerate(zip(picked_imgs, picked_labels)):
        img_aug = aug(img)
        ax = axes[row, col + 1]
        ax.imshow(img_aug)
        ax.axis('off')
        if row == 0:
            ax.set_title(label, fontsize=11)

plt.tight_layout()
plt.show()
```

# Visualizing Mixup, MiAMix, and Fusion: Same-Class vs. Different-Class Blending

This section demonstrates the effect of three blending-based augmentation techniques when combining images from:

- **Same class:** Mixing an image with another from the same class (for demonstration, the same image is used twice).
- **Different class:** Mixing an image with another from a different class (the next class in the list).

For each chosen class, we visualize:

- The two input images (A and B)
- The resulting augmented image:
    - **Mixup:** Standard linear blending with a random mixing ratio
    - **MiAMix:** Multi-stage affine transformation, color jitter, then blending
    - **Fusion:** Random flip, color jitter, and blending with a 60/40 ratio

---

## **Why Both Cases?**

- **Same-class mixing**: Illustrates creation of more plausible, realistic augmented samples.
- **Different-class mixing**: Common in modern research to create ambiguous samples, forcing models to learn smoother, more robust decision boundaries.

---

## **How to Interpret the Plots**

Each subplot set allows direct comparison of the inputs and the output for all three augmentation techniques, making it easy to observe the qualitative differences and effects of blending both within and across class boundaries.

Repeat for every class to show the impact of each method on the full range of dataset classes.


```python
mixup = MixupAugmentation()
miamix = MiAMixAugmentation()
fusion = FusionAugmentation()

for i, (img1, label1) in enumerate(zip(picked_imgs, picked_labels)):
    j = (i + 1) % len(picked_imgs)
    img2, label2 = picked_imgs[j], picked_labels[j]
    img_same = img1  # for "same class", just use the same image

    # All augmentations
    miamix_same = miamix(img1, img_same)
    miamix_diff = miamix(img1, img2)
    mixup_same = mixup(img1, img_same)
    mixup_diff = mixup(img1, img2)
    fusion_same = fusion(img1, img_same)
    fusion_diff = fusion(img1, img2)

    # Arrange results for visualization
    imgs = [
        img1, img2,
        miamix_same, miamix_diff,
        mixup_same, mixup_diff,
        fusion_same, fusion_diff
    ]
    titles = [
        f"{label1} (A)",
        f"{label2} (B)",
        "MiAMix (same class)",
        "MiAMix (diff class)",
        "Mixup (same class)",
        "Mixup (diff class)",
        "Fusion (same class)",
        "Fusion (diff class)"
    ]

    fig, axes = plt.subplots(1, 8, figsize=(12, 3))
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.suptitle(f"Blending Methods for {label1} / {label2}: Same vs. Different Class", fontsize=13)
    plt.tight_layout()
    plt.show()
```

# Visualizing VQ-VAE-Based Augmentation

This section demonstrates the effect of **VQ-VAE-based augmentation** on our selected CIFAR-10 images.

- **Top row:** Original sample image from each class
- **Bottom row:** The same images after applying the (mock) VQ-VAE augmentation

## **What is VQ-VAE Augmentation?**

**VQ-VAE (Vector Quantized Variational AutoEncoder)** is a generative model that learns to encode and reconstruct images using a discrete latent space.  
When used for augmentation, it produces realistic image variations by reconstructing samples in a way that captures underlying data patterns—beyond simple pixel-level noise.

> **Note:**  
> For this demo, we use a *mock* VQ-VAE implementation that simulates the effect by applying small random perturbations.  
> In future work, this will be replaced with a true VQ-VAE model to generate more meaningful, data-driven augmentations.

## **How to Interpret the Plot**

- **Columns:** Each class in CIFAR-10
- **Row 1:** Original image  
- **Row 2:** VQ-VAE-augmented image

This side-by-side view makes it easy to compare the subtle changes introduced by generative augmentation and provides a baseline for future comparisons with a real VQ-VAE model.


```python
vqvae = VQVAEAugmentation()
vqvae_imgs = [vqvae(img) for img in picked_imgs]

n_rows = 2
n_cols = len(picked_imgs)
img_grid = [picked_imgs, vqvae_imgs]
row_names = ["Original", "VQ-VAE"]

fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(12, 4), gridspec_kw={'width_ratios': [0.8] + [1]*n_cols})

for row in range(n_rows):
    ax = axes[row, 0]
    ax.text(0.5, 0.5, row_names[row], fontsize=14, fontweight='bold',
            va='center', ha='center', rotation=90, transform=ax.transAxes)
    ax.axis('off')
    # Images
    for col in range(n_cols):
        ax = axes[row, col + 1]
        ax.imshow(img_grid[row][col])
        ax.axis('off')
        if row == 0:
            ax.set_title(picked_labels[col], fontsize=10)

plt.tight_layout()
plt.show()

```
