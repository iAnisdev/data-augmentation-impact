# Model Architecture Guide

This document describes the three model architectures available in the data augmentation impact study, following your thesis requirements.

## Model Summary Table

| Model | Description | Role in Thesis | Parameters | Complexity |
|-------|-------------|----------------|------------|------------|
| CNN | Standard deep learning model | Baseline, reference | ~300K | Low |
| ResNet-18/50 | Deep CNN with residuals | Benchmark comparison | 11M/25M | Medium/High |
| EfficientNet-B0 | Scalable, efficient CNN | Benchmark comparison | ~5M | Medium-High |

## Detailed Model Descriptions

### 1. CNN (Baseline Model)
- **Architecture**: Simple convolutional neural network
- **Layers**: 2 conv layers (32, 64 filters) + 2 FC layers
- **Features**: ReLU activation, MaxPooling, Dropout (0.3)
- **Purpose**: Establishes baseline performance for comparison
- **Training**: Adam optimizer, CrossEntropyLoss

### 2. ResNet-18/50 (Residual Networks)
- **Architecture**: Deep residual network with skip connections
- **ResNet-18**: 18 layers with BasicBlocks
- **ResNet-50**: 50 layers with Bottleneck blocks
- **Features**: Batch normalization, residual connections
- **Purpose**: Benchmark against state-of-the-art deep architectures
- **Training**: Adam optimizer with learning rate scheduling

### 3. EfficientNet-B0 (Efficient Architecture)
- **Architecture**: Mobile-friendly efficient network
- **Features**: MBConv blocks, Squeeze-and-Excitation, Swish activation
- **Innovation**: Compound scaling (width × depth × resolution)
- **Purpose**: Modern efficient architecture comparison
- **Training**: AdamW optimizer with cosine annealing

## Usage Examples

### Train Single Model
```bash
# Train CNN on CIFAR-10 with traditional augmentation
python src/main.py --train --model cnn --dataset cifar10 --augment traditional --epochs 20

# Train ResNet-18 with pretrained weights
python src/main.py --train --model resnet18 --dataset cifar10 --pretrained --epochs 10

# Train EfficientNet-B0 on MNIST
python src/main.py --train --model efficientnet --dataset mnist --augment auto --epochs 15
```

### Train All Models
```bash
# Train all models on a dataset
python src/main.py --train --model all --dataset cifar10 --augment mixup --epochs 20

# Full pipeline with all models
python src/main.py --all --model all --dataset cifar10 --augment traditional
```

### Available Options
- **Models**: `cnn`, `resnet18`, `resnet50`, `efficientnet`, `all`
- **Datasets**: `cifar10`, `mnist`, `imagenet`, `all`
- **Augmentations**: `none`, `traditional`, `auto`, `mixup`, `miamix`, `lsb`, `vqvae`, `gan`, `fusion`, `all`
- **Flags**: `--pretrained` for ResNet/EfficientNet pretrained weights

## Output Files

Training generates several output files:
- **Model weights**: `weights/{model}_{dataset}/model_{augmentation}.pt`
- **Training logs**: `.artifacts/train_log_{model}_{dataset}_{augmentation}.json`
- **Evaluation reports**: `.artifacts/eval_report_{model}_{dataset}_{augmentation}.json`
- **Combined results**: `.artifacts/combined_results_{dataset}_{augmentation}.json`

## Performance Expectations

### Typical Accuracy Ranges (CIFAR-10)
- **CNN**: 70-80%
- **ResNet-18**: 85-92%
- **ResNet-50**: 87-94%
- **EfficientNet-B0**: 85-93%

### Training Time (20 epochs, GPU)
- **CNN**: ~5 minutes
- **ResNet-18**: ~15 minutes
- **ResNet-50**: ~30 minutes
- **EfficientNet-B0**: ~20 minutes

## Implementation Features

### Common Features Across All Models
- Unified training interface
- Consistent evaluation metrics
- Support for all augmentation strategies
- GPU acceleration
- Automatic model saving
- Comprehensive logging

### Model-Specific Optimizations
- **CNN**: Basic optimization for fast baseline
- **ResNet**: Learning rate scheduling, weight decay
- **EfficientNet**: Label smoothing, gradient clipping, cosine annealing

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- Per-class performance
- Confusion matrices
- Training/validation loss curves

This comprehensive model suite allows for thorough comparison of how different data augmentation techniques affect various model architectures, providing robust evidence for your thesis conclusions.
