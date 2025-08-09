"""
Model factory for creating different model architectures
"""
import torch
from models.cnn.model import CNNModel
from models.resnet.model import ResNet18, get_pretrained_resnet18
from models.efficientnet.model import EfficientNetB0, get_pretrained_efficientnet_b0


def get_model_config(dataset_name):
    """Get model configuration based on dataset"""
    configs = {
        "mnist": {
            "num_classes": 10,
            "in_channels": 3,  # Converted from grayscale to RGB in preprocessing
            "image_size": 64,  # Resized in preprocessing
        },
        "cifar10": {
            "num_classes": 10,
            "in_channels": 3,
            "image_size": 64,  # Resized in preprocessing
        },
        "imagenet": {
            "num_classes": 200,  # Tiny ImageNet has 200 classes
            "in_channels": 3,
            "image_size": 64,  # Resized in preprocessing
        }
    }
    return configs.get(dataset_name, configs["cifar10"])


def create_model(model_name, dataset_name, pretrained=False):
    """
    Create model instance based on model name and dataset
    
    Args:
        model_name: One of ['cnn', 'resnet18', 'efficientnet']
        dataset_name: One of ['mnist', 'cifar10', 'imagenet']
        pretrained: Whether to use pretrained weights (for ResNet and EfficientNet)
    
    Returns:
        torch.nn.Module: Model instance
    """
    config = get_model_config(dataset_name)
    
    if model_name.lower() == "cnn":
        return CNNModel(
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            image_size=config["image_size"]
        )
    
    elif model_name.lower() in ["resnet", "resnet18"]:
        if pretrained:
            return get_pretrained_resnet18(
                num_classes=config["num_classes"]
            )
        else:
            return ResNet18(
                num_classes=config["num_classes"],
                in_channels=config["in_channels"]
            )
    
    elif model_name.lower() in ["efficientnet", "efficientnet_b0"]:
        if pretrained:
            return get_pretrained_efficientnet_b0(
                num_classes=config["num_classes"]
            )
        else:
            return EfficientNetB0(
                num_classes=config["num_classes"],
                in_channels=config["in_channels"]
            )
    
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from: cnn, resnet18, efficientnet")


def get_trainer_and_evaluator(model_name):
    """
    Get appropriate trainer and evaluator functions for the model
    
    Args:
        model_name: Model name
        
    Returns:
        tuple: (train_function, evaluate_function)
    """
    if model_name.lower() == "cnn":
        from models.cnn.trainer import train_model
        from models.cnn.evaluator import evaluate_model
        return train_model, evaluate_model
    
    elif model_name.lower() in ["resnet", "resnet18"]:
        from models.resnet.trainer import train_model
        from models.resnet.evaluator import evaluate_model
        return train_model, evaluate_model
    
    elif model_name.lower() in ["efficientnet", "efficientnet_b0"]:
        from models.efficientnet.trainer import train_model
        from models.efficientnet.evaluator import evaluate_model
        return train_model, evaluate_model
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary():
    """Get a summary of all available models"""
    return {
        "CNN": {
            "description": "Standard deep learning model",
            "role": "Baseline, reference",
            "parameters": "~300K (depends on input size)",
            "complexity": "Low"
        },
        "ResNet-18": {
            "description": "Residual network with 18 layers",
            "role": "Benchmark comparison",
            "parameters": "~11M",
            "complexity": "Medium"
        },
        "EfficientNet-B0": {
            "description": "Scalable, efficient CNN",
            "role": "Benchmark comparison",
            "parameters": "~5M",
            "complexity": "Medium-High"
        }
    }
