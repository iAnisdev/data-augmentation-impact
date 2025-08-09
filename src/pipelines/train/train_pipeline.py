"""
Comprehensive training pipeline for all models
"""
import logging
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import json
from tqdm import tqdm
from models.factory import create_model, get_trainer_and_evaluator, count_parameters, get_model_summary

logger = logging.getLogger("AugmentationPipeline")


def get_class_names(dataset_name):
    """Get class names for the dataset"""
    class_names = {
        "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
        "mnist": [str(i) for i in range(10)],  # 0-9
        "imagenet": None  # Too many classes, will use indices
    }
    return class_names.get(dataset_name)


def load_processed_data(dataset_name, augmentation, batch_size=64, data_dir="./processed"):
    """
    Load preprocessed data for training
    """
    # Path to processed data
    # Training set: ALWAYS uses a specific augmentation technique
    # Test set: always clean/baseline for fair comparison across all augmentation techniques
    
    train_dir = os.path.join(data_dir, dataset_name, "train", augmentation)
    
    # Test set is always the same clean baseline for all augmentation comparisons
    test_dir = os.path.join(data_dir, dataset_name, "test")
    
    # Simple transform since data is already preprocessed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)  # Assuming RGB normalized to [-1, 1]
    ])
    
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Processed data not found. Please run preprocessing first:\n"
            f"python src/main.py --preprocess --dataset {dataset_name} --augment {augmentation}"
        )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    return train_loader, test_loader


def train_single_model(
    model_name,
    dataset_name,
    augmentation="none",
    epochs=20,
    batch_size=64,
    pretrained=False,
    device="cpu",
    log_dir=".artifacts"
):
    """
    Train a single model with specified configuration
    """
    logger.info(f"🚀 Training {model_name} on {dataset_name} with {augmentation} augmentation")
    
    # Create model
    model = create_model(model_name, dataset_name, pretrained=pretrained)
    logger.info(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Get trainer and evaluator
    train_func, eval_func = get_trainer_and_evaluator(model_name)
    
    # Load data
    train_loader, test_loader = load_processed_data(dataset_name, augmentation, batch_size)
    
    # Train model
    trained_model = train_func(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        epochs=epochs,
        model_name=model_name,
        dataset_name=dataset_name,
        augmentation=augmentation,
        log_dir=log_dir
    )
    
    # Evaluate model
    class_names = get_class_names(dataset_name)
    eval_results = eval_func(
        model=trained_model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        model_name=model_name,
        dataset_name=dataset_name,
        augmentation=augmentation,
        log_dir=log_dir
    )
    
    # Save model weights
    model_dir = os.path.join("weights", f"{model_name}_{dataset_name}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{augmentation}.pt")
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"💾 Model weights saved to {model_path}")
    
    return trained_model, eval_results


def train_all_models(
    dataset_name,
    augmentation="none",
    epochs=20,
    batch_size=64,
    pretrained=False,
    device="cpu",
    log_dir=".artifacts"
):
    """
    Train all available models
    """
    models_to_train = ["cnn", "resnet18", "efficientnet"]
    results = {}
    
    # Progress bar for models
    model_pbar = tqdm(models_to_train, desc=f"🔄 Training Models ({dataset_name}/{augmentation})", unit="model")
    
    for model_name in model_pbar:
        model_pbar.set_description(f"🚀 Training {model_name} ({dataset_name}/{augmentation})")
        try:
            model, eval_results = train_single_model(
                model_name=model_name,
                dataset_name=dataset_name,
                augmentation=augmentation,
                epochs=epochs,
                batch_size=batch_size,
                pretrained=pretrained,
                device=device,
                log_dir=log_dir
            )
            results[model_name] = eval_results
            logger.info(f"✅ {model_name} training completed successfully")
            model_pbar.set_postfix({"status": "✅ completed"})
        except Exception as e:
            logger.error(f"❌ Failed to train {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
            model_pbar.set_postfix({"status": "❌ failed"})
    
    model_pbar.close()
    
    # Save combined results
    combined_results = {
        "dataset": dataset_name,
        "augmentation": augmentation,
        "models": results,
        "summary": get_model_summary()
    }
    
    results_path = os.path.join(log_dir, f"combined_results_{dataset_name}_{augmentation}.json")
    with open(results_path, "w") as f:
        json.dump(combined_results, f, indent=2)
    
    logger.info(f"📊 Combined results saved to {results_path}")
    return results


def train_all_combinations(
    datasets=None,
    augmentations=None,
    models=None,
    epochs=20,
    batch_size=64,
    pretrained=False,
    device="cpu",
    log_dir=".artifacts"
):
    """
    Train all combinations of datasets, augmentations, and models
    """
    if datasets is None:
        datasets = ["cifar10", "mnist"]  # Exclude ImageNet by default as it's large
    if augmentations is None:
        augmentations = ["traditional", "auto", "mixup", "miamix"]  # Core augmentations
    if models is None:
        models = ["cnn", "resnet18", "efficientnet"]  # Optimized for Colab compatibility
    
    all_results = {}
    
    for dataset in datasets:
        all_results[dataset] = {}
        for augmentation in augmentations:
            all_results[dataset][augmentation] = {}
            for model in models:
                try:
                    logger.info(f"🎯 Training {model} on {dataset} with {augmentation}")
                    _, eval_results = train_single_model(
                        model_name=model,
                        dataset_name=dataset,
                        augmentation=augmentation,
                        epochs=epochs,
                        batch_size=batch_size,
                        pretrained=pretrained,
                        device=device,
                        log_dir=log_dir
                    )
                    all_results[dataset][augmentation][model] = eval_results
                except Exception as e:
                    logger.error(f"❌ Failed {model}/{dataset}/{augmentation}: {str(e)}")
                    all_results[dataset][augmentation][model] = {"error": str(e)}
    
    # Save comprehensive results
    final_results_path = os.path.join(log_dir, "comprehensive_results.json")
    with open(final_results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"🎉 Comprehensive training completed! Results saved to {final_results_path}")
    return all_results
