import os
import torch
from models.cnn import CNNModel, train_model  # Later add resnet etc.
from pipelines.load.dataset_loader import get_dataloaders
from pipelines.preprocess.verifier import verify_all_preprocessed

def run_training_pipeline(args):
    # Validate dataset and augmentations
    verify_all_preprocessed(dataset=args.dataset, augmentation=args.augment)

    datasets = ["mnist", "cifar10", "imagenet"] if args.dataset == "all" else [args.dataset]
    augmentations = [
        "auto", "traditional", "miamix", "mixup", "lsb", "fusion", "gan", "vqvae"
    ] if args.augment == "all" else [args.augment]

    for dataset in datasets:
        for aug in augmentations:
            print(f"🚀 Training {args.model} on {dataset} with {aug}...")

            # Select model (currently CNN)
            if args.model == "cnn":
                in_channels = 1 if dataset == "mnist" else 3
                image_size = 28 if dataset == "mnist" else 32
                num_classes = 10  # TODO: Dynamic from folder structure
                model = CNNModel(in_channels, num_classes, image_size)
            else:
                raise ValueError(f"Model '{args.model}' not supported yet.")

            # Load data
            train_loader, val_loader = get_dataloaders(
                dataset_name=dataset,
                augmentation=aug,
                image_size=image_size,
                batch_size=args.batch_size,
            )

            # Train model
            trained_model = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device="cuda" if torch.cuda.is_available() else "cpu",
                epochs=args.epochs,
                model_name=args.model,
                dataset_name=dataset,
                augmentation=aug,
            )

            # Save weights
            os.makedirs("weights", exist_ok=True)
            weight_path = f"weights/{args.model}_{dataset}_{aug}.pt"
            torch.save(trained_model.state_dict(), weight_path)
            print(f"💾 Saved weights to {weight_path}")
