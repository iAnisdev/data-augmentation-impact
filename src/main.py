import argparse
import logging
import sys
import torch
from pipelines.load._init__ import download_dataset
from pipelines.preprocess.__init__ import (
    run_preprocessing_pipeline,
    smart_verify_datasets_ready_for_training,
)


def setup_logger():
    """Configure logger to output to console and file."""
    logger = logging.getLogger("AugmentationPipeline")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler("run.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def main():
    logger = setup_logger()
    parser = argparse.ArgumentParser(
        description="Image Augmentation and Classification CLI"
    )

    # ACTION FLAGS
    parser.add_argument(
        "--load-data",
        "--ld",
        action="store_true",
        help="Download datasets (CIFAR-10, MNIST, etc.)",
    )
    parser.add_argument(
        "--preprocess", "--pp", action="store_true", help="Preprocess and clean data"
    )
    parser.add_argument("--train", "--tr", action="store_true", help="Train models")
    parser.add_argument(
        "--evaluate",
        "--ev",
        action="store_true",
        help="Evaluate models and output metrics/plots",
    )
    parser.add_argument(
        "--all", "--a", action="store_true", help="Run the entire pipeline"
    )
    parser.add_argument(
        "--augment",
        "--aug",
        type=str,
        default="all",
        choices=[
            "all",
            "auto",
            "traditional",
            "miamix",
            "mixup",
            "lsb",
            "vqvae",
            "gan",
            "fusion",
        ],
        help="Specify augmentation strategy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "cnn", "resnet18", "efficientnet"],
        help="Model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "cifar10", "mnist", "imagenet"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--pretrained", action="store_true", 
        help="Use pretrained models (ResNet/EfficientNet only)"
    )
    parser.add_argument(
        "--config", "-c", type=str, help="Path to config file (YAML/JSON)"
    )

    args = parser.parse_args()

    logger.info(f"Starting CLI with arguments: {vars(args)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Processing using {device}")

    if args.load_data:
        logger.info("Downloading datasets...")
        download_dataset(args.dataset)

    if args.preprocess:
        logger.info("Running preprocessing pipeline...")
        run_preprocessing_pipeline(
            dataset=args.dataset,
            augmentation=args.augment,
            batch_size=args.batch_size,
            train_size=0.8,
            test_size=0.2,
            device=device,
        )

    if args.train:
        logger.info(f"Training model: {args.model} on dataset {args.dataset} with augment: {args.augment}")
        
        # Import required modules for expansion
        from pipelines.preprocess.config import SUPPORTED_AUGMENTATIONS
        from pipelines.load.config import SUPPORTED_DATASETS
        from pipelines.train.train_pipeline import train_single_model, train_all_models
        
        # Expand "all" parameters into actual lists
        datasets_to_train = SUPPORTED_DATASETS if args.dataset == "all" else [args.dataset]
        augmentations_to_train = SUPPORTED_AUGMENTATIONS if args.augment == "all" else [args.augment]
        
        # Remove "traditional" from augmentations if it's in the list, since we know it's not available
        if "traditional" in augmentations_to_train:
            logger.warning("⚠️  Skipping 'traditional' augmentation - not available in downloaded data")
            augmentations_to_train = [aug for aug in augmentations_to_train if aug != "traditional"]
        
        # Train for each dataset-augmentation combination
        for dataset in datasets_to_train:
            for augmentation in augmentations_to_train:
                logger.info(f"🚀 Training model: {args.model} on dataset {dataset} with augment: {augmentation}")
                
                # Skip if dataset data is not available
                try:
                    smart_verify_datasets_ready_for_training(dataset=dataset, augmentation=augmentation)
                except Exception as e:
                    logger.warning(f"⚠️  Skipping {dataset} with {augmentation}: {e}")
                    continue
                
                if args.model == "all":
                    train_all_models(
                        dataset_name=dataset,
                        augmentation=augmentation,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        pretrained=args.pretrained,
                        device=device
                    )
                else:
                    train_single_model(
                        model_name=args.model,
                        dataset_name=dataset,
                        augmentation=augmentation,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        pretrained=args.pretrained,
                        device=device
                    )
            
    if args.evaluate:
        logger.info("Evaluate models")
        # TODO: Implement evaluation pipeline
        
    if args.all:
        logger.info(f"Run full pipeline with model {args.model}")
        # Download data
        download_dataset(args.dataset)
        # Preprocess
        run_preprocessing_pipeline(
            dataset=args.dataset,
            augmentation=args.augment,
            batch_size=args.batch_size,
            train_size=0.8,
            test_size=0.2,
            device=device,
        )
        # Verify datasets are ready for training (with auto-download)
        smart_verify_datasets_ready_for_training(dataset=args.dataset, augmentation=args.augment)
        
        # Train
        from pipelines.train.train_pipeline import train_single_model, train_all_models
        if args.model == "all":
            train_all_models(
                dataset_name=args.dataset,
                augmentation=args.augment,
                epochs=args.epochs,
                batch_size=args.batch_size,
                pretrained=args.pretrained,
                device=device
            )
        else:
            train_single_model(
                model_name=args.model,
                dataset_name=args.dataset,
                augmentation=args.augment,
                epochs=args.epochs,
                batch_size=args.batch_size,
                pretrained=args.pretrained,
                device=device
            )

    logger.info("Pipeline execution complete.")


if __name__ == "__main__":
    main()
