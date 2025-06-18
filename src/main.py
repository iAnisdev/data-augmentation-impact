import argparse
import logging
import sys

def setup_logger():
    """Configure logger to output to console and file."""
    logger = logging.getLogger("AugmentationPipeline")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler('run.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def main():
    logger = setup_logger()

    parser = argparse.ArgumentParser(
        description="Image Augmentation and Classification CLI"
    )

    # ACTION FLAGS
    parser.add_argument('--load-data', '--ld', action='store_true', help='Download datasets (CIFAR-10, MNIST, etc.)')
    parser.add_argument('--preprocess', '--pp', action='store_true', help='Preprocess and clean data')
    parser.add_argument('--train', '--tr', action='store_true', help='Train models')
    parser.add_argument('--evaluate', '--ev', action='store_true', help='Evaluate models and output metrics/plots')
    parser.add_argument('--all', '--a', action='store_true', help='Run the entire pipeline')
    parser.add_argument('--augment', '--aug', type=str, default='none',
                        choices=['none', 'traditional', 'advanced', 'fusion', 'gan'],
                        help='Specify augmentation strategy')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'resnet', 'efficientnet'], help='Model to train')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'cifar10', 'mnist', 'imagenet'], help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--config', '-c', type=str, help='Path to config file (YAML/JSON)')

    args = parser.parse_args()

    logger.info(f"Starting CLI with arguments: {vars(args)}")

    if args.load_data:
        logger.info("Action: Download datasets selected.")
    if args.preprocess:
        logger.info("Action: Preprocess and clean data selected.")
    if args.train:
        logger.info(f"Action: Train model selected. Model: {args.model}, Dataset: {args.dataset}, Augmentation: {args.augment}")
    if args.evaluate:
        logger.info("Action: Evaluate models selected.")
    if args.all:
        logger.info(f"Action: Run full pipeline. Model: {args.model}, Dataset: {args.dataset}, Augmentation: {args.augment}")

    if args.augment:
        logger.info(f"Augmentation method: {args.augment}")

    if args.config:
        logger.info(f"Using configuration file: {args.config}")

    logger.info("Pipeline setup complete. (No computation performed yet)")

if __name__ == "__main__":
    main()
