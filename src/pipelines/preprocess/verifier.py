import os
import json
import logging
from .config import SUPPORTED_AUGMENTATIONS
from ..load.config import SUPPORTED_DATASETS

logger = logging.getLogger("AugmentationPipeline")
IMAGE_FORMAT = "png"


def count_images_in_dir(directory, ext=IMAGE_FORMAT):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(f".{ext}")])

def load_metadata(path):
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)

def verify_preprocessed_split(
    dataset_name: str,
    augmentation: str = None,
    split: str = "train",
    expected_count: int = None,
    root: str = "./.preprocess",
    image_format: str = IMAGE_FORMAT,
):
    if split == "train" and not augmentation:
        raise ValueError("Augmentation name must be provided for train split.")

    # Build correct path
    if split == "train":
        split_dir = os.path.join(root, dataset_name, "train", augmentation)
    else:
        split_dir = os.path.join(root, dataset_name, "test")

    if not os.path.exists(split_dir):
        logger.warning(f"⛔ Preprocessed split not found: {split_dir}")
        return False

    img_count = count_images_in_dir(split_dir, ext=image_format)
    metadata = load_metadata(split_dir)

    if expected_count is not None and img_count != expected_count:
        logger.warning(f"⚠️ Image count mismatch in {split_dir}: expected {expected_count}, found {img_count}")
        return False

    if metadata:
        logger.info(f"✅ Metadata loaded: {split_dir}/meta.json")
        logger.debug(f"  └── {metadata}")
    else:
        logger.warning(f"ℹ️ Metadata not found in {split_dir}, relying on file count only")

    logger.info(f"✅ Verified {dataset_name}-{split}-{augmentation or 'none'} ({img_count} images)")
    return True


def verify_all_preprocessed(
    dataset: str,
    augmentation: str,
    root: str = "./processed",
    image_format: str = IMAGE_FORMAT,
):
    datasets_to_check = SUPPORTED_DATASETS if dataset == "all" else [dataset]
    augmentations_to_check = SUPPORTED_AUGMENTATIONS if augmentation == "all" else [augmentation]

    for ds in datasets_to_check:
        if ds == "cifar":
            ds = "cifar10"
        # Always verify test set
        if not verify_preprocessed_split(ds, split="test", root=root, image_format=image_format):
            raise FileNotFoundError(f"Test set missing or corrupted for dataset: {ds}")

        # Now verify all augmentations for train
        for aug in augmentations_to_check:
            if not verify_preprocessed_split(ds, aug, split="train", root=root, image_format=image_format):
                raise FileNotFoundError(f"Train set missing for dataset '{ds}' and augmentation '{aug}'")


def verify_datasets_ready_for_training(
    dataset: str,
    augmentation: str,
    root: str = "./processed",
    image_format: str = IMAGE_FORMAT,
):
    """
    Verifies that processed datasets are ready for training.
    Checks both test and train datasets with proper augmentation support.
    Called before training starts to ensure all required data is available.
    """
    logger.info("🔍 Verifying processed datasets are ready for training...")
    
    datasets_to_check = SUPPORTED_DATASETS if dataset == "all" else [dataset]
    augmentations_to_check = SUPPORTED_AUGMENTATIONS if augmentation == "all" else [augmentation]
    
    missing_datasets = []
    empty_datasets = []
    
    for ds in datasets_to_check:
        if ds == "cifar":
            ds = "cifar10"
            
        logger.info(f"📊 Checking dataset: {ds}")
        
        # Check if dataset directory exists
        dataset_path = os.path.join(root, ds)
        if not os.path.exists(dataset_path):
            missing_datasets.append(ds)
            logger.error(f"❌ Dataset directory missing: {dataset_path}")
            continue
            
        # Check test set (no augmentation)
        test_path = os.path.join(dataset_path, "test")
        if not os.path.exists(test_path):
            missing_datasets.append(f"{ds}/test")
            logger.error(f"❌ Test directory missing: {test_path}")
        else:
            test_count = count_images_in_dir(test_path, image_format)
            if test_count == 0:
                empty_datasets.append(f"{ds}/test")
                logger.error(f"❌ Test set is empty: {test_path}")
            else:
                logger.info(f"✅ Test set verified: {test_count} images in {ds}/test")
        
        # Check train sets with augmentations
        for aug in augmentations_to_check:
            if aug == "traditional":
                # Traditional augmentation uses base train directory
                train_path = os.path.join(dataset_path, "train")
            else:
                # Other augmentations have their own subdirectories
                train_path = os.path.join(dataset_path, "train", aug)
                
            if not os.path.exists(train_path):
                missing_datasets.append(f"{ds}/train/{aug}")
                logger.error(f"❌ Train directory missing: {train_path}")
            else:
                train_count = count_images_in_dir(train_path, image_format)
                if train_count == 0:
                    empty_datasets.append(f"{ds}/train/{aug}")
                    logger.error(f"❌ Train set is empty: {train_path}")
                else:
                    logger.info(f"✅ Train set verified: {train_count} images in {ds}/train/{aug}")
    
    # Report any issues
    if missing_datasets:
        raise FileNotFoundError(
            f"Missing processed datasets: {missing_datasets}. "
            f"Please run preprocessing with --preprocess flag first."
        )
        
    if empty_datasets:
        raise ValueError(
            f"Empty processed datasets: {empty_datasets}. "
            f"Please re-run preprocessing to generate data."
        )
    
    logger.info("🎉 All datasets verified and ready for training!")
