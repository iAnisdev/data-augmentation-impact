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
    root: str = "./.preprocess",
    image_format: str = IMAGE_FORMAT,
):
    datasets_to_check = SUPPORTED_DATASETS if dataset == "all" else [dataset]
    augmentations_to_check = SUPPORTED_AUGMENTATIONS if augmentation == "all" else [augmentation]

    for ds in datasets_to_check:
        # Always verify test set
        if not verify_preprocessed_split(ds, split="test", root=root, image_format=image_format):
            raise FileNotFoundError(f"Test set missing or corrupted for dataset: {ds}")

        # Now verify all augmentations for train
        for aug in augmentations_to_check:
            if not verify_preprocessed_split(ds, aug, split="train", root=root, image_format=image_format):
                raise FileNotFoundError(f"Train set missing for dataset '{ds}' and augmentation '{aug}'")
