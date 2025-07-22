import logging
from .downloader import SUPPORTED_DATASETS, check_dataset_exists, get_data_root

logger = logging.getLogger("AugmentationPipeline")

def verify_dataset_exists(dataset_name: str):
    dataset_name = dataset_name.lower()
    missing_datasets = []

    datasets_to_check = (
        SUPPORTED_DATASETS if dataset_name == "all" else [dataset_name]
    )

    for ds in datasets_to_check:
        if ds not in SUPPORTED_DATASETS:
            logger.error(f"Dataset '{ds}' is not supported. Supported: {SUPPORTED_DATASETS}")
            raise ValueError(f"Unsupported dataset: {ds}")

        if not check_dataset_exists(ds):
            logger.warning(f"❌ Dataset '{ds}' not found in {get_data_root()}.")
            missing_datasets.append(ds)
        else:
            logger.info(f"✅ Dataset '{ds}' is verified and available.")

    if missing_datasets:
        raise FileNotFoundError(
            f"The following dataset(s) are missing: {missing_datasets}. "
            f"Please download them using `download_dataset()`."
        )
