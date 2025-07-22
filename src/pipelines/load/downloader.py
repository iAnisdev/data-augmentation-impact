import os
import urllib.request
import zipfile
from torchvision import datasets
from tqdm import tqdm
from .config import SUPPORTED_DATASETS

import logging

logger = logging.getLogger("AugmentationPipeline")

DATASET_DIR_NAMES = {
    "cifar10": "cifar-10-batches-py",
    "mnist": "MNIST",
    "imagenet": "tiny-imagenet-200",
}

def get_data_root():
    return os.path.join(".", ".data")

def check_dataset_exists(dataset_name):
    data_root = get_data_root()
    folder = DATASET_DIR_NAMES.get(dataset_name.lower())
    if folder is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return os.path.exists(os.path.join(data_root, folder))

def download_tiny_imagenet(data_root):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    extract_path = os.path.join(data_root, "tiny-imagenet-200")
    os.makedirs(data_root, exist_ok=True)

    if os.path.exists(extract_path):
        logger.info(f"Tiny ImageNet already downloaded in {data_root}.")
        return

    logger.info("Downloading Tiny ImageNet...")
    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading ZIP"
    ) as t:

        def reporthook(block_num, block_size, total_size):
            if t.total is None and total_size is not None:
                t.total = total_size
            downloaded = block_num * block_size
            t.update(downloaded - t.n)

        urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)

    logger.info("Extracting Tiny ImageNet...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc="Extracting ZIP"):
            zip_ref.extract(member, data_root)
    logger.info("Extraction complete.")
    os.remove(zip_path)


def download_single_dataset(dataset_name):
    data_root = get_data_root()
    if check_dataset_exists(dataset_name):
        logger.info(f"{dataset_name.upper()} already downloaded in {data_root}.")
        return
    logger.info(f"Downloading {dataset_name.upper()} to {data_root}...")
    if dataset_name == "cifar10":
        datasets.CIFAR10(root=data_root, train=True, download=True)
        datasets.CIFAR10(root=data_root, train=False, download=True)
    elif dataset_name == "mnist":
        datasets.MNIST(root=data_root, train=True, download=True)
        datasets.MNIST(root=data_root, train=False, download=True)
    elif dataset_name == "imagenet":
        download_tiny_imagenet(data_root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def download_dataset(dataset_name=None):
    if dataset_name is None or dataset_name.lower() == "all":
        logger.info("Downloading all supported datasets...")
        for ds in tqdm(SUPPORTED_DATASETS, desc="Downloading datasets"):
            download_single_dataset(ds)
    else:
        download_single_dataset(dataset_name.lower())
