import os
import json
import logging
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
from augmentations import (
    TraditionalAugmentation,
    MiAMixAugmentation,
    MixupAugmentation,
    LSBAugmentation,
    FusionAugmentation,
)

preprocess_logger = logging.getLogger("AugmentationPipeline")

PAIRED_AUGS = ["miamix", "mixup", "fusion"]
IMAGE_FORMAT = "png"

AUG_IMAGE_MODE = {
    "traditional": 1,
    "lsb": 1,
    "miamix": 2,
    "mixup": 2,
    "fusion": 2,
}

def get_augmentation(dataset_name, aug_type: str, device="cpu"):
    AUGS = {
        "traditional": TraditionalAugmentation(),
        "miamix": MiAMixAugmentation(),
        "mixup": MixupAugmentation(),
        "lsb": LSBAugmentation(),
        "fusion": FusionAugmentation()
    }
    if aug_type in AUGS:
        return AUGS[aug_type]
    elif aug_type in [None, "", "none"]:
        return None
    else:
        raise ValueError(f"Unsupported augmentation type: {aug_type}")

def get_transform(dataset_name, augmentation=None):
    base = []
    if dataset_name == "mnist":
        base.append(transforms.Grayscale(num_output_channels=3))
    base.append(transforms.Resize((64, 64)))
    if augmentation and (
        not hasattr(augmentation, "__call__") or
        augmentation.__class__.__name__.lower() not in [a + "augmentation" for a in PAIRED_AUGS]
    ):
        base.append(transforms.Lambda(lambda img: augmentation(img)))
    base.append(transforms.ToTensor())
    base.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
    return transforms.Compose(base)

def load_base_dataset(dataset_name, transform, data_dir="./.data"):
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "mnist":
        return datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    elif dataset_name == "imagenet":
        return datasets.ImageFolder(os.path.join(data_dir, "tiny-imagenet-200", "train"), transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def is_preprocessed(out_dir, expected_count):
    if not os.path.exists(out_dir):
        return False
    num_images = len([f for f in os.listdir(out_dir) if f.endswith(f'.{IMAGE_FORMAT}')])
    return num_images == expected_count

def split_dataset(dataset, train_size=0.9, test_size=0.1):
    n = len(dataset)
    train_len = int(n * train_size)
    test_len = n - train_len
    return random_split(dataset, [train_len, test_len])

def save_metadata(out_dir, metadata):
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)

def preprocess_data(
    dataset_name,
    augmentation_type,
    batch_size=64,
    raw_data_dir="./.data",
    out_root="./processed",
    device="cpu",
    train_size=None,
    test_size=None,
):
    if train_size is None and test_size is None:
        train_size, test_size = 0.9, 0.1
    elif train_size is None:
        train_size = 1.0 - test_size
    elif test_size is None:
        test_size = 1.0 - train_size

    aug = get_augmentation(dataset_name, augmentation_type, device=device)
    required_images = AUG_IMAGE_MODE.get(augmentation_type, 1)

    if required_images == 2:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3) if dataset_name == "mnist" else transforms.Lambda(lambda x: x),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        transform = get_transform(dataset_name, augmentation=aug)

    full_dataset = load_base_dataset(dataset_name, transform, data_dir=raw_data_dir)
    total_size = len(full_dataset)
    train_len = int(total_size * train_size)
    test_len = total_size - train_len

    preprocess_logger.info(f"Dataset: {dataset_name} | Total images: {total_size}")
    preprocess_logger.info(f"Train/Test split: {train_size*100:.1f}%/{test_size*100:.1f}% => Train: {train_len}, Test: {test_len}")

    train_dataset, test_dataset = split_dataset(full_dataset, train_size=train_size, test_size=test_size)

    train_out_dir = os.path.join(out_root, dataset_name, "train", augmentation_type if augmentation_type else "none")
    os.makedirs(train_out_dir, exist_ok=True)

    if not is_preprocessed(train_out_dir, len(train_dataset)):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        img_count = 0
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Saving {dataset_name}-train-{augmentation_type}")):

            if required_images == 1:
                for i in range(images.size(0)):
                    save_path = os.path.join(train_out_dir, f"{img_count:05d}.{IMAGE_FORMAT}")
                    save_image(images[i], save_path)
                    img_count += 1

            elif required_images == 2:
                for i in range(images.size(0)):
                    j = (i + 1) % images.size(0)
                    img1 = transforms.ToPILImage()(images[i].cpu())
                    img2 = transforms.ToPILImage()(images[j].cpu())
                    img_aug = aug(img1, img2)
                    img_aug = transforms.ToTensor()(img_aug)
                    save_path = os.path.join(train_out_dir, f"{img_count:05d}.{IMAGE_FORMAT}")
                    save_image(img_aug, save_path)
                    img_count += 1

        tqdm.write(f"Saved {img_count} images to {train_out_dir}")
        save_metadata(train_out_dir, {
            "dataset": dataset_name,
            "augmentation": augmentation_type,
            "split": "train",
            "count": img_count,
            "image_format": IMAGE_FORMAT,
        })
    else:
        tqdm.write(f"Skipping existing: {train_out_dir}")

    test_out_dir = os.path.join(out_root, dataset_name, "test")
    os.makedirs(test_out_dir, exist_ok=True)

    if not is_preprocessed(test_out_dir, len(test_dataset)):
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        img_count = 0
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f"Saving {dataset_name}-test")):
            for i in range(images.size(0)):
                save_path = os.path.join(test_out_dir, f"{img_count:05d}.{IMAGE_FORMAT}")
                save_image(images[i], save_path)
                img_count += 1
        tqdm.write(f"Saved {img_count} test images to {test_out_dir}")
        save_metadata(test_out_dir, {
            "dataset": dataset_name,
            "augmentation": "none",
            "split": "test",
            "count": img_count,
            "image_format": IMAGE_FORMAT,
        })
    else:
        tqdm.write(f"Skipping existing: {test_out_dir}")

    return {
        "train_dir": train_out_dir,
        "test_dir": test_out_dir,
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
        "augmentation": augmentation_type,
        "image_format": IMAGE_FORMAT
    }

def preprocess_all(
    dataset="all",
    augmentation="all",
    batch_size=64,
    train_size=0.8,
    test_size=0.2,
    device="cpu"
):
    datasets_to_run = ["cifar10", "mnist", "imagenet"] if dataset == "all" else [dataset]
    augs_to_run = ["traditional", "miamix", "mixup", "lsb", "fusion"] if augmentation == "all" else [augmentation]

    for ds in datasets_to_run:
        for aug in augs_to_run:
            preprocess_logger.info(f"Preprocessing dataset '{ds}' with augmentation '{aug}'")
            try:
                result = preprocess_data(
                    dataset_name=ds,
                    augmentation_type=aug,
                    batch_size=batch_size,
                    raw_data_dir="./.data",
                    out_root="./processed",
                    device=device,
                    train_size=train_size,
                    test_size=test_size,
                )
                preprocess_logger.info(f"✅ Done: {result}")
            except Exception as e:
                preprocess_logger.error(f"❌ Failed for {ds}-{aug}: {str(e)}")
