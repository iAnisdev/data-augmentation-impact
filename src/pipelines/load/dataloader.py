import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(image_size=32, in_channels=3):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1) if in_channels == 1 else transforms.Lambda(lambda x: x),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

def get_train_loader(
    dataset_name,
    augmentation,
    image_size=32,
    batch_size=64,
    root_dir="processed"
):
    train_dir = os.path.join(root_dir, dataset_name, "train", augmentation)
    in_channels = 1 if dataset_name == "mnist" else 3
    transform = get_transforms(image_size, in_channels)
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_test_loader(
    dataset_name,
    image_size=32,
    batch_size=64,
    root_dir="processed"
):
    test_dir = os.path.join(root_dir, dataset_name, "test")
    in_channels = 1 if dataset_name == "mnist" else 3
    transform = get_transforms(image_size, in_channels)
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
