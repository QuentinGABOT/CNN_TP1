import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np

# Training transforms

def get_train_transform(IMAGE_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return train_transform

# Validation transforms

def get_valid_transform(IMAGE_SIZE):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return valid_transform

# Test transforms

def get_test_transform(IMAGE_SIZE):
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    return test_transform

# Initial entire datasets, same for the entire and test dataset.

def get_datasets(IMAGE_SIZE, ROOT_DIR, VALID_SPLIT, random_seed = 2022):
    dataset = datasets.CIFAR10(
              root=ROOT_DIR, train=True,
              download=True, transform=get_train_transform(IMAGE_SIZE))
    dataset_test = datasets.CIFAR10(
                   root=ROOT_DIR, train=False,
                   download=True, transform=get_train_transform(IMAGE_SIZE))
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Classes: {classes}")
    dataset_size = len(dataset)
    print(f"Total number of images in the train dataset: {dataset_size}")
    valid_size = int(VALID_SPLIT*dataset_size)
    # Training and validation sets
    indices = torch.randperm(len(dataset)).tolist()
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset, indices[-valid_size:])
    print(f"Total train images: {len(dataset_train)}")
    print(f"Total valid images: {len(dataset_valid)}")
    print(f"Total test images: {len(dataset_test)}")
    return dataset_train, dataset_valid, dataset_test, classes

# Training and validation data loaders.

def get_data_loaders(
    IMAGE_SIZE, ROOT_DIR, VALID_SPLIT, BATCH_SIZE, NUM_WORKERS, training = False):
    dataset_train, dataset_valid, dataset_test, dataset_classes = get_datasets(
        IMAGE_SIZE, ROOT_DIR, VALID_SPLIT
    )
    if (training) :
        dataset_train = ConcatDataset([dataset_train, dataset_valid])
        train_loader = DataLoader(
            dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        valid_loader = None
        test_loader = DataLoader(
            dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
    else :
        train_loader = DataLoader(
            dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        valid_loader = DataLoader(
            dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        test_loader = DataLoader(
            dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
    return train_loader, valid_loader, test_loader, dataset_classes 