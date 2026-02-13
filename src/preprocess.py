import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

DATA_DIR = "data/raw/archive/PetImages"
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Data augmentation for training
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

# No augmentation for val & test
val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def get_dataloaders():

    # Full dataset without transforms first
    full_dataset = datasets.ImageFolder(DATA_DIR)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Apply transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_test_transforms
    test_dataset.dataset.transform = val_test_transforms

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
