import torch
import tiatoolbox
from tiatoolbox.models import PatchPredictor
import os
import random
import shutil
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

print(torch.cuda.is_available())

image_dir = 'Tiles'
train_dir = 'Tiles/train/'
val_dir = 'Tiles/val/'
test_dir = 'Tiles/test/'

def CreateDataset():
    # Paths
    image_dir = 'Tiles'
    train_dir = 'Tiles/train/'
    val_dir = 'Tiles/val/'
    test_dir = 'Tiles/test/'

    # Create directories
    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(folder, exist_ok=True)

    # Get all image paths
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.jpeg')]

    # Shuffle and split
    random.shuffle(image_paths)
    train_split = int(0.8 * len(image_paths))
    val_split = int(0.9 * len(image_paths))

    train_images = image_paths[:train_split]
    val_images = image_paths[train_split:val_split]
    test_images = image_paths[val_split:]

    # Move images to corresponding folders
    for img in train_images:
        shutil.copy(img, train_dir)

    for img in val_images:
        shutil.copy(img, val_dir)

    for img in test_images:
        shutil.copy(img, test_dir)

    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}, Test images: {len(test_images)}")

#CreateDataset()

# Get the PyTorch model (ResNet50) from the TIAToolbox predictor
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Modify the final layer to have one output for binary classification (presence or absence of BAP1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

# Transfer the model to the GPU (if available)
model = model.to(device)

#define loss criterion and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# Define the image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# Load the datasets
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['val'])

# Create the DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
