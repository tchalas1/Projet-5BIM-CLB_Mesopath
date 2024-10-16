import torch
import tiatoolbox
from tiatoolbox.models import PatchPredictor
import os
import random
import shutil

# Load pre-trained ResNet50 from TIAToolbox
#predictor = PatchPredictor(pretrained_model='resnet50-kather100k', batch_size=32)


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

CreateDataset()