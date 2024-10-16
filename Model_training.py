import os
import pandas as pd
import random
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import pretty_errors

folder = 'Tiles'

# Initialize empty lists for filenames and labels
filenames = []
labels = []

    
def Putting_labels(image_folder):
    # Iterate over the files in the Tiles folder
    for file in os.listdir(image_folder):
        if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):  # Add other extensions if needed
            filenames.append(file)
            
            # You can assign labels based on the filename or any other criterion.
            # Here, I'm using an example where if the file starts with "bap1", it's labeled 1, else 0.
            # Randomly assign 0 or 1 as the label
            labels.append(random.choice(["presence of BAP1", "Absence of BAP1"]))

    # Create a DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'label': labels
    })

    # Save the DataFrame to a CSV file
    df.to_csv('labels.csv', index=False)

    return(df)

# Print out the DataFrame for verification
Image_labeled=Putting_labels(folder)



# Custom Dataset class
class ImageLabelDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = read_image(img_name)  # Read image as tensor
        label = self.dataframe.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Load the dataset
df = pd.read_csv('labels.csv')

def training(image_folder, df):
    # Define parameters
    img_height, img_width = 512, 512  # Use your original image size

    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Normalize((0.5,), (0.5,))  # Adjust based on your data
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.Normalize((0.5,), (0.5,))  # Adjust based on your data
    ])

    # Create the dataset
    dataset = ImageLabelDataset(dataframe=df, img_dir=image_folder, transform=train_transform)

    # Split the dataset into training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Update the transforms for validation and test datasets
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the pre-trained ResNet50 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Change output layer for binary classification
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.float().to('cuda' if torch.cuda.is_available() else 'cpu'), labels.float().to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Validation step
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.float().to('cuda' if torch.cuda.is_available() else 'cpu'), labels.float().to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

    # Evaluation on the test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.float().to('cuda' if torch.cuda.is_available() else 'cpu'), labels.float().to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')

    torch.save(model.state_dict(), 'resnet50_model.pth')


training(folder, df)