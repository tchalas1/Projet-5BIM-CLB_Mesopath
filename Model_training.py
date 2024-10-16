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
