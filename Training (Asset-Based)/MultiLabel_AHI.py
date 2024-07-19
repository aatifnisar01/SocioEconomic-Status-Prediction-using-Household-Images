# -*- coding: utf-8 -*-
"""
Created on Sat May 27 15:59:27 2023

@author: Aatif
"""

import os
import pandas as pd
import numpy as np
import shutil
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
warnings.filterwarnings("ignore")

IMAGE_SIZE = 384
BATCH_SIZE = 8
num_epochs = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


list_images = os.listdir('C:/AHI Data/Interior_Split8')
filename = "C:/Users/Aatif/OneDrive/Desktop/filtered_data.csv"
df = pd.read_csv(filename)

df.columns

Interior_Assets = [ 
                   'Unique HHD identifier',
                   'House ownership',
                   'M14A12-Black and White television _ At present',
                   'M14A13-Colour television _ At present',
                   'M14A16-Refrigerator? At present',
                   'M14A17-Air conditioner / Cooler _ At present',
                   'M14A20-Computer? At present',
                   ]

df_Assets = df[Interior_Assets]

# House ownership
mapping = {
    'Own house': 1,
}
df_Assets['House ownership'] = df_Assets['House ownership'].replace(mapping)


# Create a variable Img which contains the image name
df_Assets['img'] = df_Assets['Unique HHD identifier'].astype(str) + '_HouseInterior_6.jpg'

# Delete 'Unique HHD identifier'
df_Assets = df_Assets.drop('Unique HHD identifier', axis=1)

df_Assets.columns

# Set 'img' column as the first column
df_Assets = df_Assets[['img'] + [col for col in df_Assets.columns if col != 'img']]

columns = [
'House ownership',
'M14A12-Black and White television _ At present',
'M14A13-Colour television _ At present',
'M14A16-Refrigerator? At present',
'M14A17-Air conditioner / Cooler _ At present',
'M14A20-Computer? At present'
]

class InteriorDataset(Dataset):
    def __init__(self, dataframe, directory, x_col, y_cols, transform=None):
        super(InteriorDataset, self).__init__()
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_cols = [dataframe.columns.get_loc(col) for col in y_cols]  # Convert column names to numeric indexers
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_name = self.dataframe.iloc[index][self.x_col]
        img_path = os.path.join(self.directory, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        labels = self.dataframe.iloc[index, self.y_cols].values.astype(np.float32)
        labels = torch.tensor(labels).float()
        return image, labels








transform = transforms.Compose([
transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = InteriorDataset(df_Assets[:5000], 'C:/AHI Data/Interior_Split8', 'img', columns, transform)
valid_dataset = InteriorDataset(df_Assets[5000:], 'C:/AHI Data/Interior_Split8', 'img', columns, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ******************************************* MODEL ************************************

import timm

model = timm.create_model('swin_large_patch4_window12_384_in22k', in_chans = 3, pretrained = True,)
#model = timm.create_model('swin_large_patch4_window12_384_in22k', in_chans = 3, pretrained = True,)

# Model summary before freezing
from torchinfo import summary
#summary(model, input_size=(BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE))


n_inputs = model.head.in_features
model.head = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, len(columns)),
)
model = model.to(DEVICE)


print(len(list(model.parameters())))

for name, param in model.named_parameters():
    print(name)
    
    
for name, param in list(model.named_parameters())[:323]:    
    print('I will be frozen: {}'.format(name)) 
    param.requires_grad = False
    
#summary(model, input_size=(BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE))

print(model)


optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Define the loss function
criterion = nn.BCEWithLogitsLoss()


def calculate_metrics(predictions, targets, threshold=0.5):
    predicted_labels = (predictions > threshold).int()
    true_positives = (predicted_labels * targets).sum().item()
    false_positives = (predicted_labels * (1 - targets)).sum().item()
    false_negatives = ((1 - predicted_labels) * targets).sum().item()
    
    accuracy = (predicted_labels == targets).float().mean().item()
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1_score




# Initialize variables to store metrics
train_accuracy_list = []
train_precision_list = []
train_recall_list = []
train_f1_score_list = []
valid_accuracy_list = []
valid_precision_list = []
valid_recall_list = []
valid_f1_score_list = []

# Training loop
for epoch in range(num_epochs):
    # Set the model to training mode
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    
    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = torch.tensor(labels).to(DEVICE).float()  # Convert labels to tensor and float
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        outputs = torch.sigmoid(outputs)  # Apply sigmoid activation to model outputs
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update training loss
        train_loss += loss.item() * images.size(0)
        
        # Calculate training metrics
        accuracy, precision, recall, f1_score = calculate_metrics(outputs, labels)
        train_accuracy += accuracy * images.size(0)
        
        # Store metrics
        train_accuracy_list.append(accuracy)
        train_precision_list.append(precision)
        train_recall_list.append(recall)
        train_f1_score_list.append(f1_score)
    
    # Calculate average training loss and accuracy
    train_loss = train_loss / len(train_dataset)
    train_accuracy = train_accuracy / len(train_dataset)
    
    # Set the model to evaluation mode
    model.eval()
    valid_loss = 0.0
    valid_accuracy = 0.0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE).float()  # Convert labels to tensor and float
            
            # Forward pass
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid activation to model outputs
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Update validation loss
            valid_loss += loss.item() * images.size(0)
            
            # Calculate validation metrics
            accuracy, precision, recall, f1_score = calculate_metrics(outputs, labels)
            valid_accuracy += accuracy * images.size(0)
            
            # Store metrics
            valid_accuracy_list.append(accuracy)
            valid_precision_list.append(precision)
            valid_recall_list.append(recall)
            valid_f1_score_list.append(f1_score)
    
    # Calculate average validation loss and accuracy
    valid_loss = valid_loss / len(valid_dataset)
    valid_accuracy = valid_accuracy / len(valid_dataset)
    
    # Print training and validation metrics for each epoch
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_accuracy:.4f}")
    print("----------------------------")
