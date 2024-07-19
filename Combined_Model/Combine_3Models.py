#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:32:52 2023

@author: aatif
"""

from torchvision import transforms as T
import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets
import warnings
import timm
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
import random

# Set random seed for reproducibility
random.seed(42)

# Suppress all warnings
warnings.filterwarnings("ignore")

Interior = "A:/AHI Data/Demo Dataset/"
Kitchen = "A:/AHI Data/Demo Dataset/"
Front = "A:/AHI Data/Demo Dataset/"

BATCH_SIZE = 2
IMAGE_SIZE = 384
num_workers = 4
num_classes = 5
num_epochs = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ********************************* Train Loader ************************

transform = T.Compose([
    T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                       ])


# INTERIOR
interior_data = []        
interior_data.append(datasets.ImageFolder(os.path.join(Interior, "train/"), transform = transform))
interior_data = torch.utils.data.ConcatDataset(interior_data)
interior_loader = DataLoader(interior_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

# KITCHEN
kitchen_data = []        
kitchen_data.append(datasets.ImageFolder(os.path.join(Kitchen, "train/"), transform = transform))
kitchen_data = torch.utils.data.ConcatDataset(kitchen_data)
kitchen_loader = DataLoader(kitchen_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

# FRONT
front_data = []        
front_data.append(datasets.ImageFolder(os.path.join(Front, "train/"), transform = transform))
front_data = torch.utils.data.ConcatDataset(front_data)
front_loader = DataLoader(front_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)


# ********************************** MODELS ***************************

# KITCHEN
model_kitchen = timm.create_model('swin_large_patch4_window12_384_in22k', in_chans = 3, pretrained = True,)

n_inputs1 = model_kitchen.head.in_features
model_kitchen.head = torch.nn.Sequential(
    torch.nn.Linear(n_inputs1, num_classes),
)
model_kitchen = model_kitchen.to(DEVICE) 
    
for name, param in list(model_kitchen.named_parameters())[:323]:    
    print('I will be frozen: {}'.format(name)) 
    param.requires_grad = False
    

# INTERIOR
model_interior = timm.create_model('swin_large_patch4_window12_384_in22k', in_chans = 3, pretrained = True,)

n_inputs2 = model_interior.head.in_features
model_interior.head = torch.nn.Sequential(
    torch.nn.Linear(n_inputs2, num_classes),
)
model_interior = model_interior.to(DEVICE) 
    
for name, param in list(model_interior.named_parameters())[:323]:    
    print('I will be frozen: {}'.format(name)) 
    param.requires_grad = False
 
    
# FRONT
model_front = timm.create_model('swin_large_patch4_window12_384_in22k', in_chans = 3, pretrained = True,)

n_inputs3 = model_front.head.in_features
model_front.head = torch.nn.Sequential(
    torch.nn.Linear(n_inputs3, num_classes),
)
model_front = model_front.to(DEVICE) 
    
for name, param in list(model_front.named_parameters())[:323]:    
    print('I will be frozen: {}'.format(name)) 
    param.requires_grad = False
    
    
    
# ********************************** TRAIN THESE 3 MODELS *******************

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(DEVICE)
optimizer_kitchen = optim.SGD(model_kitchen.head.parameters(), lr=0.0001)
optimizer_interior = optim.SGD(model_interior.head.parameters(), lr=0.0001)
optimizer_front = optim.SGD(model_front.head.parameters(), lr=0.0001)

def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        _, predicted_labels = torch.max(outputs, 1)
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions

    return avg_loss, accuracy

# Variables to store loss and accuracy after each epoch
kitchen_loss_values = []
kitchen_accuracy_values = []
interior_loss_values = []
interior_accuracy_values = []
front_loss_values = []
front_accuracy_values = []

# Training loop for each model
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Train the Kitchen model
    avg_loss_kitchen, accuracy_kitchen = train(model_kitchen, kitchen_loader, optimizer_kitchen, criterion)
    kitchen_loss_values.append(avg_loss_kitchen)
    kitchen_accuracy_values.append(accuracy_kitchen)
    print(f"Kitchen Loss: {avg_loss_kitchen:.4f}, Accuracy: {accuracy_kitchen:.4f}")

    # Train the Interior model
    avg_loss_interior, accuracy_interior = train(model_interior, interior_loader, optimizer_interior, criterion)
    interior_loss_values.append(avg_loss_interior)
    interior_accuracy_values.append(accuracy_interior)
    print(f"Interior Loss: {avg_loss_interior:.4f}, Accuracy: {accuracy_interior:.4f}")

    # Train the Front model
    avg_loss_front, accuracy_front = train(model_front, front_loader, optimizer_front, criterion)
    front_loss_values.append(avg_loss_front)
    front_accuracy_values.append(accuracy_front)
    print(f"Front Loss: {avg_loss_front:.4f}, Accuracy: {accuracy_front:.4f}")
    
    
    
# ************************************* SAVE WEIGHTS **************************

for name, param in list(model_front.named_parameters())[:330]:    
    param.requires_grad = False
    
for name, param in list(model_interior.named_parameters())[:330]:    
    param.requires_grad = False
    
for name, param in list(model_kitchen.named_parameters())[:330]:    
    param.requires_grad = False
    
# Save the weights of the models
torch.save(model_kitchen.state_dict(), "C:/Users/Public/Downloads/model_kitchen_weights.pt")
torch.save(model_interior.state_dict(), "C:/Users/Public/Downloads/model_interior_weights.pt")
torch.save(model_front.state_dict(), "C:/Users/Public/Downloads/model_front_weights.pt")


# *********************************** Valid Loader ****************************

transform = T.Compose([
    T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                       ])


# INTERIOR
interior_data_valid = []        
interior_data_valid.append(datasets.ImageFolder(os.path.join(Interior, "valid/"), transform = transform))
interior_data_valid = torch.utils.data.ConcatDataset(interior_data_valid)
interior_loader_valid = DataLoader(interior_data_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

# KITCHEN
kitchen_data_valid = []        
kitchen_data_valid.append(datasets.ImageFolder(os.path.join(Kitchen, "valid/"), transform = transform))
kitchen_data_valid = torch.utils.data.ConcatDataset(kitchen_data_valid)
kitchen_loader_valid = DataLoader(kitchen_data_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

# FRONT
front_data_valid = []        
front_data_valid.append(datasets.ImageFolder(os.path.join(Front, "valid/"), transform = transform))
front_data_valid = torch.utils.data.ConcatDataset(front_data_valid)
front_loader_valid = DataLoader(front_data_valid, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)




# ********************************* VALIDATION ********************************


# Randomly select a class label
class_label = random.randint(0, num_classes-1)
print(class_label)

# INTERIOR
interior_dataset = datasets.ImageFolder(os.path.join(Interior, "valid/"), transform=transform)
interior_indices = [idx for idx, (_, label) in enumerate(interior_dataset) if label == class_label]
interior_image = interior_dataset[random.choice(interior_indices)][0]

# KITCHEN
kitchen_dataset = datasets.ImageFolder(os.path.join(Kitchen, "valid/"), transform=transform)
kitchen_indices = [idx for idx, (_, label) in enumerate(kitchen_dataset) if label == class_label]
kitchen_image = kitchen_dataset[random.choice(kitchen_indices)][0]

# FRONT
front_dataset = datasets.ImageFolder(os.path.join(Front, "valid/"), transform=transform)
front_indices = [idx for idx, (_, label) in enumerate(front_dataset) if label == class_label]
front_image = front_dataset[random.choice(front_indices)][0]

# Pass the images to the respective models and get logits
model_kitchen.eval()
model_interior.eval()
model_front.eval()

with torch.no_grad():
    kitchen_logits = model_kitchen(kitchen_image.unsqueeze(0).to(DEVICE))
    interior_logits = model_interior(interior_image.unsqueeze(0).to(DEVICE))
    front_logits = model_front(front_image.unsqueeze(0).to(DEVICE))

# Print the logits
print("Kitchen Logits:", kitchen_logits)
print("Interior Logits:", interior_logits)
print("Front Logits:", front_logits)

# Apply softmax to obtain probabilities
kitchen_probs = F.softmax(kitchen_logits, dim=1)
interior_probs = F.softmax(interior_logits, dim=1)
front_probs = F.softmax(front_logits, dim=1)

print(kitchen_probs)
print(interior_probs)
print(front_probs)

# Calculate the average probabilities
average_probs = (kitchen_probs + interior_probs + front_probs) / 3.0
print(average_probs)

# Get the predicted class index
_, predicted_class = torch.max(average_probs, 1)

def get_classes(Interior):
    all_data = datasets.ImageFolder(Interior)
    return all_data.classes
class_labels = get_classes(Interior + "train/")

# Convert the predicted class index to the corresponding class label
predicted_label = class_labels[predicted_class.item()]
print(predicted_label)


# *********** ACTUAL CLASSES OF 3 IMAGES

interior_image_path = interior_dataset.samples[random.choice(interior_indices)][0]
print(interior_image_path)
kitchen_image_path = kitchen_dataset.samples[random.choice(kitchen_indices)][0]
print(kitchen_image_path)
front_image_path = front_dataset.samples[random.choice(front_indices)][0]
print(front_image_path)

















# **************** Manually Load images and combine logits ********************

# Provide the file paths to the images 
# NOTE: Should be from same class
kitchen_image_path = "/hdd/Aatif/AHI Data/Cooking_Images/valid/class0/6007_CookingArea_7.jpg"
interior_image_path = "/hdd/Aatif/interior/valid/class0/8007_HouseInterior_6.jpg"
front_image_path = "/hdd/Aatif/AHI Data/Front_Images/valid/class0/6007_HouseFront_2.jpg"

# Open the images using PIL
kitchen_image = Image.open(kitchen_image_path)
interior_image = Image.open(interior_image_path)
front_image = Image.open(front_image_path)

# Apply transformations
kitchen_image = transform(kitchen_image).unsqueeze(0)
interior_image = transform(interior_image).unsqueeze(0)
front_image = transform(front_image).unsqueeze(0)

# Move the images to the specified device
kitchen_image = kitchen_image.to(DEVICE)
interior_image = interior_image.to(DEVICE)
front_image = front_image.to(DEVICE)

# Pass the images through the models
kitchen_logits = model_kitchen(kitchen_image)
interior_logits = model_interior(interior_image)
front_logits = model_front(front_image)

# Apply softmax to obtain probabilities
kitchen_probs = F.softmax(kitchen_logits, dim=1)
interior_probs = F.softmax(interior_logits, dim=1)
front_probs = F.softmax(front_logits, dim=1)

print(kitchen_probs)
print(interior_probs)
print(front_probs)

# Calculate the average logits
average_logits = (kitchen_logits + interior_logits + front_logits) / 3.0
print(average_logits)

# Calculate the average probabilities
average_probs = (kitchen_probs + interior_probs + front_probs) / 3.0
print(average_probs)


# Get the predicted class index
_, predicted_class = torch.max(average_probs, 1)

def get_classes(Interior):
    all_data = datasets.ImageFolder(Interior)
    return all_data.classes
class_labels = get_classes(Interior + "train/")

# Convert the predicted class index to the corresponding class label
predicted_label = class_labels[predicted_class.item()]
print(predicted_label)
