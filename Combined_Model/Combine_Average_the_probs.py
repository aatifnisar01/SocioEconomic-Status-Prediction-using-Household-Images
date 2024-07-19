# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:27:57 2023

@author: Aatif
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
import numpy as np
import glob

# Set random seed for reproducibility
random.seed(42)

# Suppress all warnings
warnings.filterwarnings("ignore")

Interior = "A:/AHI Data/Demo Dataset/Interior/"
Kitchen = "A:/AHI Data/Demo Dataset/Cooking/"
Front = "A:/AHI Data/Demo Dataset/Front/"

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



# ***************************************** TESTING using Manual Images ***************************

# Define the paths or data for the three images
cooking_image_path = "A:/AHI Data/Demo Dataset/Cooking/train/class0/6007_CookingArea_7.jpg"
front_image_path = "A:/AHI Data/Demo Dataset/Front/train/class0/6007_HouseFront_2.jpg"
interior_image_path = "A:/AHI Data/Demo Dataset/Interior/valid/class0/6007_HouseInterior_6.jpg"

# Open the images using PIL
kitchen_image = Image.open(cooking_image_path)
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

# Pass each image through its corresponding model to obtain the logits
front_logits = model_front(front_image)
kitchen_logits = model_kitchen(kitchen_image)
interior_logits = model_interior(interior_image)
print(front_logits, kitchen_logits, interior_logits)


# Convert logits to probabilities using sigmoid function
front_probabilities = torch.sigmoid(front_logits).cpu().numpy()
kitchen_probabilities = torch.sigmoid(kitchen_logits).cpu().numpy()
interior_probabilities = torch.sigmoid(interior_logits).cpu().numpy()
print(front_probabilities, kitchen_probabilities, interior_probabilities)

# Find the average of the probabilities
average_probabilities = np.mean([front_probabilities, kitchen_probabilities, interior_probabilities], axis=0)
print(average_probabilities)


# Get the predicted class index
predicted_class = np.argmax(average_probabilities)

def get_classes(Interior):
    all_data = datasets.ImageFolder(Interior)
    return all_data.classes
class_labels = get_classes(Interior + "train/")

# Convert the predicted class index to the corresponding class label
predicted_label = class_labels[predicted_class.item()]
print(predicted_label)


# ************************ Automatically picking 3 images with same id ***************************


failed = 0
passed = 0
actual_class = "class1"

# Define the image directory paths
cooking_dir = "A:/AHI Data/Demo Dataset/Cooking/train/class1/"
front_dir = "A:/AHI Data/Demo Dataset/Front/train/class1/"
interior_dir = "A:/AHI Data/Demo Dataset/Interior/train/class1/"

# Get Interior images' IDS (Because front has the maximum images)
front_images = glob.glob(os.path.join(front_dir, "*.jpg"))
image_ids = []

for image_path in front_images:
    image_name = os.path.basename(image_path)
    image_id = image_name.split("_")[0]
    image_ids.append(image_id)
    
# Track the correct predictions
correct_predictions = 0

for image_id in image_ids:
    interior_images = glob.glob(os.path.join(interior_dir, f"{image_id}_HouseInterior_6.jpg"))
    cooking_images = glob.glob(os.path.join(cooking_dir, f"{image_id}_CookingArea_7.jpg"))
    front_images = glob.glob(os.path.join(front_dir, f"{image_id}_HouseFront_2.jpg"))
    
    # Check if there are at least three images with the same ID
    if len(cooking_images) == 1 and len(front_images) == 1 and len(interior_images) == 1:
        # Pick the first image from each directory
        cooking_image_path = cooking_images[0]
        front_image_path = front_images[0]
        interior_image_path = interior_images[0]

        # Continue with the rest of the code...
        # Open the images using PIL
        kitchen_image = Image.open(cooking_image_path)
        interior_image = Image.open(interior_image_path)
        front_image = Image.open(front_image_path)

        # Apply transformations
        kitchen_image = transform(kitchen_image).unsqueeze(0).to(DEVICE)
        interior_image = transform(interior_image).unsqueeze(0).to(DEVICE)
        front_image = transform(front_image).unsqueeze(0).to(DEVICE)

        # Move the images to the specified device
        kitchen_image = kitchen_image.to(DEVICE)
        interior_image = interior_image.to(DEVICE)
        front_image = front_image.to(DEVICE)

        # Pass each image through its corresponding model to obtain the logits
        front_logits = model_front(front_image)
        kitchen_logits = model_kitchen(kitchen_image)
        interior_logits = model_interior(interior_image)
        #print(front_logits, kitchen_logits, interior_logits)


        # Convert logits to probabilities using sigmoid function
        front_probabilities = torch.sigmoid(front_logits).cpu().numpy()
        kitchen_probabilities = torch.sigmoid(kitchen_logits).cpu().numpy()
        interior_probabilities = torch.sigmoid(interior_logits).cpu().numpy()
        #print(front_probabilities, kitchen_probabilities, interior_probabilities)

        # Find the average of the probabilities
        average_probabilities = np.mean([front_probabilities, kitchen_probabilities, interior_probabilities], axis=0)
        #print(average_probabilities)


        # Get the predicted class index
        predicted_class = np.argmax(average_probabilities)

        def get_classes(Interior):
            all_data = datasets.ImageFolder(Interior)
            return all_data.classes
        class_labels = get_classes(Interior + "train/")

        # Convert the predicted class index to the corresponding class label
        predicted_label = class_labels[predicted_class.item()]
        #print(predicted_label)
        
        # Compare the predicted label with the actual class (class0)
        if predicted_label == actual_class:
            correct_predictions += 1
        passed+=1
         
    else:
        print("Insufficient images with the same ID for testing.")
        failed+=1

# Calculate the accuracy
accuracy = (correct_predictions / passed) * 100
print(f"Accuracy: {accuracy}%")
    
    
    