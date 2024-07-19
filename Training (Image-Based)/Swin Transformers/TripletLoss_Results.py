#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 11:38:23 2023

@author: aatif
"""

import sys
sys.path.insert(1,'/home/aatif/Documents/AlexNet_Triplet')
import os
import json
import torch
import PIL
from model.AlexNet import AlexNet
#from utils.dataset import build_dataloader
#from utils.common import get_all_embeddings, get_accuracy, log_to_file
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torchvision import datasets, transforms
from multiprocessing import cpu_count
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Augmentation.RandomRotation import RandomRotationWithProb

torch.backends.cudnn.benchmark = True

# Constants
EPOCHS = 40
BATCH_SIZE = 8

IMAGE_SIZE = 384
EMBEDDING_SIZE = 64

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda:0")
TRAIN_DATASET = "/hdd/Aatif/cleaned/train"
VAL_DATASET = "/hdd/Aatif/cleaned/train"
SAVE_PATH = "/home/aatif/Documents/AlexNet_Triplet/weights"

data_dir = "/hdd/Aatif/cleaned/"


# ********************************************* LOAD DATA ***********************************

transform1 = T.Compose([
 T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
 

 T.ToTensor(),
 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
 T.RandomHorizontalFlip(p=0.5),
 #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
 #T.RandomVerticalFlip(0.5),
 #T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
 #T.Resize(256),
 #T.CenterCrop(224),
 #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
 #T.RandomErasing(p=0.1, value='random')
                       ])

transform2 = T.Compose([
 T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
 

 T.ToTensor(),
 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
 #T.RandomHorizontalFlip(p=0.4),
 #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
 #T.RandomVerticalFlip(0.2),
 T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.2),
 RandomRotationWithProb(degrees=15, p=0.4)
 #T.RandomApply(T.RandomRotation(degrees=(-5,5)),p=0.3),
 #T.RandomRotation(degrees=(-5,5),p=0.3),
 #T.Resize(256),
 #T.CenterCrop(224),
 #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
 #T.RandomErasing(p=0.1, value='random')
                       ])

        
transform3 = T.Compose([
 T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
 T.ToTensor(),
 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
 #T.RandomHorizontalFlip(p=0.25),
 #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
 #T.RandomVerticalFlip(0.1),
 #T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
 #T.Resize(256),
 #T.CenterCrop(224),
 #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
 T.RandomErasing(p=0.15, value='random')
                       ])





        
train_data = []        
train_data.append(datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform1))
train_data.append(datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform2))
train_data.append(datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform3))


train_data = torch.utils.data.ConcatDataset(train_data)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

train_data_len = len(train_data)

# Shape and columns for dataloader
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(type(images))
print(images.shape)
print(labels.shape)




# ********************************************* VALIDATION DATA *********************************


transform4 = T.Compose([
 T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
 

 T.ToTensor(),
 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
 #T.RandomHorizontalFlip(p=0.5),
 #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
 #T.RandomVerticalFlip(0.5),
 #T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
 #T.Resize(256),
 #T.CenterCrop(224),
 #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
 #T.RandomErasing(p=0.1, value='random')
                       ])

# =============================================================================
# transform5 = T.Compose([
#  T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
#  
# 
#  T.ToTensor(),
#  #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#  #T.RandomHorizontalFlip(p=0.4),
#  #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
#  #T.RandomVerticalFlip(0.2),
#  T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.2),
#  #T.RandomApply(T.RandomRotation(degrees=(-5,5)),p=0.3),
#  #T.RandomRotation(degrees=(-5,5),p=0.3),
#  #T.Resize(256),
#  #T.CenterCrop(224),
#  #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
#  #T.RandomErasing(p=0.1, value='random')
#                        ])
# 
#         
# transform6 = T.Compose([
#  T.Resize([IMAGE_SIZE,IMAGE_SIZE]),
#  T.ToTensor(),
#  #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#  #T.RandomHorizontalFlip(p=0.25),
#  #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
#  #T.RandomVerticalFlip(0.5),
#  #T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
#  #T.Resize(256),
#  #T.CenterCrop(224),
#  #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
#  T.RandomErasing(p=0.15, value='random')
#                        ])
# =============================================================================





        
valid_data = []        
valid_data.append(datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform = transform4))
#valid_data.append(datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform = transform5))
#valid_data.append(datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform = transform6))

valid_data = torch.utils.data.ConcatDataset(valid_data)

valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

valid_data_len = len(valid_data)




# ********************************************** GET CLASSES ********************************************


def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

classes = get_classes(data_dir + "train/")
print(classes, len(classes))


dataloaders = {
    "train": train_loader,
    "valid": valid_loader
}
dataset_sizes = {
    "train": train_data_len,
    "valid": valid_data_len
}

print(len(train_loader), len(valid_loader))

print(train_data_len, valid_data_len)




model = torch.load("C:/Users/Public/Downloads/Swinmodel_weights.pt")




# ********************** Classifier **********************

# =============================================================================
# for param in model.parameters():
#     param.requires_grad = False
# =============================================================================
    
from torchinfo import summary
summary(model, input_size=(BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE))

print(model)


new_layer = torch.nn.Linear(512,5)

model.head.add_module("dropout", torch.nn.Dropout(0.3))
model.head.add_module("new_layer", new_layer)

print(model)

summary(model, input_size=(BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE))

print(len(list(model.parameters())))
 

    
    
for name, param in list(model.named_parameters())[:327]:    # 315 for 1_24Jan 
    print('I will be frozen: {}'.format(name)) 
    param.requires_grad = False
    
summary(model, input_size=(BATCH_SIZE,3,IMAGE_SIZE,IMAGE_SIZE))

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(DEVICE)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

from tqdm import tqdm


# ************************************** TRAINING *****************************************************

train_losses=[]
train_accu=[]

def train(epoch):
  print('\nEpoch : %d'%epoch)
 
  model.train()

  running_loss=0
  correct=0
  total=0

  for data in tqdm(train_loader):
   
    inputs,labels=data[0].to(DEVICE),data[1].to(DEVICE)
   
    outputs=model(inputs)
   
    loss=criterion(outputs,labels)
   
   
   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
   
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
     
  train_loss=running_loss/len(train_loader)
  accu=100.*correct/total
 
  train_accu.append(accu)
  train_losses.append(train_loss)
  print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
 
# ****************************************** VALIDATION ***************************************
 
eval_losses=[]
eval_accu=[]

def test(epoch):
  model.eval()

  running_loss=0
  correct=0
  total=0

  with torch.no_grad():
    for data in tqdm(valid_loader):
      images,labels=data[0].to(DEVICE),data[1].to(DEVICE)
     
      outputs=model(images)

      loss= criterion(outputs,labels)
      running_loss+=loss.item()
     
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
 
  test_loss=running_loss/len(valid_loader)
  accu=100.*correct/total

  eval_losses.append(test_loss)
  eval_accu.append(accu)

  print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
 
# ***************************************** RUN ***************************************

epochs=5
for epoch in range(1,epochs+1):
  train(epoch)
  test(epoch)
 
# ***************************************** VISUALIZATION ******************************

#plot accuracy

plt.plot(train_accu,'-o')
plt.plot(eval_accu,'-o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Accuracy')

plt.show()


#plot losses

plt.plot(train_losses,'-o')
plt.plot(eval_losses,'-o')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Losses')

plt.show()