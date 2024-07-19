# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 21:03:46 2022

@author: Aatif
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T # for simplifying the transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy

# remove warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import sys
from tqdm import tqdm
import time
import copy




#******************************************** LOAD DATA ******************************************


def get_data_loaders(data_dir, batch_size, train = False):
    
    if train:
        #train
        transform = T.Compose([
        T.Resize([224,224]),
        

        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomHorizontalFlip(p=0.25),
        #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
        #T.RandomVerticalFlip(),
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
        #T.Resize(256),
        #T.CenterCrop(224),
        #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
        T.RandomErasing(p=0.2, value='random')
                              ])
       
        
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, len(train_data)
    
    else:
        # val/test
        transform = T.Compose([ # We dont need augmentation for test transforms
        T.Resize([224,224]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomHorizontalFlip(p=0.25),
        #T.RandomBrightnessContrast(bightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.25),
        T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.15),
        T.RandomErasing(p=0.1, value='random')
                             ])
       
       
        val_data = datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform=transform)
        #test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return val_loader, len(val_data)




dataset_path = "C:/AHI data/cleaned/"
batch_size = 8

(train_loader, train_data_len) = get_data_loaders(dataset_path, batch_size , train=True)
(val_loader, valid_data_len) = get_data_loaders(dataset_path, batch_size , train=False)


# Shape and columns for dataloader
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(type(images))
print(images.shape)
print(labels.shape)




# ********************************************** GET CLASSES ********************************************



def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

classes = get_classes(dataset_path + "train/")
print(classes, len(classes))


dataloaders = {
    "train": train_loader,
    "valid": val_loader
}
dataset_sizes = {
    "train": train_data_len,
    "valid": valid_data_len
}

print(len(train_loader), len(val_loader))

print(train_data_len, valid_data_len)

# *********************************************** VISUALIZE BATCH *********************************



# =============================================================================
# from torchvision.utils import make_grid
# def show_batch(dl):
#     for images, labels in dl:
#         fig, ax = plt.subplots(figsize=(24, 24))
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.imshow(make_grid(images[:64], nrow=4).permute(1, 2, 0))
#         break
# 
# show_batch(train_loader)
# =============================================================================


# ******************************************** MODEL ********************************************************


# now, for the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

HUB_URL = "SharanSMenon/swin-transformer-hub:main"
MODEL_NAME = "swin_tiny_patch4_window7_224"
# check hubconf for more models.
model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True) # load from torch hub


# Model summary before freezing
from torchinfo import summary
summary(model, input_size=(batch_size,3,224,224))

# As we can see last dense layer is still there, so I will remove it manually

  
#model3 = torch.nn.Sequential(*(list(model.children())[:-1]))        
 


for param in model.parameters():
    param.requires_grad = False



n_inputs = model.head.in_features
model.head = nn.Sequential(
    #nn.Linear(n_inputs, 512),
    #nn.ReLU(),
    #nn.Dropout(0.3),
    #nn.Linear(512, len(classes))
    nn.Linear(n_inputs, len(classes))
)
model = model.to(device)

# overall model summary
summary(model, input_size=(batch_size,3,224,224))


print(model.head)
#print(model)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.SGD(model.head.parameters(), lr=0.001)

# lr scheduler : Decay lr by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)




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
   
    inputs,labels=data[0].to(device),data[1].to(device)
   
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
    for data in tqdm(val_loader):
      images,labels=data[0].to(device),data[1].to(device)
     
      outputs=model(images)

      loss= criterion(outputs,labels)
      running_loss+=loss.item()
     
      _, predicted = outputs.max(1)
      total += labels.size(0)
      correct += predicted.eq(labels).sum().item()
 
  test_loss=running_loss/len(val_loader)
  accu=100.*correct/total

  eval_losses.append(test_loss)
  eval_accu.append(accu)

  print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
 
# ***************************************** RUN ***************************************

epochs=12
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