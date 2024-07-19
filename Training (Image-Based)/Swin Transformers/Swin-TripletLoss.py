# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:28:45 2022

@author: Aatif
"""



# **************************************** IMPORT LIBRARIES ***********************

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

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

# *************************************** CREATE MODEL *****************************
classes = 5
batch_size = 16

# now, for the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


HUB_URL = "SharanSMenon/swin-transformer-hub:main"
MODEL_NAME = "swin_tiny_patch4_window7_224"
# check hubconf for more models.
model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)

n_inputs = model.head.in_features
model.head = nn.Sequential(
    #nn.Linear(n_inputs, 512),
    #nn.ReLU(),
    #nn.Dropout(0.3),
    #nn.Linear(512, len(classes))
    #nn.Linear(n_inputs, len(classes))
    nn.Flatten()
)
model = model.to(device)

# Model summary before freezing
from torchinfo import summary
summary(model, input_size=(batch_size,3,224,224))

optimizer = optim.SGD(model.parameters(), lr=0.001)


# ******************************* FUNCTIONS ******************************
 
### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


# ********************************************* LOAD DATA *****************************

def get_data_loaders(data_dir, batch_size, train = False):
    if train:
        #train
        transform = T.Compose([
            T.Resize([224,224]),
        #T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #T.RandomHorizontalFlip(),
            #T.RandomVerticalFlip(),
           # T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            #T.Resize(256),
            #T.CenterCrop(224),
            #T.ToTensor(),
            #T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
            #T.RandomErasing(p=0.1, value='random')
        ])
        
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, len(train_data)
    else:
        # val/test
        transform = T.Compose([ # We dont need augmentation for test transforms
                    T.Resize([224,224]),
        #T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        
        val_data = datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform=transform)
        #test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return val_loader, len(val_data)
    
    
dataset_path = "C:/AHI data/Interior_Images/"

(train_loader, train_data_len) = get_data_loaders(dataset_path, batch_size, train=True)
(val_loader, valid_data_len) = get_data_loaders(dataset_path, batch_size, train=False)

# Shape and columns for dataloader
dataiter = iter(train_loader)
images, labels = next(dataiter)
print(type(images))
print(images.shape)
print(labels.shape)

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



# ******************************************** TRIPLET LOSS TRAINING ******************

### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)

train_losses = []  #new


def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    
    running_loss = 0  #new
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() #new
        train_loss = running_loss/len(train_loader)  #new
        
        train_losses.append(train_loss)
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )
            
num_epochs = 2

for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)

# Save the weights of the models
torch.save(model.state_dict(), "C:/Users/Public/Downloads/Swinmodel_weights.pt")

#plot losses

plt.plot(train_losses,'-o')
#plt.plot(eval_losses,'-o')
plt.xlabel('epoch')
plt.ylabel('losses')
plt.legend(['Train','Valid'])
plt.title('Train vs Valid Losses')

plt.show()