# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:15:51 2022

@author: Aatif
"""

import hashlib
#from scipy.misc import imread, imresize, imshow
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%matplotlib inline
import time
import numpy as np

def file_hash(filepath):
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()
    
import os

os.getcwd()
os.chdir(r'C:/AHI data/int')
os.getcwd()

file_list = os.listdir()
print(len(file_list))

import hashlib, os
duplicates = []
hash_keys = dict()
for index, filename in  enumerate(os.listdir('.')):  #listdir('.') = current directory
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            filehash = hashlib.md5(f.read()).hexdigest()
        if filehash not in hash_keys: 
            hash_keys[filehash] = index
        else:
            duplicates.append((index,hash_keys[filehash]))
            
duplicates

for file_indexes in duplicates[:30]:
    try:
    
        plt.subplot(121),plt.imshow(cv.imread(file_list[file_indexes[1]]))
        plt.title(file_indexes[1]), plt.xticks([]), plt.yticks([])

        plt.subplot(122),plt.imshow(cv.imread(file_list[file_indexes[0]]))
        plt.title(str(file_indexes[0]) + ' duplicate'), plt.xticks([]), plt.yticks([])
        plt.show()
    
    except OSError as e:
        continue
    
for index in duplicates:
    os.remove(file_list[index[0]])