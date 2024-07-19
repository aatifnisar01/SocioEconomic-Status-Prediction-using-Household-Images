# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:54:10 2022

@author: Aatif
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import seaborn as sn
import pandas as pd
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.applications.vgg16 import preprocess_input
print(tf.__version__)

#!nvidia-smi

#import tensorflow as tf
#device_name = tf.test.gpu_device_name()
#if device_name != '/device:GPU:0':
#  raise SystemError('GPU device not found')
#print('Found GPU at: {}'.format(device_name))
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(True)

#a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#c = tf.matmul(a, b)
#print(c)


data_dir = 'C:/Users/Aatif/Downloads/archive/HouseInterior/HouseInterior'
# data_dir = './small_dataset/'
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# .glob() retrieves pathnames having ".jpg" in name.
# Image.open() opens and identifies given image.

from PIL import Image
all_images_list_names = list(data_dir.glob('*/*.jpg'))
im = Image.open(list(data_dir.glob('*/*.jpg'))[0])

# Aspect Ratio: proportional relationship between an image's width and height.
# Aspect Ratio = 1 for square images 

aspect_ratios = np.array([])

for path in tqdm(all_images_list_names[:50]):
  im = Image.open(path)
  aspect_ratios = np.append(aspect_ratios, im.size[0]/im.size[1])
print(np.mean(aspect_ratios))


# Model Architecture

IMG_SIZE = 120
batch_size = 32
#img_height = int(IMG_SIZE//np.mean(aspect_ratios))
img_height = 75
img_width = IMG_SIZE
print('image shape: ', (img_height,img_width))


base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='max', input_tensor=None, input_shape=(img_height,img_width,3))
#base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='./inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', pooling='avg', input_shape=(img_height,img_width,3), input_tensor=None)


# base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', pooling='avg', input_tensor=None, input_shape=(img_height,img_width,3))
# base_model = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg', input_shape=(img_height,img_width,3), input_tensor=None)

model = Sequential()

# base_model + model

model.add(base_model)

# initializer = tf.keras.initializers.GlorotUniform()
# model.add(Dropout(0.2))
model.add(Dense( 128, kernel_initializer='normal', kernel_regularizer='l2'))
# model.add(Dense(256, kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(64, kernel_initializer='normal', kernel_regularizer='l2'))
# model.add(Dense(32, kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(5, activation='softmax'))

model.layers[0].trainable = True

for layerr in model.layers[0].layers:
  # print(layerr)
  layerr.trainable = False

model.summary()

# Variable 'basemodel' contains InceptionV3 architecture except for fc layers
# Variable 'model' contains fc layers. 
# Only parameters of fc are trainable

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
#with tf.device('/gpu:0'):


# ImageDataGenerator:lets you augment your images in real-time while your model is still training     
# You can apply any random transformations on each training image as it is passed to the model

# ImageDataGenerator of Training data

image_gen_train = ImageDataGenerator(rescale=1./255, 
                                         zoom_range=0.2, 
                                         rotation_range=10,
                                         shear_range=0.09,
                                         horizontal_flip=True,
                                         vertical_flip=False,
                                        validation_split=0.2)

# ImageDataGenerator of Training data

image_gen_val = ImageDataGenerator(rescale=1./255,
                                        validation_split=0.2)

# Train Split

train_data_gen = image_gen_train.flow_from_directory(subset='training',
                                                         batch_size=batch_size,
                                                         directory=data_dir,
                                                         shuffle=True,
                                                         target_size=(img_height, img_width),
                                                         class_mode='sparse',seed=42)

# Validation Split

val_data_gen = image_gen_val.flow_from_directory(subset='validation',
                                                         batch_size=batch_size,
                                                         directory=data_dir,
                                                         shuffle=True,
                                                         target_size=(img_height, img_width),
                                                         class_mode='sparse',seed=42)


lr = 5e-3
epochs = 10
min_lr = 1e-8
batch_size = batch_size

# Complile Model

from tensorflow.keras.callbacks import ReduceLROnPlateau,LambdaCallback

opt = tf.keras.optimizers.Adam(learning_rate=lr)

# ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=min_lr)

model.compile(optimizer=opt, loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print('\n',model.layers[-5].get_weights()[0][60]))


# Training 

#with tf.device('/gpu:0'):
hisTory = model.fit(train_data_gen,
        steps_per_epoch = train_data_gen.samples // batch_size,
        validation_steps = val_data_gen.samples // batch_size,
        validation_data=val_data_gen,
        epochs=epochs, callbacks=[reduce_lr])

# Training VS Validation Loss

import matplotlib.pyplot as plt
hist = hisTory.history
x_range = range(epochs)
plt.plot(x_range, hist['loss'] ,label='train loss')
plt.plot(x_range, hist['val_loss'] ,label='valid loss')
plt.legend(loc='best')
plt.savefig('loss'+'.png')

# Training VS Validation Accuracy

x_range = range(epochs)
plt.plot(x_range, hist['accuracy'], label='train acc')
plt.plot(x_range, hist['val_accuracy'], label='valid acc')
plt.legend(loc='best')
plt.savefig('acc'+'.png')

# Confusion Matrix

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_pred = np.array(model.predict(val_data_gen), dtype='float64')
y_pred = [np.argmax(l) for l in y_pred] 
y_true = val_data_gen.labels

df_cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index = range(5),
              columns = range(5))
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True,cmap="OrRd")

