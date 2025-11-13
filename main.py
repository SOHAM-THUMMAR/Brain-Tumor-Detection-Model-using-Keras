# general libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob

# main libraries
import keras
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#cnn model
def Model():
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = (3,3), activation="relu", input_shape=(224,224,3) ))
    
    model.add(Conv2D(filters = 32, kernel_size = (3,3), activation="relu" ))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation="relu" ))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Conv2D(filters = 128, kernel_size = (3,3), activation="relu" ))
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Dropout(rate = .25))
    
    model.add(Flatten())
    model.add(Dense(units = 64, activation= "relu"))
    model.add(Dropout(rate = 0.25))
    model.add(Dense( units = 1, activation= "sigmoid"))
    
    model.compile(optimizer="adam",loss=keras.losses.BinaryCrossentropy,metrics=["accuracy"])
    
    model.summary()

# preaparing data using data generator
def imagePreaparation(path):
    
    # ip : Path
    # op : processed images
    
    imageData = ImageDataGenerator(zoom_range = .2, shear_range = .2, rescale = 1/255, horizontal_flip = True)
    image = imageData.flow_from_directory(directory = path,target_size = (224,224), batch_size = 32, class_mode = "binary")
    
    return image
# run
Model()
path = "data/Training"
trainingData = imagePreaparation(path)




