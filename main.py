# general libraries
import numpy as np
import matplotlib.pyplot as plt

# main libraries
import keras
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#cnn model

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
    
model.compile(optimizer="adam",loss=keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
    
model.summary()

# preaparing data using data generator
def imagePreaparation1(path):
    
    # ip : Path
    # op : processed images
    
    imageData = ImageDataGenerator(zoom_range = .2,         #data augmentation
                                   shear_range = .2, 
                                   rescale = 1/255, 
                                   horizontal_flip = True)
    
    image = imageData.flow_from_directory(directory = path, 
                                          target_size = (224,224), 
                                          batch_size = 32, 
                                          class_mode = "binary")
    
    return image


def imagePreaparation2(path):
    
    # ip : Path
    # op : processed images
    
    imageData = ImageDataGenerator(rescale = 1/255)
    
    image = imageData.flow_from_directory(directory = path, 
                                          target_size = (224,224), 
                                          batch_size = 32, 
                                          class_mode = "binary")
    
    return image

# calling and preaparing
tainingDataPath = "data/Training"
trainingData = imagePreaparation1(tainingDataPath)
testingDataPath = "data/Testing"
testingData = imagePreaparation2(testingDataPath)
validationDataPath = "data/Validation"
validationData = imagePreaparation2(validationDataPath)

# early stopping and model checking
# early stopping
es = EarlyStopping(monitor="val_accuracy", 
                   min_delta=.01, 
                   patience=3, 
                   verbose=1, 
                   mode= 'auto')
# model check point
mc = ModelCheckpoint(monitor="val_accuracy",
                     filepath='./bestModel.keras',
                     verbose=1,
                     save_best_only=True,
                     mode='auto')

cd = [es,mc]

# Model Training
hs = model.fit(trainingData,
               steps_per_epoch= 8,
               epochs = 30,
               verbose = 1,
               validation_data = validationData,
               validation_steps = 16,
               callbacks = cd)
















