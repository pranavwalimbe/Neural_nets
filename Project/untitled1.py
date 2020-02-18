# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 16:28:24 2018

@author: Pranav
"""
#Setting the working directory
#setwd = 'D:/Udemy Machine learning/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/Convolutional_Neural_Networks/dataset'

#Importing the dataset
import matplotlib as plt
import pandas as pd
import numpy as np
#Part 1 - Building the CNN
import keras

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers
#Initializing the convolution model
classifier = Sequential()

#Creating the method for model
#Step 1- Convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64,64,3), activation = 'relu'))

#Step 2- Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

#adding another layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
#Pooling it
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Adding another layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Adding another layer
classifier.add(Convolution2D(32,3,3, activation = 'relu'))
#Pooling
classifier.add(MaxPooling2D(pool_size =(2,2)))

#Add reularizers
classifier.add(Dense(128, input_dim = 64, 
                     kernel_regularizer = regularizers.l2(0.01), 
                     activity_regularizer = regularizers.l1(0.01), activation = 'relu'))

#Step 3- Flattening
classifier.add(Flatten())

#Step 4- Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
#For the output step
classifier.add(Dense(units = 10, activation = 'softmax'))

#Compiling the  CNN, i.e gradient descent
classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display 
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from PIL import _imaging
from PIL import ImageTk


train_datagen = ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Test',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='categorical')

classifier.fit_generator(training_set,
                    steps_per_epoch=(1592/32),
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=(351/32))

#Testing out on our input image
from keras.preprocessing import image as image_utils
test_image = image_utils.load_img('18808911.jpg', target_size=(64, 64))
test_image = image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

test_set.class_indices
 
result = classifier.predict_on_batch(test_image)
result.argmax()

for category, value in test_set.class_indices.items():
    if value == result.argmax():
        print(category)


from helper import get_class_names
class_names = get_class_names()

#saving the model as h5
#pranav swaminathan github
#classifier.save("D:/MS course work/Python projects/objectdetection.h5")