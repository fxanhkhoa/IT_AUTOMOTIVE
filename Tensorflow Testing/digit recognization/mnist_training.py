# TensorFlow and tf.keras
import tensorflow as tf
from keras.datasets import mnist
from tensorflow import keras
from skimage.feature import hog
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

print(tf.__version__)

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


# load (downloaded if needed) the MNIST dataset
(train_images, train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

num_pixels = train_images.shape[1] * train_images.shape[2]

print('num pixels', num_pixels)
    
#train_images = train_images.reshape(train_images.shape[0], num_pixels).astype('float32')
# reshape to be [samples][pixels][width][height]
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

train_images = train_images / 255

# one hot encode outputs
train_labels = np_utils.to_categorical(train_labels)
num_classes = train_labels.shape[1]


print('train label', num_classes)


# create model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(train_images, train_labels, epochs=5, batch_size=200, verbose=2)

##########################
#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape))
# 64 3x3 kernels
#model.add(Conv2D(64, (3, 3), activation='relu'))
# Reduce by taking the max of each 2x2 block
#model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
#model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
#model.add(Flatten())
# A hidden layer to learn with
#model.add(Dense(128, activation='relu'))
# Another dropout
#model.add(Dropout(0.5))
# Final categorization from 0-9 with softmax
#model.add(Dense(10, activation='softmax'))

#model.summary()

#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
              
#history = model.fit(train_images, train_labels,batch_size=32,epochs=5,verbose=2)
##########################

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save entire model to a HDF5 file
#model.save_weights('./checkpoints/my_checkpoint')
model.save_weights('digit.h5')
#model.save_weights('digit_weight.h5')