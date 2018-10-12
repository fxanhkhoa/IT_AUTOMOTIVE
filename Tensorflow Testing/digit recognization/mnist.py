
# coding: utf-8

# In[1]:


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#from skimage.feature import hog
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json
from keras.utils import np_utils
import cv2


# In[2]:


#Helper library
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

num_pixels = 784
num_classes = 10
# create model
#model = Sequential()
#model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
#model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

model.summary()
# Compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights('digit.h5')
print("Loaded model from disk")
model.summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[5]:


cap = cv2.VideoCapture(0)

while True:
    #ret, im = cap.read()
    ret, im = cap.retrieve(cv2.CAP_OPENNI_BGR_IMAGE)
    ret, im_gray = cap.retrieve(cv2.CAP_OPENNI_GRAY_IMAGE)
    # Convert to grayscale and apply Gaussian filtering
    #im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)        
    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ima, ctrs, hier = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, ctrs, -1, (173,213,0), 3)
    
    if len(ctrs) > 0:
        for contour in ctrs:
            try:
                if (cv2.contourArea(contour) > 300):
                    # Get rectangles contains each contour
                    rect = cv2.boundingRect(contour) 
                    print(rect)
                    # Draw the rectangles
                    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
                    # Make the rectangular region around the digit
                    leng = int(rect[3] * 1.6)
                    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
                    # Resize the image
                    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                    roi = cv2.dilate(roi, (3, 3))
                    roi = np.array(roi)
                    roi = roi.astype('float32')
                    roi /= 255
                    roi = roi.reshape(28,28,1)

                    roi = np.expand_dims(roi, axis=0)
                    predictions = model.predict(roi).argmax()

                    cv2.putText(im, str(int(predictions)), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
                    print('answer: ',predictions)

                    cv2.imshow('result', im)
            except:
                pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

