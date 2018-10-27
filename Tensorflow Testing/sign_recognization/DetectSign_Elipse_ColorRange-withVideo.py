
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure, transform

import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')

NUM_CLASSES = 4
IMG_SIZE = 48

def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, IMG_SIZE, IMG_SIZE),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,-1)

    return img


def getMask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([82, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img ,img , mask= mask)
    
    rgb = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    #cv2.imshow('res',rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    #cv2.imshow('mask', mask)
    return mask, gray


# ## Function For Get Limit Of Color (In Opencv is BGR not RGB)

# In[3]:


def getLowerUpper():
	#24,146,220
	green = np.uint8([[[220, 146, 24]]]) #IN BGR
	hsvGreen = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
	print(hsvGreen)
	lowerLimit = (hsvGreen[0][0][0]-10,100,100)
	upperLimit = (hsvGreen[0][0][0]+10,255,255)
	print(upperLimit)
	print(lowerLimit)

try:
	with  h5py.File('model.h5') as hf: 
		X, Y = hf['imgs'][:], hf['labels'][:]
	print("Loaded images from X.h5")
except:
	pass
	
model = cnn_model()
model.summary()
model.load_weights('model.h5')

# let's train the model using SGD + momentum (how original).
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
          optimizer=sgd,
          metrics=['accuracy'])
	
#im = cv2.imread('002.jpg')
#cv2.imshow('im',im)
#path = os.getcwd() + '/images/turn_left_extracted' 
#getLowerUpper()


# In[7]:
bienbao = ['stop','right', 'left', 'another']
cap = cv2.VideoCapture('video_withSign.mp4')
out = cv2.VideoWriter('00202_out1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))
while(True):
    # Capture frame-by-frame
    ret, im = cap.read()
    im = cv2.resize(im, (640,480))
    im_save = im.copy()
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    masked, im_gray1 = getMask(im_save)

    ret, im_th = cv2.threshold(masked, 95, 255, cv2.THRESH_BINARY)

    im_contours, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('contour',masked)

    list_ellipse = []
    i = 0
    j = 0
    if len(contours) > 0:
        for contour in contours:
            i = i + 1
            area = cv2.contourArea(contour)
            #print(i, ' ', area)
            if area <= 1000:  # skip ellipses smaller than 10x10
                continue
            try:
                #print(i, ' ', area)
                ellipse = cv2.fitEllipse(contour)
                poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
                # if contour has enough similarity to an ellipse
                similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, cv2.CONTOURS_MATCH_I2, 0)
                #this is an ellpise
                rect = cv2.boundingRect(contour) 
                # Draw the rectangles
                cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
                if similarity <= 0.2: 
                    #print(i, ' ', similarity, area)
                    list_ellipse.append(similarity)
                    #cv2.drawContours(im, contours, i, (0,255,0), 3)
                    # Get rectangles contains each contour
                    #rect = cv2.boundingRect(contour) 
                    # Draw the rectangles
                    #cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
                    leng = int(rect[3] * 1.)
                    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                    roi = im_save[pt1:pt1+leng, pt2:pt2+leng]
                    roi = cv2.dilate(roi, (3, 3))
                    #cv2.imwrite('extracted.png',roi)
                    output_roi = preprocess_img(roi)
                    #cv2.imshow('roi', output_roi)
                    output_roi = np.array(output_roi)
                    output_roi = np.expand_dims(output_roi, axis=0)
                    #print(output_roi)
                    result = model.predict_classes(output_roi)
                    cv2.putText(im, bienbao[int(result)], (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
                    #print(result)
                    # 0_stop, 1_right, 2_left
            except:
                pass
    out.write(im)
    cv2.imshow('im', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
