import cv2
import numpy as np

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

import driver.driver_Lib as driverLib

NUM_CLASSES = 4
IMG_SIZE = 48

# upper blue = rgb(0, 117, 196)
# lower blue = rgb(0, 50, 84)

class Sign:
	
	def __init__(self):
		self.driver = driverLib.DRIVER()
		self.model = self.cnn_model()
		self.model.summary()
		self.model.load_weights('model.h5')
		
		lr = 0.01
		self.sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy',
			optimizer=self.sgd,
			metrics=['accuracy'])
		print('init tf done')
		return
		
	def predict(self, img):
		result = -1
		bienbao = ['stop','right', 'left', 'another']
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		#img = cv2.bilateralFilter(img, 5, 175, 175) # smoothing
		img = cv2.GaussianBlur(img,(15,15),0) #smoothing this is better
		masked = self.getMask(img)
		ret, im_th = cv2.threshold(masked, 95, 255, cv2.THRESH_BINARY_INV)
		#im_th = cv2.Canny(im_th, 75, 200)
		im_contours, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		#cv2.imshow('contour', im_contours)
		list_ellipse = []
		if len(contours) > 0:
			
			for contour in contours:
				area = cv2.contourArea(contour)
				if area <= 1000:  # skip ellipses smaller than 10x10
					continue
				try:
					#ellipse = cv2.fitEllipse(contour)
					#poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
					# if contour has enough similarity to an ellipse
					#similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, cv2.CONTOURS_MATCH_I2, 0)
					approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
					#print(area,' ', approx)
					#this is an ellpise
					if ((len(approx) > 8) & (area > 2000) & (area < 70000) ):
						self.driver.setSpeed(5)
					#if similarity <= 0.2: # result is not good as approx
						#print(similarity)
						#list_ellipse.append(similarity)
						#cv2.drawContours(img, contours, i, (0,255,0), 3)
						# Get rectangles contains each contour
						rect = cv2.boundingRect(contour) 
						# Draw the rectangles
						cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
						leng = int(rect[3] * 1.)
						pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
						pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
						
						
						roi = img[pt1:pt1+leng, pt2:pt2+leng]
						roi = cv2.dilate(roi, (3, 3))
						output_roi = self.preprocess_img(roi)
						output_roi = np.array(output_roi)
						output_roi = np.expand_dims(output_roi, axis=0)
						result = self.model.predict_classes(output_roi)
						
						cv2.putText(img, bienbao[int(result)], (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

				except:
					pass
		#out.write(im)
		cv2.imshow('signhere', img)
		return result
		
	def getMask(self, img):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		# define range of blue color in HSV
		lower_blue = np.array([70, 50, 50])
		upper_blue = np.array([150, 255, 255])
		
		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
		
		#cv2.imshow('mask', mask)
		return mask
	
	def getLowerUpper(self):
		#24,146,220
		green = np.uint8([[[220, 146, 24]]]) #IN BGR
		hsvGreen = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
		print(hsvGreen)
		lowerLimit = (hsvGreen[0][0][0]-10,100,100)
		upperLimit = (hsvGreen[0][0][0]+10,255,255)
		print(upperLimit)
		print(lowerLimit)
		
	def preprocess_img(self,img):
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
		
	def cnn_model(self):
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
