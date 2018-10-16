import cv2
import numpy as np

# upper blue = rgb(0, 117, 196)
# lower blue = rgb(0, 50, 84)

class Sign:
	
	def __init__(self):
		return
		
	def getMask(self, img):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		# define range of blue color in HSV
		lower_blue = np.array([91, 100, 100])
		upper_blue = np.array([111, 255, 255])
		
		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
		
		cv2.imshow('mask', mask)
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
