import cv2
import numpy as np

class deviation:
	
	def __init__(self):
		pass

	def getDeviation(self, image):
		crop_img = image[100:240, 0:320]
		im_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		return 
