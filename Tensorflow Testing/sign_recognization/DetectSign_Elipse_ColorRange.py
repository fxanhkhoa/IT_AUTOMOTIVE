
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

import os


# In[2]:


def getMask(img):
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		# define range of blue color in HSV
		lower_blue = np.array([82, 27, 6])
		upper_blue = np.array([130, 255, 255])
		
		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lower_blue, upper_blue)
		
		#cv2.imshow('mask', mask)
		return mask


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


# In[4]:


im = cv2.imread('images/turn_left/002.jpg')
path = os.getcwd() + '/images/turn_left_extracted' 
getLowerUpper()


# In[7]:


im = cv2.resize(im, (768,1024))
im_save = im.copy()
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
masked = getMask(im_save)
#plt.imshow(im)
#cv2.imshow('im',masked)

ret, im_th = cv2.threshold(im_gray, 95, 255, cv2.THRESH_BINARY_INV)

im_contours, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
list_ellipse = []
i = 0
j = 0
if len(contours) > 0:
    for contour in contours:
        i = i + 1
        area = cv2.contourArea(contour)
        if area <= 50000:  # skip ellipses smaller than 10x10
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
            poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
            # if contour has enough similarity to an ellipse
            similarity = cv2.matchShapes(poly.reshape((poly.shape[0], 1, poly.shape[1])), contour, cv2.CONTOURS_MATCH_I2, 0)
            #this is an ellpise
            if similarity <= 0.2: 
                print(i, ' ', similarity, area)
                list_ellipse.append(similarity)
                cv2.drawContours(im, contours, i, (0,255,0), 3)
                # Get rectangles contains each contour
                rect = cv2.boundingRect(contour) 
                # Draw the rectangles
                cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
                leng = int(rect[3] * 1.)
                pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
                pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
                roi = im_save[pt1:pt1+leng, pt2:pt2+leng]
                roi = cv2.dilate(roi, (3, 3))
                plt.imshow(roi)
                plt.show()
                #cv2.imshow('roi', roi)
                #name_file = str(j) + '.jpg'
                #cv2.imwrite(os.path.join(path , name_file), roi)
        except:
            pass
        
cv2.waitKey()