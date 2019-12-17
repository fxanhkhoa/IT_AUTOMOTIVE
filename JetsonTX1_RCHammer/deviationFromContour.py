import cv2
import numpy as np
import math
import time

class deviation:

   def __init__(self):
      self.y = 150
      self.factor = 2
      self.preAngle = 0
      return

   def process(self, image):

      frame = cv2.resize(image, (320, 160))
   
      frame = frame[130:160, 0:320]

      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      lower = np.uint8([83, 0, 0])
      upper = np.uint8([180, 255, 255])

      white_mask = cv2.inRange(hsv, lower, upper)
      result = cv2.bitwise_and(frame, frame, mask = white_mask)
      cv2.imshow('inrange', result)
      print(result.shape)
      gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(gray ,140 ,255, cv2.THRESH_BINARY_INV)
      cv2.imshow('thresh', thresh)
      print(thresh.shape)
      ima, contours, hier = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      count = 0
      sum_cx = 0
      for contour in contours:
         area = cv2.contourArea(contour)
         print(area)
         if area < 6000:
            continue
         # cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
         M = cv2.moments(contour)
         cx = int(M['m10']/M['m00'])
         cy = int(M['m01']/M['m00'])
         sum_cx = sum_cx + cx
         count = count + 1
         # print('here')
      angle = 0
      print('count= ', count)
      if count > 0:
         X = sum_cx / count
         print('X = ', X)
         #cv2.circle(image ,(int(X), 20), 10, (0,0,255), -1)
         angle = self.GetAngle(X)
         self.preAngle = angle
         if (angle > 28):
            angle = 45
         elif (angle < -28):
            angle = -45
         # print('Angle: ', angle)
         # speed = GetSpeed(angle)
         # print('Speed: ', speed)
      cv2.imshow('contour', image)

      return angle

   def GetAngle(self, x, xshape = 160):
      # Calculate Angle
      #x = middle
      #xshape = (image.shape[1] / 2) 
      print('hieu so', x-xshape)
      value = math.atan2((x-xshape), self.y)
      result = value * 180 / math.pi
      # if ((result > 2) and (result < -2)):
      result = result * self.factor
      # if result < 0:
         # result = result + result / 7.5
      print('goc lech = ', result)
      return result

   def GetSpeed(self):
      pass
