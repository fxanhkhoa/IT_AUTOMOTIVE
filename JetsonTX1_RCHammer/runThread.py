import cv2
import numpy as np
#from openni import openni2
#from openni import _openni2 as c_api
import deviationFromSobel
#import deviationRC
import detectSign
import sys
#sys.path.append('/driver')
#import driver.driver_Lib as driverLib

#import tty
#import termios	
from termcolor import colored

import _thread, queue
import time

global img
#Frame 640 x 480

def initialize():
	openni2.initialize()     # can also accept the path of the OpenNI redistribution
	
	dev = openni2.Device.open_any()
	print(dev.get_device_info())

	rgb_stream = dev.create_color_stream()

	print('The rgb video mode is', rgb_stream.get_video_mode()) # Checks rgb video configuration
	rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))

	## Start the streams
	rgb_stream.start()
	
	return rgb_stream

def detect_lane(threadName, q, q_angle):
	
	#driver.setAngle(int(angle))
	#driver.setSpeed(0)
	devi = deviationFromSobel.deviation()
	try:
		while True:
			img = q.get()
			angle = devi.process_image(img)
			#print(int(angle))
			q_angle.put(int(angle))
			#cv2.imshow('lane', img)
			if cv2.waitKey(33)& 0xFF == ord('q'):
				break
	except:
		stop()
		pass
	return

def detect_sign(threadName, q, q_sign):
	sign = detectSign.Sign()
	try:
		while True:
			img = q.get()
			#sign_ = q_model.get()
			result = sign.predict(img)
			#cv2.imshow('sign', img)
			if cv2.waitKey(33)& 0xFF == ord('q'):
				break
	except:
		stop()
	return

def control_thread(threadName, q_angle, q_sign):
	try:
		while True:
			angle = q_angle.get()
			print(angle)
	except:
		stop()
	return

def main():
	print("Hello World!")
	
	#orig_settings = termios.tcgetattr(sys.stdin)
	
	#tty.setraw(sys.stdin)
	x = 0
	
	running = 1
	q = queue.Queue()
	q_angle = queue.Queue()
	q_sign = queue.Queue() # to set result of sign
	_thread.start_new_thread(detect_lane, ('lane', q, q_angle, ))
	_thread.start_new_thread(detect_sign, ('sign', q, q_sign))
	_thread.start_new_thread(control_thread, ('control', q_angle, q_sign, ))
	cap = cv2.VideoCapture('video_withSign.mp4')

	while True:
		try:
			ret,image = cap.read()
			image = cv2.resize(image, (640, 480))
			q.put(image)
			#cv2.imshow('in read', image)
			if cv2.waitKey(33)& 0xFF == ord('q'):
				break
		except:
			pass
		
	cap.release()
	cv2.destroyAllWindows()
	#rgb_stream = initialize()
	#devi2 = deviationRC.deviation()
	#driver = driverLib.DRIVER()
	#driver.turnOnLed1()
	#driver.turnOffLed2()
	#driver.turnOnLed3()
	#driver.setAngle(0)
	#driver.setSpeed(0)
  
  
	
	
def stop():
	#driver = driverLib.DRIVER()
	#driver.turnOnLed1()
	#driver.turnOffLed2()
	#driver.turnOnLed3()
	#driver.setAngle(0)
	#driver.setSpeed(0)
	print(colored('program crash', 'red'))
	print(colored('stopped motor', 'green'))
	return
    
if __name__== "__main__":
	main()
  #try:
	#  main()
  #except:
	#  stop()

