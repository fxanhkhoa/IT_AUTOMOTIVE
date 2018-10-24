import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import deviationFromSobel
#import deviationRC
#import detectSign
import sys
#sys.path.append('/driver')
import driver.driver_Lib as driverLib

global rgb_stream
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

def main():
	print("Hello World!")
	running = 0
	
	rgb_stream = initialize()
	devi = deviationFromSobel.deviation()
	#devi2 = deviationRC.deviation()
	#sign = detectSign.Sign()
	driver = driverLib.DRIVER()
	driver.turnOnLed1()
	driver.turnOffLed2()
	driver.turnOnLed3()
	driver.setAngle(0)
	driver.setSpeed(0)
  
  
	#cap = cv2.VideoCapture('video.mp4')
	while True:
	  #print(driver.getValuebtnStartStop())
	  if (driver.getValuebtnStartStop() == 1):
		  
		  running = running -1
		  running = abs(running)
		  print(running)
		  driver.setSpeed(60)
		  #time.sleep(1)
		  while (driver.getValuebtnStartStop() != 0):
			  pass
	  if running == 0:
		  driver.setSpeed(0)
		  driver.setAngle(0)
      
	  if running == 1:
		  # for openni
		  bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
		  img = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
		  img = cv2.flip( img, 1 )
		  #ret,img = cap.read()
		  #img = cv2.resize(img, (640, 480))
		  #mask = sign.getMask(img)
		  #sign.getLowerUpper()
		  angle = devi.process_image(img)
		  #signResult = sign.predict(img)
		  driver.setAngle(int(angle))
		  #driver.setSpeed(0)
		  print(int(angle))
		  
		  if cv2.waitKey(33)& 0xFF == ord('q'):
			  break
        
	cap.release()
	cv2.destroyAllWindows()
    
if __name__== "__main__":
  main()

