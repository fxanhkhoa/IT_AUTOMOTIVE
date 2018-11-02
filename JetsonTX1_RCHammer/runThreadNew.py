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
import threading
import time
#Frame 640 x 480

class detectLaneThread(threading.Thread):
    def __init__(self, threadID, name, frame, q_angle):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame = frame
        self.q_angle = q_angle
        self.devi = deviationFromSobel.deviation()
        self.lock = threading.Lock()
        self.lock.acquire()
        return

    def run(self):
        try:
            #img = self.frame.get()
            img = self.frame
            img = cv2.resize(img, (640, 480))
            cv2.imshow('lane', img)
            angle = self.devi.process_image(img)
        except:
            pass
        finally:
            pass

    def setFrame(self, frame):
        self.frame = frame
        self.run()

class detectSignThread(threading.Thread):
    def __init__(self, threadID, name, frame, q_sign):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame = frame
        self.q_sign = q_sign
        self.sign = detectSign.Sign()
        return

    def run(self):
        try:
            #img = self.frame.get()
            img = self.frame
            img = cv2.resize(img, (640, 480))
            #cv2.imshow('lane', img)
            result = self.sign.predict(img)
        except:
            pass
        finally:
            #self.lock.release()
            pass

    def setFrame(self, frame):
        self.frame = frame
        self.run()

def main():
    print("Hello World!")
    frame = queue.Queue()
    q_angle = queue.Queue()
    q_sign = queue.Queue()

    threadLane = detectLaneThread(1, 'lane', frame, q_angle)
    threadSign = detectSignThread(2, 'sign', frame, q_sign)
    threadLane.start()
    threadSign.start()
    source = 'video_withSign.mp4'

    cap = cv2.VideoCapture(source)
    while True:
        start = time.time()
        ret, frame = cap.read()

        threadLane.setFrame(frame)
        threadSign.setFrame(frame)

        end = time.time()
        elapsed = end - start
        print('1 FRAME USE: ', elapsed)

        if cv2.waitKey(1)& 0xFF == ord('q'):
            break

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
