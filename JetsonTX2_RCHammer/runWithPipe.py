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
from multiprocessing import Process
#Frame 640 x 480
sumtime = 0
class detectLaneThread(Process):
    def __init__(self, threadID, name, frame, q_angle, q_done):
        Process.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame = frame
        self.q_angle = q_angle
        self.q_done = q_done
        self.devi = deviationFromSobel.deviation()
        self.lock = threading.Lock()
        self.lock.acquire()
        return

    def run(self):
        global sumtime
        try:
            start = time.time()
            #img = self.frame.get()
            img = self.frame
            img = cv2.resize(img, (640, 480))
            #cv2.imshow('lane', img)
            angle = self.devi.process_image(img)
            self.q_done.put(1)
            end = time.time()
            sumtime = sumtime + (end - start)
            print('sumtime = ', sumtime)
        except:
            pass
        finally:
            pass

    def setFrame(self, frame):
        self.frame = frame
        self.run()

class detectSignThread(Process):
    def __init__(self, threadID, name, frame, q_sign, q_done):
        Process.__init__(self)
        self.threadID = threadID
        self.name = name
        self.frame = frame
        self.q_sign = q_sign
        self.q_done = q_done
        self.sign = detectSign.Sign()
        return

    def run(self):
        global sumtime
        try:
            start = time.time()
            #img = self.frame.get()
            img = self.frame
            img = cv2.resize(img, (640, 480))
            #cv2.imshow('lane', img)
            result = self.sign.predict(img)
            self.q_done.put(1)
            end = time.time()
            sumtime = sumtime + (end - start)
            print('sumtime = ', sumtime)
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
    q_done1 = queue.Queue()
    q_done2 = queue.Queue()
    q_done3 = queue.Queue()
    q_done4 = queue.Queue()

    q_done1.put(1)
    q_done2.put(1)
    q_done3.put(1)
    q_done4.put(1)

    threadLane = detectLaneThread(1, 'lane', frame, q_angle, q_done1)
    threadSign = detectSignThread(2, 'sign', frame, q_sign, q_done2)
    threadLane1 = detectLaneThread(3, 'lane', frame, q_angle, q_done3)
    threadSign1 = detectSignThread(4, 'sign', frame, q_sign, q_done4)

    #threadLane.start()
    #threadSign.start()
    source = 'video_withSign.mp4'
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        start = time.time()
        if q_done1.get() == 1:
            ret, frame = cap.read()
            threadLane.setFrame(frame)
            #threadSign.setFrame(frame)
        if q_done2.get() == 1:
            ret, frame = cap.read()
            threadSign.setFrame(frame)
        if q_done3.get() == 1:
            ret, frame = cap.read()
            threadLane1.setFrame(frame)
        if q_done4.get() == 1:
            ret, frame = cap.read()
            threadSign1.setFrame(frame)

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
