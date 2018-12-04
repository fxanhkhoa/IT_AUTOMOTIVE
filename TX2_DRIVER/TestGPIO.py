import JETSON_GPIO as jet
import time
import cv2

inputPin = 0
outputPin = 1
low = 0
high = 1
off = 0
on = 1

#gpio10 -> led1
#gpio9 -> led2
#gpio36 -> led3

#gpio187 -> button1
#gpio186 -> button2
#gpio163 -> button3
#gpio184 -> button4

#init led & button
led1 = jet.JETSON_GPIO(393)
led2 = jet.JETSON_GPIO(394)
led3 = jet.JETSON_GPIO(297)

btn1 = jet.JETSON_GPIO(388)
btn2 = jet.JETSON_GPIO(298)
btn3 = jet.JETSON_GPIO(467)
btn4 = jet.JETSON_GPIO(255)

#unexport
try:
	led1.gpioUnexport()
	led2.gpioUnexport()
	led3.gpioUnexport()

	btn1.gpioUnexport()
	btn2.gpioUnexport()
	btn3.gpioUnexport()
	btn4.gpioUnexport()
	print('unexported')
except:
	pass

time.sleep(.100)

#export
led1.gpioExport()
led2.gpioExport()
led3.gpioExport()

btn1.gpioExport()
btn2.gpioExport()
btn3.gpioExport()
btn4.gpioExport()

led1.gpioSetDirection(outputPin)
led2.gpioSetDirection(outputPin)
led3.gpioSetDirection(outputPin)

btn1.gpioSetDirection(inputPin)
btn2.gpioSetDirection(inputPin)
btn3.gpioSetDirection(inputPin)
btn4.gpioSetDirection(inputPin)

print('exported & set direction')
led1.gpioSetValue(on)
led2.gpioSetValue(on)
led3.gpioSetValue(on)

time.sleep(1)
led1.gpioSetValue(off)
led2.gpioSetValue(off)
led3.gpioSetValue(off)
while (True):
	try:
		#pass
		value1 = (int)(btn1.gpioGetValue())
		value2 = (int)(btn2.gpioGetValue())
		value3 = (int)(btn3.gpioGetValue())
		value4 = (int)(btn4.gpioGetValue())
		#print('1 -> ',btn1.gpioGetValue())
		#print('2 -> ',btn2.gpioGetValue())
		#print('3 -> ',btn3.gpioGetValue())
		print('4 -> ',btn4.gpioGetValue())
		if (value1 == 1):
			led1.gpioSetValue(on)
		else:
			led1.gpioSetValue(off)
		
		if (value2 == 1):
			led2.gpioSetValue(on)
		else:
			led2.gpioSetValue(off)
			
		if (value3 == 1):
			led3.gpioSetValue(on)
		else:
			led3.gpioSetValue(off)
			
		if (value4 == 1):
			led3.gpioSetValue(on)
		else:
			led3.gpioSetValue(off)
			
	except KeyboardInterrupt:
		#led1.gpioUnexport()
		#led2.gpioUnexport()
		#led3.gpioUnexport()

		btn1.gpioUnexport()
		#btn2.gpioUnexport()
		#btn3.gpioUnexport()
		#btn4.gpioUnexport()
		print('unexported')
	
