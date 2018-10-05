import JETSON_GPIO as jet
import time

inputPin = 0
outputPin = 1
low = 0
high = 1
off = 0
on = 1

#gpio184 -> led1
#gpio510 -> led2
#gpio37 -> led3

#gpio36 -> button1
#gpio9 -> button2
#gpio10 -> button3
#gpio187 -> button4

#init led & button
led1 = jet.JETSON_GPIO(10)
#led2 = jet.JETSON_GPIO(510)
#led3 = jet.JETSON_GPIO(37)

btn1 = jet.JETSON_GPIO(184)
#btn2 = jet.JETSON_GPIO(38)
#btn3 = jet.JETSON_GPIO(10)
#btn4 = jet.JETSON_GPIO(187)

#unexport
try:
	led1.gpioUnexport()
	#led2.gpioUnexport()
	#led3.gpioUnexport()

	btn1.gpioUnexport()
	#btn2.gpioUnexport()
	#btn3.gpioUnexport()
	#btn4.gpioUnexport()
	print('unexported')
except:
	pass

time.sleep(.100)

#export
led1.gpioExport()
#led2.gpioExport()
#led3.gpioExport()

btn1.gpioExport()
#btn2.gpioExport()
#btn3.gpioExport()
#btn4.gpioExport()

led1.gpioSetDirection(outputPin)
#led2.gpioSetDirection(outputPin)
#led3.gpioSetDirection(outputPin)

btn1.gpioSetDirection(inputPin)
#btn2.gpioSetDirection(inputPin)
#btn3.gpioSetDirection(inputPin)
#btn4.gpioSetDirection(inputPin)

print('exported & set direction')
led1.gpioSetValue(on)
#led2.gpioSetValue(off)
#led3.gpioSetValue(off)

time.sleep(1)
led1.gpioSetValue(off)
while (True):
	#pass
	value = (int)(btn1.gpioGetValue())
	print(btn1.gpioGetValue())
	if (value == 1):
		led1.gpioSetValue(off)
	else:
		led1.gpioSetValue(on)
	
	#if (btn2.gpioGetValue() == high):
		#led2.gpioSetValue(on)
	#else:
		#led2.gpioSetValue(off)
		
	#if (btn3.gpioGetValue() == high):
		#led3.gpioSetValue(on)
	#else:
		#led3.gpioSetValue(off)
