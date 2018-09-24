import JETSON_GPIO as jet
import time

inputPin = 0
outputPin = 1
low = 0
high = 1
off = 0
on = 1


gpio36 = jet.JETSON_GPIO(36)
gpio36.gpioUnexport()
time.sleep(.100)
gpio36.gpioExport()
gpio36.gpioSetDirection(outputPin)

while (True):
	gpio36.gpioSetValue(on)
	time.sleep(.500)
	gpio36.gpioSetValue(off)
	time.sleep(.500)

