import JETSON_GPIO as Jet

inputPin = 0
outputPin = 1
low = 0
high = 1
off = 0
on = 1


gpio36 = jet.JETSON_GPIO(36)
gpio36.gpioExport()
gpio36.gpioSetDirection(outputPin)

gpio36.gpioSetValue(on)

while (True):
	pass
