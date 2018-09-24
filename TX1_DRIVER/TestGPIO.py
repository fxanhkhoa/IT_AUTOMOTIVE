import JETSON_GPIO

inputPin = 0
outputPin = 1
low = 0
high = 1
off = 0
on = 1


gpio36 = JETSON_GPIO(36)
gpio36.gpioExport()
gpio36.gpioSetDirection(outputPin)

