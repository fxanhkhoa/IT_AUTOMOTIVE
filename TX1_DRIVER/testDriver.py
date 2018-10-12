#sys.path.insert(0, r'/from/root/directory/application')
import driver_Lib as dl
from driver_Lib import DRIVER
import time

driver = DRIVER()
driver.turnOnLed1()
driver.turnOnLed2()
driver.turnOnLed3()

driver.setAngle(0)
time.sleep(2)

while True:
	if driver.getValuebtnStartStop() == 1:
		driver.turnOffLed1()
	else:
		driver.turnOnLed1()
	
	driver.setAngle(30)
	time.sleep(1)
	driver.setAngle(-30)
	time.sleep(1)
