import JETSON_GPIO as jet
import PCA9685 as pca
from termcolor import colored

######## PCA9685 ########
STEERING_CHANNEL = 0
MOTOR_CHANNEL = 1

# toi 363 min -> 420
# neutral 332
# lui 309 min -> 270 max
MAX_FORWARD = 420
MIN_FORWARD = 363
FORWARD = 57

NEUTRAL = 332

MAX_REVERSE = 270
MIN_REVERSE = 309
REVERSE = 39

####### GPIO ########
inputPin = 0
outputPin = 1
low = 0
high = 1
off = 0
on = 1

####### SERVO #######
middle = 349
maxLeft = 445
maxRight = 264
servo_range = 85


class DRIVER:
	
	def __init__(self):
		self.pca9865 = pca.PCA9685()
		self.led1 = jet.JETSON_GPIO(10)
		self.led2 = jet.JETSON_GPIO(9)
		self.led3 = jet.JETSON_GPIO(36)

		self.btn_start_stop = jet.JETSON_GPIO(187)
		self.btn_mode = jet.JETSON_GPIO(186)
		self.btn_speed_plus = jet.JETSON_GPIO(163)
		self.btn_speed_minus = jet.JETSON_GPIO(511)
		
		self.resetLed1()
		self.resetLed2()
		self.resetLed3()
		
		self.openLed1()
		self.openLed2()
		self.openLed3()
		
		#set direction GPIO #
		self.led1.gpioSetDirection(outputPin)
		self.led2.gpioSetDirection(outputPin)
		self.led3.gpioSetDirection(outputPin)
		
		self.resetButtonStartStop()
		self.resetButtonMode()
		self.resetButtonSpeedPlus()
		self.resetButtonSpeedMinus()
		
		self.openButtonStartStop()
		self.openButtonMode()
		self.openButtonSpeedPlus()
		self.openButtonSpeedMinus()
		
		#set direction BTN #
		self.btn_start_stop.gpioSetDirection(inputPin)
		self.btn_mode.gpioSetDirection(inputPin)
		self.btn_speed_plus.gpioSetDirection(inputPin)
		self.btn_speed_minus.gpioSetDirection(inputPin)
		
		print(colored('INIT ALL DONE', 'blue'))
				
	############# FOR LED ############
		#### RESET ####
	def resetLed1(self):
		try:
			self.led1.gpioUnexport()
			print(cologreen('led1 unexported', 'red'))
		except:
			pass
		return
		
	def resetLed2(self):
		try:
			self.led2.gpioUnexport()
			print('led2 unexported')
		except:
			pass
		return
		
	def resetLed3(self):
		try:
			self.led3.gpioUnexport()
			print('led3 unexported')
		except:
			pass
		return
		
		#### OPEN ####
	def openLed1(self):
		try:
			self.led1.gpioExport()
			print(colored('led1 Exported', 'green'))
		except:
			pass
		return
	
	def openLed2(self):
		try:
			self.led2.gpioExport()
			print('led2 Exported')
		except:
			pass
		return
		
	def openLed3(self):
		try:
			self.led3.gpioExport()
			print('led3 Exported')
		except:
			pass
		return
		
		#### On/Off ####
	
	def turnOnLed1(self):
		self.led1.gpioSetValue(on)
		return
	
	def turnOnLed2(self):
		self.led2.gpioSetValue(on)
		return
	
	def turnOnLed3(self):
		self.led3.gpioSetValue(on)
		return
		
	def turnOffLed1(self):
		self.led1.gpioSetValue(off)
		return
		
	def turnOffLed2(self):
		self.led2.gpioSetValue(off)
		return
		
	def turnOffLed3(self):
		self.led3.gpioSetValue(off)
		return
		
	########### FOR BUTTON ###########
		#### RESET ####
	def resetButtonStartStop(self):
		try:
			self.btn_start_stop.gpioUnexport()
			print('button start stop Unexported')
		except:
			pass
		return
		
	def resetButtonMode(self):
		try:
			self.btn_mode.gpioUnexport()
			print('button mode Unexported')
		except:
			pass
		return
		
	def resetButtonSpeedPlus(self):
		try:
			self.btn_speed_plus.gpioUnexport()
			print('button speed plus Unexported')
		except:
			pass
		return
		
	def resetButtonSpeedMinus(self):
		try:
			self.btn_speed_minus.gpioUnexport()
			print('button speed minus Unexported')
		except:
			pass
		return
		
		#### OPEN ####
		
	def openButtonStartStop(self):
		try:
			self.btn_start_stop.gpioExport()
			print('button start stop Exported')
		except:
			pass
		return
	
	def openButtonMode(self):
		try:
			self.btn_mode.gpioExport()
			print('button mode Exported')
		except:
			pass
		return
		
	def openButtonSpeedPlus(self):
		try:
			self.btn_speed_plus.gpioExport()
			print('button speed plus Exported')
		except:
			pass
		return
		
	def openButtonSpeedMinus(self):
		try:
			self.btn_speed_minus.gpioExport()
			print('button speed minus Exported')
		except:
			pass
		return
		
		#### Get Value ####
	def getValuebtnStartStop(self):
		value = (int)(self.btn_start_stop.gpioGetValue())
		return value
		
	def getValuebtnMode(self):
		value = (int)(self.btn_mode.gpioGetValue())
		return value
		
	def getValuebtnSpeedPlus(self):
		value = (int)(self.btn_speed_plus.gpioGetValue())
		return value
		
	def getValuebtnSpeedMinus(self):
		value = (int)(self.btn_speed_minus.gpioGetValue())
		return value
		
	########### FOR SERVO ###########
		
	def setAngle(self, angle):
		if (angle >= 0): #right
			# calculate percentage
			percentage = (int)((angle * 100) / 45)
			value = (int)((85 * percentage) / 100)
			self.pca9865.setPWM(STEERING_CHANNEL, 0, middle - value)
			print(colored(middle - value, 'red'))
		elif (angle < 0): #left
			angle = -angle
			# calculate percentage
			percentage = (int)((angle * 100) / 45)
			value = (int)((100 * percentage) / 100)
			self.pca9865.setPWM(STEERING_CHANNEL, 0, value + middle)
			print(colored(middle + value, 'red'))
		return
	########### FOR MOTOR ###########
	
	def setSpeed(self, speed):
		if speed > 100:
			speed = 100
		elif speed < -100:
			speed = -100
		if speed > 0:
			#calculate value
			value = (int)((FORWARD * 100) * speed)
			self.pca9865.setPWM(MOTOR_CHANNEL, 0, value + MIN_FORWARD)
		elif speed < 0:
			#calculate value
			value = - value
			value = (int)((FORWARD * 100) * speed)
			self.pca9865.setPWM(MOTOR_CHANNEL, 0, MIN_REVERSE - value)
			
