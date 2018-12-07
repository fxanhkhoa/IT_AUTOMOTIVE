import PCA9685 as pca
import time

ratio = 0.5
multiplier = 4096
STEERING_CHANNEL = 0
MOTOR_LEFT_IN1  = 4
MOTOR_LEFT_IN2  = 5
MOTOR_RIGHT_IN1  = 6
MOTOR_RIGHT_IN2  = 7
# toi 363 min -> 420
# neutral 332
# lui 309 min -> 270 max
PWM_FULL_REVERSE = 332 # 1ms/20ms * 4096
PWM_NEUTRAL = 307      # 1.5ms/20ms * 4096
PWM_FULL_FORWARD = 480 # 2ms/20ms * 4096

value = 364

servoMin = 120 * 2 
servoMax = 720 / 2
# servo giam qua phai
# tang qua trai
# middle = 349
# max left = 434
# max right = 264

#init, open and set address
pca9865 = pca.PCA9685()
pca9865.setAllPWM(0,0)
pca9865.reset()
pca9865.setPWMFrequency(60)
#pca9865.setPWM(0, 0, 370)
time.sleep(2)
pca9865.setPWM(1, 0, value)

#pca9865.close()
while (True):
	#pass
	#pca9865.setPWM(1, 0, value)
	#time.sleep(.3)
	#value = value - 1
	#print(value)
	#pca9865.setPWM(0, 0, 500)
	#time.sleep(.8)
	value = value + 1
	print('pwm', value)
	pca9865.setPWM(0, 0, value)
	time.sleep(.5)
	#pca9865.setPWM(MOTOR_LEFT_IN1, 0, 0)
	#pca9865.setPWM(MOTOR_LEFT_IN2, 0, 0)


