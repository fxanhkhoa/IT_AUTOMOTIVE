import PCA9685 as pca

ratio = 0.5
multiplier = 4096
STEERING_CHANNEL = 0
MOTOR_LEFT_IN1  = 4
MOTOR_LEFT_IN2  = 5
MOTOR_RIGHT_IN1  = 6
MOTOR_RIGHT_IN2  = 7

servoMin = 120 * 2 
servoMax = 720 / 2

#init, open and set address
pca9865 = pca.PCA9685()
#pca9865.setAllPWM(0,0)
pca9865.reset()
#pca9865.setPWMFrequency(60)
pca9865.close()
while (True):
	pass
	#pca9865.setPWM(STEERING_CHANNEL, 0, 500)
	#pca9865.setPWM(MOTOR_LEFT_IN1, 0, 0)
	#pca9865.setPWM(MOTOR_LEFT_IN2, 0, 0)


