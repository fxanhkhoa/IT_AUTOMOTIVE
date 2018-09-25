import PCA9865 as pca

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
pca9865.setAllPWM(0,0)
pca.setPWM(STEERING_CHANNEL, 0, (servoMax - servoMin) / 2);