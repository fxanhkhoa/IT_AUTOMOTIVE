# IT_AUTOMOTIVE
-----
## Before use, install these
* Install I2C DEV for Python
https://pypi.org/project/i2cdev/
* Install SMBus
https://pypi.org/project/smbus2/
* Install tensorflow for Nvidia Jetson TX1 and Nvida Jetson TK1
https://github.com/peterlee0127/tensorflow-nvJetson
* Install hp5y
http://docs.h5py.org/en/latest/build.html or https://stackoverflow.com/questions/37778299/cannot-install-h5py
* Install Keras
https://www.hackster.io/wilson-wang/jetson-tx2-tensorflow-opencv-keras-install-b74e40
* Install Openni for Python and Camera test is in Example and Tutorial
https://pypi.org/project/openni/
* Openni2 with python
https://github.com/elmonkey/Python_OpenNI2
* Install sklearn and skimage
https://gist.github.com/xydrolase/977979968728c9eb0e48 (With python3 change python-* to python3-*)

## About project
* Simulation of Automatic car
* Working on Jetson TX1 with CUDA and Tensorflow

## Libraries in project
- [x] GPIO library in Python

- [x] I2C library in Python

- [x] PCA9685 library

- [ ] SPI library

- [ ] Pwm library

## schematics and PCBs in project
* Shield for Nvidia Jetson TX1
  - [x] 3 Leds (GPIO)
  
  - [x] 4 buttons (GPIO)
  
  - [x] I2C for MPU6050 and PCA9685
  
  - [ ] SPI
  
* Shiled for Nvidia Jetson TK1
  - [x] 4 buttons (GPIO)
  
  - [x] I2C for MPU6050 and PCA9685
  
## Tensorflow and Machine learning
* Traffic sign recognization
  - [x] Turn left
  
  - [x] Turn right
  
  - [ ] Stop
  
  - [ ] Park
