import os
import fcntl
from smbus2 import SMBus

class PCA9685:
    kI2Caddress = 0x40
    # Constants scraped from <linux/i2c-dev.h> and <linux/i2c.h>
    _I2C_SLAVE = 0x0703
    _I2C_IOC_FUNCS = 0x705
    _I2C_IOC_RDWR = 0x707
    _I2C_FUNC_I2C = 0x1
    _I2C_M_TEN = 0x0010
    _I2C_M_RD = 0x0001
    _I2C_M_STOP = 0x8000
    _I2C_M_NOSTART = 0x4000
    _I2C_M_REV_DIR_ADDR = 0x2000
    _I2C_M_IGNORE_NAK = 0x1000
    _I2C_M_NO_RD_ACK = 0x0800
    _I2C_M_RECV_LEN = 0x0400
    
    def __init__(self):
        self.kI2CBus = 1
        self.error = 0
        self.bus = smbus.SMBus(0)
    
    def openPCA9685(self):
        # Open i2c device
        try:
            self.fd = os.open("/dev/i2c-" + str(self.kI2CBus), os.O_RDWR)
        except OSError as e:
            raise I2CError(e.errno, "Opening I2C device: " + e.strerror)
            return False
        try:
            fcntl.ioctl(self.fd, self._I2C_SLAVE, self.kI2Caddress)
        except OSError as e:
            raise I2CError(e.errno, "Opening I2C device: " + e.strerror)
            return False
        return True
    
    def closePCA9685(self):
        try:
            self.fd.close()
        except OSError as e:
            raise I2CError(e.errno, "Closing I2C device: " + e.strerror)
            return False
        return True
    
    # Read the given register
    def readByte(readRegister):
        