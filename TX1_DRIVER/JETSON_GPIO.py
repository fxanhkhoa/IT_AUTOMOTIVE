
#gpio36 = 36,      // J21 - Pin 32 - Unused - AO_DMIC_IN_CLK
#gpio37 = 37,      // J21 - Pin 16 - Unused - AO_DMIC_IN_DAT
#gpio38 = 38,      // J21 - Pin 13 - Bidir  - GPIO20/AUD_INT
#gpio63 = 63,      // J21 - Pin 33 - Bidir  - GPIO11_AP_WAKE_BT
#gpio184 = 184,    // J21 - Pin 18 - Input  - GPIO16_MDM_WAKE_AP
#gpio186 = 186,    // J21 - Pin 31 - Input  - GPIO9_MOTION_INT
#gpio187 = 187,    // J21 - Pin 37 - Output - GPIO8_ALS_PROX_INT
#gpio219 = 219,    // J21 - Pin 29 - Output - GPIO19_AUD_RST

class JETSON_GPIO:
    SYSFS_GPIO_DIR = '/sys/class/gpio'

    def __init__(self, jetsonGPIO):
        self.jetsonGPIO = jetsonGPIO
        

    # gpioExport
    # Export the given gpio to userspace;
    # Return: Success = 0 ; otherwise open file error
    def gpioExport ( self ):
        f = open(SYSFS_GPIO_DIR + "/export", "w")
        f.write(self.jetsonGPIO)
        f.close()
        return
    
    
    # gpioUnexport
    # Unexport the given gpio from userspace
    # Return: Success = 0 ; otherwise open file error
    def  gpioUnexport ( self ):
        f = open(SYSFS_GPIO_DIR + "/unexport", "w")
        f.write(self.jetsonGPIO)
        f.close()
        return

    # gpioSetDirection
    # Set the direction of the GPIO pin 
    # Return: Success = 0 ; otherwise open file error
    def gpioSetDirection (self, out_flag):
        f = open(SYSFS_GPIO_DIR + "/gpio" + self.jetsonGPIO + "/direction", "w")
        if (out_flag == 1):
            f.write('out')
        else:
            f.write('in')
        f.close()
        return

    # gpioSetValue
    # Set the value of the GPIO pin to 1 or 0
    # Return: Success = 0 ; otherwise open file error
    def gpioSetValue( self, value):
        f = open(SYSFS_GPIO_DIR + "/gpio" + self.jetsonGPIO + "/value", "w")
        if (value == 1):
            f.write('1')
        else:
            f.write('0')
        f.close()
        return
        
    # gpioGetValue
    # Get the value of the requested GPIO pin ; value return is 0 or 1
    # Return: Success = 0 ; otherwise open file error
    def gpioGetValue( self ):
        f = open(SYSFS_GPIO_DIR + "/gpio" + self.jetsonGPIO + "/value", "r")
        return f.read()
    
    # gpioSetEdge
    # Set the edge of the GPIO pin
    # Valid edges: 'none' 'rising' 'falling' 'both'
    # Return: Success = 0 ; otherwise open file error
    def gpioSetEdge( self, edge):
        f = open(SYSFS_GPIO_DIR + "/gpio" + self.jetsonGPIO + "/edge", "w")
        f.write(edge)
        f.close()
        return
    

