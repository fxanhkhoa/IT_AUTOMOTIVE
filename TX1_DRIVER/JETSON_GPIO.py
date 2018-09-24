
#gpio36 = 36,      // J21 - Pin 32 - Unused - AO_DMIC_IN_CLK
#gpio37 = 37,      // J21 - Pin 16 - Unused - AO_DMIC_IN_DAT
#gpio38 = 38,      // J21 - Pin 13 - Bidir  - GPIO20/AUD_INT
#gpio63 = 63,      // J21 - Pin 33 - Bidir  - GPIO11_AP_WAKE_BT
#gpio184 = 184,    // J21 - Pin 18 - Input  - GPIO16_MDM_WAKE_AP
#gpio186 = 186,    // J21 - Pin 31 - Input  - GPIO9_MOTION_INT
#gpio187 = 187,    // J21 - Pin 37 - Output - GPIO8_ALS_PROX_INT
#gpio219 = 219,    // J21 - Pin 29 - Output - GPIO19_AUD_RST

SYSFS_GPIO_DIR = '/sys/class/gpio'



# gpioExport
# Export the given gpio to userspace;
# Return: Success = 0 ; otherwise open file error
def gpioExport ( jetsonGPIO ):
    f = open(SYSFS_GPIO_DIR + "/export", "w")
    f.write(jetsonGPIO)
    f.close()
    
    
# gpioUnexport
# Unexport the given gpio from userspace
# Return: Success = 0 ; otherwise open file error
def  gpioUnexport ( jetsonGPIO ):
    f = open(SYSFS_GPIO_DIR + "/unexport", "w")
    f.write(jetsonGPIO)
    f.close()
    
# gpioSetDirection
# Set the direction of the GPIO pin 
# Return: Success = 0 ; otherwise open file error
def gpioSetDirection (jetsonGPIO, out_flag):
    f = open(SYSFS_GPIO_DIR + "/gpio" + jetsonGPIO + "/direction", "w")
    if (out_flag == 1):
        f.write('out')
    else:
        f.write('in')
    f.close()

def 
