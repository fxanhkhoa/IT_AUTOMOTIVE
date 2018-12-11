#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include "jetsonGPIO.h"
#include "JHPWMPCA9685.h"

#define ESC_CHANNEL	0
#define SERVO_CHANNEL   4

#define MAX_FORWARD  300
#define MIN_FORWARD  380
#define FORWARD  80

#define NEUTRAL  264
#define CENTRAL	345 

using namespace std;

int main()
{
	jetsonTX2GPIONumber LED1 = gpio393 ;     // Ouput
	jetsonTX2GPIONumber LED2 = gpio394 ;     // Ouput
	jetsonTX2GPIONumber LED3 = gpio297 ;     // Ouput

	jetsonTX2GPIONumber btn_start_stop = gpio388 ; // Input
	jetsonTX2GPIONumber btn_mode = gpio298 ; // Input
	jetsonTX2GPIONumber btn_speed_plus = gpio467 ; // Input
	jetsonTX2GPIONumber btn_speed_minus = gpio255 ; // Input
	
	gpioUnexport(LED1) ;
	gpioUnexport(LED2) ;
	gpioUnexport(LED3) ;
	gpioUnexport(btn_start_stop) ;
	gpioUnexport(btn_mode) ;
	gpioUnexport(btn_speed_plus) ;
	gpioUnexport(btn_speed_minus) ;
	
	cout<<"All Unexported";

	gpioExport(LED1) ;
	gpioExport(LED2) ;
	gpioExport(LED3) ;
	gpioExport(btn_start_stop) ;
	gpioExport(btn_mode) ;
	gpioExport(btn_speed_plus) ;
	gpioExport(btn_speed_minus) ;

	cout<<"All Exported";
	
	gpioSetDirection(btn_start_stop, inputPin) ;
	gpioSetDirection(btn_mode, inputPin) ;
	gpioSetDirection(btn_speed_plus, inputPin) ;
	gpioSetDirection(btn_speed_minus, inputPin) ;

	gpioSetDirection(LED1,outputPin) ;
	gpioSetDirection(LED2,outputPin) ;
	gpioSetDirection(LED3,outputPin) ;

	cout<<"Setted Direction";

	gpioSetValue(LED1,off) ;
	gpioSetValue(LED2,off) ;
	gpioSetValue(LED3,off) ;

	PCA9685 *pca9685 ; // init motor
	pca9685 = new PCA9685();

	int err = pca9685->openPCA9685();
	if (err < 0){
		printf("Error: %d", pca9685->error);
	}

	pca9685->setAllPWM(0,0) ;
	pca9685->reset() ;
	pca9685->setPWMFrequency(60) ;
	sleep(1) ;
	
	unsigned int value = low;
	unsigned int speed = NEUTRAL;
	unsigned int angle = CENTRAL;	

	pca9685->setPWM(ESC_CHANNEL,0 , 0);
	pca9685->setPWM(SERVO_CHANNEL, 0, angle);
	
	while (1)
	{	
		gpioGetValue(btn_start_stop, &value) ;
		if (value == high)
		{
			gpioSetValue(LED1,on) ;
		}
		else
		{
			gpioSetValue(LED1,off) ;
		}
		gpioGetValue(btn_mode, &value) ;
		if (value == high)
		{	
			speed = NEUTRAL;
			pca9685->setPWM(ESC_CHANNEL, 0, 0);
			gpioSetValue(LED2,on) ;
		}
		else
		{
			gpioSetValue(LED2,off) ;
		}
		gpioGetValue(btn_speed_plus, &value) ;
		if (value == high)
		{
			speed--; // speed tang
			angle++; // qua trai
			cout<<speed<<"\n";
			pca9685->setPWM(ESC_CHANNEL, 0, speed);
			pca9685->setPWM(SERVO_CHANNEL, 0, angle);
			gpioSetValue(LED3,on) ;
			sleep(1);
		}
		else
		{
			gpioSetValue(LED3,off) ;
		}
		gpioGetValue(btn_speed_minus, &value) ;
		if (value == high)
		{
			speed++; // speed giam
			angle--; // qua phai
			cout<<speed<<"\n";
			pca9685->setPWM(ESC_CHANNEL, 0, speed);
			pca9685->setPWM(SERVO_CHANNEL, 0, angle);
			gpioSetValue(LED1,on) ;
			sleep(1);
		}
		else
		{
			gpioSetValue(LED1,off) ;
		}
	}
}
