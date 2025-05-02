#ifndef UART_DEVICE_H
#define UART_DEVICE_H

#include "main.h"

#define MAX_DATASIZE 256

typedef struct uart_device
{
	char rx_data[MAX_DATASIZE];
	int rx_length;
	UART_HandleTypeDef serial;	
}uart_device;

#endif