#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

#define DEVICE_FILENAME  "/dev/savanna"
#define IOCTL_SETSM 13

int main(){
	int dev, ret;	//device number, return value
	int result;

	//system("mknod /dev/savanna c 250 32");

	//open device file
	dev = open(DEVICE_FILENAME, O_RDWR|O_NDELAY);
	if(dev < 0){
		printf("dev open error\nmake dev file (mknod /dev/savanna c 250 32) or insmod module (insmod savanna_mod.ko)\n");
		return 1;
	}

	result = ioctl(dev, IOCTL_SETSM, 0);
	
	ret = close(dev);

	//printf("closing dev file... ret: %d\n", ret);

	//system("rm -rf /dev/savanna");

	return 0;
}