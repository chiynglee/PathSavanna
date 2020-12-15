#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

#define DEVICE_FILENAME  "/dev/savanna"
#define IOCTL_BIND 5

typedef struct savanna_bind_info {
	unsigned short otherend_id;
	int ring_ref;
	int evtchn_port;
} savanna_bind_info;

int main(){
	int dev, ret;	//device number, return value
	int result;
	savanna_bind_info info;

	system("mknod /dev/savanna c 250 32");

	//open device file
	dev = open(DEVICE_FILENAME, O_RDWR|O_NDELAY);
	if(dev < 0){
		printf("dev open error\nmake dev file (mknod /dev/savanna c 250 32) or insmod module (insmod savanna_mod.ko)\n");
		return 1;
	}

	printf("insert otherend domain id : ");
	scanf("%d", &info.otherend_id);
	printf("insert shared buffer reference number : ");
	scanf("%d", &info.ring_ref);
	printf("insert otherend eventchannel port : ");
	scanf("%d", &info.evtchn_port);

	result = ioctl(dev, IOCTL_BIND, &info);
	if(result < 0)
		printf("Savanna connect fail\n");
	else 
		printf("Savanna connect success\n");

	ret = close(dev);

	//printf("closing dev file... ret: %d\n", ret);

	//system("rm -rf /dev/savanna");

	return 0;
}
