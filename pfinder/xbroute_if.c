#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <net/route.h>
#include <netinet/in.h>

int main(int argc, char* argv[]){
	int fd;
	struct rtentry rtentry;
	struct sockaddr_in sin_dest, sin_mask, sin_gate;
	struct nexthop *nexthop;
		
	printf("--------------------------------------\n");
	printf("* KU_OSL_PathFinder: Routing Table Update *\n");
	printf("--------------------------------------\n");

	if (argc < 2) {
		printf("Add Usage: ./xbroute -a [dest ip addr] [netmask] [gateway] [interface]\n");
		printf("Del Usage: ./xbroute -d [dest ip addr] [netmask]\n");
		printf("netmask example : 255.255.255.0\n");
		return 0;
		}
	
	int dst = inet_addr(argv[2]);
	int mask = inet_addr(argv[3]);
	
	memset (&rtentry, 0, sizeof (struct rtentry));
	sin_dest.sin_family = AF_INET;
	sin_dest.sin_addr.s_addr = dst;
	memset (&sin_mask, 0, sizeof (struct sockaddr_in));
	sin_mask.sin_family = AF_INET;
  	sin_mask.sin_addr.s_addr = mask;
	memcpy (&rtentry.rt_dst, &sin_dest, sizeof (struct sockaddr_in));
	memcpy (&rtentry.rt_genmask, &sin_mask, sizeof (struct sockaddr_in));
			
	if (!strcmp(argv[1], "-a")) {

		memset (&sin_gate, 0, sizeof (struct sockaddr_in));
		sin_gate.sin_family = AF_INET;
		sin_gate.sin_addr.s_addr = inet_addr(argv[4]);
		rtentry.rt_dev = argv[5];

		memcpy (&rtentry.rt_gateway, &sin_gate, sizeof (struct sockaddr_in));
		fd = open("/proc/net/xebra/ipv4", O_RDWR);
		ioctl(fd, SIOCADDRT, &rtentry);

		printf("Route table entry add complete.\n");
		
	}
	else if (!strcmp(argv[1], "-d")) {
		fd = open("/proc/net/xebra/ipv4", O_RDWR);
		ioctl(fd, SIOCDELRT, &rtentry);

		printf("Route table entry delete complete.\n");

	}

	else {
		printf("Add Usage: ./xbroute -a [dest ip addr] [netmask] [gateway] [interface]\n");
		printf("Del Usage: ./xbroute -d [dest ip addr] [netmask]\n");
		printf("netmask example : 255.255.255.0\n");
	}

/*	switch(*argv[0]) {
	case '-a':	
		int gateway = inet_addr(*argv[3]);
		char ifname[] = *argv[4];
	

		memset (&sin_gate, 0, sizeof (struct sockaddr_in));
		sin_gate.sin_family = AF_INET;
		sin_gate.sin_addr.s_addr = gateway;
		rtentry.rt_dev = ifname;

		memcpy (&rtentry.rt_gateway, &sin_gate, sizeof (struct sockaddr_in));
		fd = open("/proc/net/xebra/ipv4", O_RDWR);
		ioctl(fd, SIOCADDRT, &rtentry);

		printf("Route table entry add complete.\n");
		
		return 0;

	case '-d':
	
		fd = open("/proc/net/xebra/ipv4", O_RDWR);
		ioctl(fd, SIOCDELRT, &rtentry);

		return 0;
	default:
		printf("Add Usage: ./xbroute -a [dest ip addr] [netmask] [gateway] [interface]\n");
		printf("Del Usage: ./xbroute -d [dest ip addr] [netmask]\n");
		printf("netmask example : 255.255.255.0\n");
	}
*/
	return 0;
}
