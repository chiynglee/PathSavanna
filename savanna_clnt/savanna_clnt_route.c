////////////////////////////////////////////////
// savanna server in DomG					  //
////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <errno.h>
#include <unistd.h>
#include <memory.h>
#include <netdb.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h> 

#define SAVANNA_IP_NUM 100000
#define SAVANNA_BUF_NUM   200     // 커널과 유저 사이의 mmap space의 개수.
#define SAVANNA_ROUTE_NUM 64     // 라우팅 테이블의 규칙이 최대 몇 개인가를 나타냄. 필요하면 늘리면 됨. 별로 의미는 없음.

#define IP_GEN_POOL 0xFFFFFFFF

#define PAGE_NUM 10

int state;
struct timeval op_time[6];
struct timeval total_time[6];
struct timeval gpu_launch_time[6];

unsigned int *route_pages[PAGE_NUM];
unsigned int *hop_pages[PAGE_NUM];
unsigned char *h_TBLlong;
unsigned short *h_TBL24;

// DIR-24-8-BASIC 알고리즘 구현 중 TBL 테이블 생성에 사용됨
#define f1 0x1
#define f2 0x100
#define f3 0x10000
#define f4 0x1000000

// DIR-24-8-BASIC 알고리즘 구현 중 TBL 테이블 생성에 사용됨
#define f1 0x1
#define f2 0x100
#define f3 0x10000
#define f4 0x1000000

// DIR-24-8-BASIC 알고리즘 구현 중 TBL 테이블 생성에 사용됨
unsigned int mask[25]={
	0x000000,
	0x800000,
	0xC00000,
	0xE00000,
	0xF00000,
	0xF80000,
	0xFC0000,
	0xFE0000,
	0xFF0000,
	0xFF8000,
	0xFFC000,
	0xFFE000,
	0xFFF000,
	0xFFF800,
	0xFFFC00,
	0xFFFE00,
	0xFFFF00,
	0xFFFF80,
	0xFFFFC0,
	0xFFFFE0,
	0xFFFFF0,
	0xFFFFF8,
	0xFFFFFC,
	0xFFFFFE,
	0xFFFFFF
};

// DIR-24-8-BASIC 알고리즘 구현 중 TBL 테이블 생성에 사용됨
unsigned char charmask[9]={
	0x00,
	0x80,
	0xC0,
	0xE0,
	0xF0,
	0xF8,
	0xFC,
	0xFE,
	0xFF
};

void alloc_memory()
{
	int count;
	for(count = 0 ; count < PAGE_NUM; count++) 
	{
		route_pages[count] = (unsigned int *)malloc(sizeof(unsigned int) * 1024 * SAVANNA_BUF_NUM / 2);
		hop_pages[count] = (unsigned int *)malloc(sizeof(unsigned int) * 1024 * SAVANNA_BUF_NUM / 2);
	}

	h_TBL24 = (unsigned short *)malloc(sizeof(unsigned short) * f4);
	h_TBLlong = (unsigned char *)malloc(sizeof(unsigned char) * f3);
}

void free_memory() 
{
	int count;
	for(count = 0 ; count < PAGE_NUM; count++) 
	{
		free(route_pages[count]);
		free(hop_pages[count]);
	}
	free(h_TBL24);
	free(h_TBLlong);
}

// 라우팅 테이블 규칙을 입력받는 포맷. 어떤 ip의 prefix 몇짜리가 오면 pointer로 보낸다는 뜻
struct route_entry{
	unsigned char prefix;
	unsigned int ip;
	unsigned int pointer;
};

// 라우팅 규칙이 들어갈 배열
struct route_entry rentry[SAVANNA_ROUTE_NUM];

// 거듭제곱 함수
unsigned int pow_t(unsigned int n){
	unsigned int i;
	unsigned int r = 1;
	for(i=0; i<n; i++)
		r*=2;
	return r;
}

// 각 모듈별 소요 시간을 계산하는 함수
int getusec(struct timeval start, struct timeval end){
	return ((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec - start.tv_usec;
}

// 시험용 라우팅 테이블 엔트리.
void input_rentry(){
	rentry[0].prefix = 1; rentry[0].ip = 0x00000000; rentry[0].pointer = 1;
	rentry[1].prefix = 2; rentry[1].ip = 0x40000000; rentry[1].pointer = 2;
	rentry[2].prefix = 1; rentry[2].ip = 0x80000000; rentry[2].pointer = 3;
	rentry[3].prefix = 2; rentry[3].ip = 0xC0000000; rentry[3].pointer = 4;
	rentry[4].prefix = 8; rentry[4].ip = 0x0A000000; rentry[4].pointer = 5;
	rentry[5].prefix = 24; rentry[5].ip = 0x0182D078; rentry[5].pointer = 11;
	rentry[6].prefix = 26; rentry[6].ip = 0x030579E0; rentry[6].pointer = 12;
	rentry[7].prefix = 28; rentry[7].ip = 0x04882348; rentry[7].pointer = 13;
	rentry[8].prefix = 30; rentry[8].ip = 0xFE76D370; rentry[8].pointer = 14;
	rentry[9].prefix = 32; rentry[9].ip = 0xFFF2EF20; rentry[9].pointer = 15;
	rentry[10].prefix = 24; rentry[10].ip = 0x0026D177; rentry[10].pointer = 21;
	rentry[11].prefix = 26; rentry[11].ip = 0x004D7BDE; rentry[11].pointer = 22;
	rentry[12].prefix = 28; rentry[12].ip = 0x00742645; rentry[12].pointer = 23;
	rentry[13].prefix = 30; rentry[13].ip = 0xFFD7CE04; rentry[13].pointer = 24;
	rentry[14].prefix = 32; rentry[14].ip = 0xFFFDD0A6; rentry[14].pointer = 25;
	rentry[15].prefix = 0;
	return;
}

// DIR-24-8-BASIC 알고리즘의 TBL 테이블 생성 함수.
void create_table(){
	int a, b, c, d;
	unsigned int i, flag, prefix, temp, indextemp, hopbit, hoptemp;
	unsigned int n=0, index_count=0;
	unsigned char *t;

	while(rentry[n].prefix != 0 && n < SAVANNA_ROUTE_NUM){
		prefix = rentry[n].prefix;
		hopbit = rentry[n].pointer;
		t = (unsigned char *)(&(rentry[n].ip));
		a = (int)t[3];
		b = (int)t[2];
		c = (int)t[1];
		d = (int)t[0];

		printf("%3d.%3d.%3d.%3d, prefix %d, hopbit %d\n", a,b,c,d,prefix,hopbit); //getchar();

		flag = f3 * a;
		flag+= f2 * b;
		flag+= f1 * c;

		if(prefix <= 24){
			flag = flag & mask[prefix];
			temp = pow_t(24 - prefix) + flag;
			for(i=flag; i<temp; i++)
				h_TBL24[i] = (unsigned short)hopbit;
		}
		else{ //Prefix is bigger than 24
			if(h_TBL24[flag] == 0){ //Empty entry
				h_TBL24[flag] = 0x8000 | index_count;
				flag = d & charmask[prefix - 24];
				temp = pow_t(32 - prefix) + flag;			
				for(i=index_count*256+flag; i<index_count*256+temp; i++)
					h_TBLlong[i] = (unsigned char)hopbit;
				index_count++;
			}
			else{ //Something in entry
				if((h_TBL24[flag] & 0x8000) == 0x0000){ //If TBL24 entry
					hoptemp = h_TBL24[flag];
					h_TBL24[flag] = 0x8000 | index_count;
					for(i=index_count*256; i<index_count*256+256; i++) h_TBLlong[i] = (unsigned char)hoptemp;
					flag = d & charmask[prefix - 24];
					temp = pow_t(32 - prefix) + flag;			
					for(i=index_count*256+flag; i<index_count*256+temp; i++)
						h_TBLlong[i] = (unsigned char)hopbit;
					index_count++;
				}
				else{ //If TBLlong entry
					indextemp = h_TBL24[flag] & 0x7F;
					flag = d & charmask[prefix - 24];
					temp = pow_t(32 - prefix) + flag;			
					for(i=indextemp*256+flag; i<indextemp*256+temp; i++)
						h_TBLlong[i] = (unsigned char)hopbit;
				}
			}
		}
		n++;
	}
	return;
}

// 디버그용 테이블 출력 함수.
void print_table(){
	unsigned int i;
	FILE *fp24, *fplong;
	fp24 = fopen("TBL24.txt", "w");
	for(i=0; i<f4; i++)
		fprintf(fp24, "%u\t%X\n", h_TBL24[i], h_TBL24[i]);
	fplong = fopen("TBLlong.txt", "w");
	for(i=0; i<f3; i++)
		fprintf(fplong, "%u\t%X\n", h_TBLlong[i], h_TBLlong[i]);
	fclose(fp24);
	fclose(fplong);
}

/* CPU를 사용하는 실제 라우팅 함수.
pckt 배열에 처리할 IP주소가 쌓여 있으면, (이때는 4바이트의 a.b.c.d 형태)
우선 a, b, c, d로 쪼개어 받고, 이를 처리한 후 종료함.
결과는 h_hop에 들어 있으니 이걸 send udp로 다시 클라이언트에게 보내주면 됨.
*/ 
void route(int _count, int ipnum){
	int i = 0;
	unsigned int index;
	unsigned short TEMP;
	unsigned char *t;
	unsigned int *ip;
	unsigned int *hop;

	// input IP address data
	gettimeofday(&op_time[0], NULL);

	// transfer IP address data - CPU는 전송이 필요없음
	gettimeofday(&op_time[1], NULL);

	memset(route_pages[_count], 0, sizeof(unsigned int) * 1024 * SAVANNA_BUF_NUM / 2);
	memset(hop_pages[_count], 0, sizeof(unsigned int) * 1024 * SAVANNA_BUF_NUM / 2);

	create_sample_ip(_count, SAVANNA_IP_NUM);

	// create and start CUDA timer
	gettimeofday(&op_time[2], NULL);
	/////////////////////////////////////////////////////////////////////////////////////
	ip = (unsigned int *)route_pages[_count];
	hop = (unsigned int *)hop_pages[_count];
	for(i = 0 ; i < ipnum ; i++){
		t = (unsigned char *)&(ip[i]);
		index = t[3] * 65536 + t[2] * 256 + t[1];
		TEMP = h_TBL24[index];
		if((TEMP & 0x8000) == 0x0000)
			hop[i] = TEMP;
		else
			hop[i] = h_TBLlong[(TEMP & 0x7FFF)*256 + t[0]];
	}
	/////////////////////////////////////////////////////////////////////////////////////

	// copy result from device to host - CPU는 전송이 필요없음
	gettimeofday(&op_time[3], NULL);

	gettimeofday(&op_time[4], NULL); 

	//printf("----------- result ---------\n");
	//printf("memory allcation    : %d us\n", getusec(op_time[0], op_time[1]));
	//printf("memory              : %d us\n", getusec(op_time[1], op_time[2]));
	printf("cpu run time        : %5d us ( route page : %d )\n", getusec(op_time[2], op_time[3]), _count);
	//printf("memory deallocation : %d us\n", getusec(op_time[3], op_time[4]));
	//printf("---> total : %d us\n", getusec(op_time[0], op_time[4]));
	//printf("----------------------------\n");
	return;
}

int create_sample_ip(int _count, int ipnum) 
{
	unsigned int temp;
	unsigned long seed;
	unsigned long hop;
	unsigned char *t;
	unsigned int *ip;
	int count;

	t = (unsigned char *)&temp;
	hop = IP_GEN_POOL / ipnum;
	seed = 10000;

	//printf("Creating IP data %d... seed = %lu, hop = %lu(%lx)\n", ipnum, seed, hop, hop);
	ip = (unsigned int *)route_pages[_count];
	for(count = 0 ; count < ipnum ; count++){
		t[3] = (unsigned char)(seed/16777216);
		t[2] = (unsigned char)((seed/65536) % 256);
		t[1] = (unsigned char)((seed/256) % 256);
		t[0] = (unsigned char)(seed%256);
		ip[count] = temp;
		seed+= hop;
	}
	return 0;
}

// 디버그용 데이터 출력 - 파일.
void print_output_file(int _count, unsigned int ipnum)
{
	int count;
	unsigned char *t;
	unsigned int *ip;
	unsigned int *hop;

	FILE *fout = fopen("output_pv.txt", "w");
	ip = (unsigned int *)route_pages[_count];
	hop = (unsigned int *)hop_pages[_count];
	for(count = 0 ; count < ipnum; count++){
		t = (unsigned char *)&(ip[count]);
		fprintf(fout, "%3d   %3d.%3d.%3d.%3d - ", count, (int)t[3], (int)t[2], (int)t[1], (int)t[0]);
		fprintf(fout, "%d\n", hop[count]);
	}
	fclose(fout);
	printf("see output_pv.txt for output\n");
}

// 디버그용 데이터 출력 - 화면.
void print_output_screen(int _count, unsigned int ipnum) 
{
	int count;
	unsigned char *t;
	unsigned int *ip;
	unsigned int *hop;

	ip = (unsigned int *)route_pages[_count];
	hop = (unsigned int *)hop_pages[_count];
	for(count = 0 ; count < ipnum; count++){
		t = (unsigned char *)&(ip[count]);
		printf("%3d   %3d.%3d.%3d.%3d - ", count, (int)t[3], (int)t[2], (int)t[1], (int)t[0]);
		printf("%d\n", hop[count]);
	}
}

int main(){
	int count;

	alloc_memory();


	printf("input rentry...\n");
	input_rentry(); // 라우팅 테이블 입력
	printf("creating table...\n");
	create_table(); // TBL테이블 생성	
	printf("printing table...\n");
	print_table(); // 디버그 출력


	for(count = 0 ; count < 100 ; count++) {
		route(count%PAGE_NUM, SAVANNA_IP_NUM);
		usleep(100);
	}	

	print_output_file(count%PAGE_NUM, SAVANNA_IP_NUM);

	free_memory();

	printf("Goodbye!!\n");
	return 0;
}