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
#include <cuda.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>



#define SAVANNA_IP_NUM		10000
#define SAVANNA_BUF_NUM		20

#define SAVANNA_SIG_NUM   44 

#define SAVANNA_BUF_SIZE  4096   // 한 mmap space가 4KB임.
#define SAVANNA_ROUTE_NUM 64     // 라우팅 테이블의 규칙이 최대 몇 개인가를 나타냄. 필요하면 늘리면 됨. 별로 의미는 없음.

#define IOCTL_SETPID 13
#define IOCTL_MMAP 31
#define IOCTL_GPUCOMP 41

#define IOCTL_RESP_DATA 21
#define IOCTL_SIGTEST_START 23
#define IOCTL_SIGTEST_END 25

#define SIG_SEND_DATA 27
#define SIG_SIGTEST_START 29
#define SIG_CLOSE 0

#define FALSE 0
#define TRUE 1

#define INSERT_RT 10
#define DELETE_RT 11

#define MAX_RT_COUNT 28


struct timeval op_time[24];
struct timeval total_time[6];
struct timeval gpu_launch_time[6];

typedef struct 
{

	unsigned int 	dest_addr;
	int 			 prefix;
	int 		 	cmd;
	unsigned int 	nexthop;
	
}_rt_info;


_rt_info g_rt_info[SAVANNA_ROUTE_NUM];



#define MAX_NETDEV_NAME 11

typedef struct {

	char	dest_addr[MAX_NETDEV_NAME];
	int		gateway;
	int		prefix;
	char 	cmd[MAX_NETDEV_NAME];
	int		nexthop;
	
}routing_table_entry;





int create_sample_ip_to_testMemory(int ipnum);

// DIR-24-8-BASIC 알고리즘 구현 중 TBL 테이블 생성에 사용됨
#define f1 0x1
#define f2 0x100
#define f3 0x10000
#define f4 0x1000000

#define IP_GEN_POOL 0xFFFFFFFF

// 쿠다 함수를 안전하게 사용하는 매크로. 그냥 디버그용이라고 보면 됨
#define checkCudaErrors(err) __ (err, __FILE__, __LINE__)
inline void __( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

// 쿠다 함수를 안전하게 사용하는 매크로. 디버그용이라고 보면 됨. 주로 커널 실행 후 호출
#define getLastCudaError(msg) __getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

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

// 라우팅 테이블 규칙을 입력받는 포맷. 어떤 ip의 prefix 몇짜리가 오면 pointer로 보낸다는 뜻
struct route_entry{
	unsigned char prefix;
	unsigned int ip;
	unsigned int pointer;
};

// GPU memory var
// a, b, c, d는 차례로 ip주소 a.b.c.d 에 대응됨
// hop은 결과 배열 즉 어디로 나갈지가 들어가는 배열
// TBL24, TBLlong은 DIR-24-8-BASIC 알고리즘에 사용되는 배열
unsigned int *h_hop, *d_hop;
unsigned short *h_TBL24, *d_TBL24;
unsigned char *h_TBLlong, *d_TBLlong;
unsigned char *h_sm, *d_sm;


int done_update_work ;


// TEST용 Sample ip 저장
unsigned int *sample_IP;

// 라우팅 규칙이 들어갈 배열
struct route_entry rentry[SAVANNA_ROUTE_NUM];

// DIR-24-8-BASIC 알고리즘이 실제로 수행되는 부분
// blockIdx.x 는 각 쓰레드가 속한 블록 번호
// 마찬가지로 threadIdx.x 및 threadIdx.y 는 각 쓰레드의 x, y좌표
__global__ void routing(unsigned char* _sm, unsigned int* HOP, unsigned short* TBL24_, unsigned char* TBLlong_){
	int I = blockIdx.y * 10240 + blockIdx.x  * 1024 + threadIdx.y * 32 + threadIdx.x * 1;

	unsigned int index = _sm[I*4+3] * 65536 + _sm[I*4+2] * 256 + _sm[I*4+1];

	unsigned short TEMP = TBL24_[index];

	if((TEMP & 0x8000) == 0x0000)
		HOP[I] = TEMP;
	else
		HOP[I] = TBLlong_[(TEMP & 0x7FFF)*256 + _sm[I*4]];
}

// 거듭제곱 함수
unsigned int pow(int n){
	int i;
	unsigned int r = 1;
	for(i=0; i<n; i++)
		r*=2;
	return r;
}

// 각 모듈별 소요 시간을 계산하는 함수
unsigned int getusec(struct timeval start, struct timeval end){
	return ((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec - start.tv_usec;
}


unsigned int getusec_(struct timeval start, struct timeval end){
	return end.tv_usec - start.tv_usec;
}




// GPU 메모리 할당. 우선 장치를 초기화하고, GPU-CPU사이에 Pinned 메모리로 선언하기 때문에
// cudaMemcpy를 사용하여 명시적으로 데이터를 교환하지 않아도 됨!
// 즉 호스트와 디바이스 사이에 공유 메모리가 생겼다고 보면 됨.
// 단, 호스트 코드에서는 h_변수이름 을 사용하고, 디바이스 코드에서는 d_변수이름 을 사용할것!
// 예: CPU에서 돌아가는 메인 함수 등에서는 h_hop인 배열을 CUDA 커널 코드에서는 d_hop으로 접근
void alloc_memory(){ 
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

	checkCudaErrors(cudaHostAlloc((void**)&h_sm,		SAVANNA_BUF_NUM/2 * 4096 * sizeof(unsigned char),		cudaHostAllocMapped||cudaHostAllocWriteCombined));
	checkCudaErrors(cudaHostAlloc((void**)&h_hop,		SAVANNA_BUF_NUM/2 * 1024 * sizeof(unsigned int),		cudaHostAllocMapped||cudaHostAllocWriteCombined));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&d_sm,			(void*)h_sm, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void**)&d_hop,		(void*)h_hop, 0));

	checkCudaErrors(cudaMalloc((void**)&d_TBL24,		f4 * sizeof(unsigned short)));
	checkCudaErrors(cudaMalloc((void**)&d_TBLlong,		f3 * sizeof(unsigned char)));
	h_TBL24 = (unsigned short*)malloc(f4 * sizeof(unsigned short));
	h_TBLlong = (unsigned char*)malloc(f3 * sizeof(unsigned char));

	sample_IP = (unsigned int *)malloc(sizeof(unsigned int) * SAVANNA_BUF_NUM * 1024);
	return;
}


// 종료시 해제
void free_memory(){
	checkCudaErrors(cudaFreeHost(h_sm));
	checkCudaErrors(cudaFreeHost(h_hop));

	checkCudaErrors(cudaFree(d_TBL24));
	checkCudaErrors(cudaFree(d_TBLlong));

	checkCudaErrors(cudaThreadExit());
	free(h_TBL24);
	free(h_TBLlong);

	free(sample_IP);
	return;
}



// 시험용 라우팅 테이블 엔트리.
void create_rentry()
{	
	g_rt_info[0].prefix = 8; g_rt_info[0].dest_addr = 0x6601a8c0; g_rt_info[0].nexthop = 2; g_rt_info[0].cmd = INSERT_RT;
	g_rt_info[1].prefix = 16; g_rt_info[1].dest_addr = 0x6601a8c0; g_rt_info[1].nexthop = 4; g_rt_info[1].cmd = INSERT_RT;
	g_rt_info[2].prefix = 24; g_rt_info[2].dest_addr = 0x6601a8c0; g_rt_info[2].nexthop = 6; g_rt_info[2].cmd = INSERT_RT;
	g_rt_info[3].prefix = 32; g_rt_info[3].dest_addr = 0x6601a8c0; g_rt_info[3].nexthop = 8; g_rt_info[3].cmd = INSERT_RT;
	g_rt_info[4].prefix = 8; g_rt_info[4].dest_addr = 0x6601a8c0; g_rt_info[4].nexthop = 8; g_rt_info[4].cmd = INSERT_RT;
	g_rt_info[5].prefix = 16; g_rt_info[5].dest_addr = 0x6601a8c0; g_rt_info[5].nexthop = 6; g_rt_info[5].cmd = INSERT_RT;
	g_rt_info[6].prefix = 24; g_rt_info[6].dest_addr = 0x6601a8c0; g_rt_info[6].nexthop = 4; g_rt_info[6].cmd = INSERT_RT;
	g_rt_info[7].prefix = 32; g_rt_info[7].dest_addr = 0x6601a8c0; g_rt_info[7].nexthop = 2; g_rt_info[7].cmd = INSERT_RT;
	g_rt_info[8].prefix = 8; g_rt_info[8].dest_addr = 0x6601a8c0; g_rt_info[8].nexthop = 8; g_rt_info[8].cmd = DELETE_RT;
	g_rt_info[9].prefix = 16; g_rt_info[9].dest_addr = 0x6601a8c0; g_rt_info[9].nexthop = 2; g_rt_info[9].cmd = DELETE_RT;
	g_rt_info[10].prefix = 24; g_rt_info[10].dest_addr = 0x6601a8c0; g_rt_info[10].nexthop = 4; g_rt_info[10].cmd = DELETE_RT;
	g_rt_info[11].prefix = 32; g_rt_info[11].dest_addr = 0x6601a8c0; g_rt_info[11].nexthop = 6; g_rt_info[11].cmd = DELETE_RT;
	g_rt_info[12].prefix = 8; g_rt_info[12].dest_addr = 0xc0a80166; g_rt_info[12].nexthop = 4; g_rt_info[12].cmd = INSERT_RT;
	g_rt_info[13].prefix = 16; g_rt_info[13].dest_addr = 0xc0a80166; g_rt_info[13].nexthop = 6; g_rt_info[13].cmd = INSERT_RT;
	g_rt_info[14].prefix = 24; g_rt_info[14].dest_addr = 0xc0a80166; g_rt_info[14].nexthop = 8; g_rt_info[14].cmd = INSERT_RT;
	g_rt_info[15].prefix = 32; g_rt_info[15].dest_addr = 0xc0a80166; g_rt_info[15].nexthop = 8; g_rt_info[15].cmd = INSERT_RT;
	g_rt_info[16].prefix = 8; g_rt_info[16].dest_addr = 0xc0a80166; g_rt_info[16].nexthop = 6; g_rt_info[16].cmd = INSERT_RT;
	g_rt_info[17].prefix = 16; g_rt_info[17].dest_addr = 0xc0a80166; g_rt_info[17].nexthop = 4; g_rt_info[17].cmd = INSERT_RT;
	g_rt_info[18].prefix = 24; g_rt_info[18].dest_addr = 0xc0a80166; g_rt_info[18].nexthop = 2; g_rt_info[18].cmd = INSERT_RT;
	g_rt_info[19].prefix = 32; g_rt_info[19].dest_addr = 0xc0a80166; g_rt_info[19].nexthop = 8; g_rt_info[19].cmd = DELETE_RT;
	g_rt_info[20].prefix = 8; g_rt_info[20].dest_addr = 0xc0a80166; g_rt_info[20].nexthop = 2; g_rt_info[20].cmd = DELETE_RT;
	g_rt_info[21].prefix = 16; g_rt_info[21].dest_addr = 0xc0a80166; g_rt_info[21].nexthop = 4; g_rt_info[21].cmd = DELETE_RT;
	g_rt_info[22].prefix = 22; g_rt_info[22].dest_addr = 0xc0a80166; g_rt_info[22].nexthop = 6; g_rt_info[22].cmd = DELETE_RT;


	return;
}


__global__ void routing_TBLtable_update_with_gpu(unsigned int range ,unsigned int hop, unsigned short* TBL24_, unsigned int strval_, unsigned int endval_)
{

	int I = blockIdx.y * range + blockIdx.x  * 20 + threadIdx.y * 10 + threadIdx.x * 1;

	unsigned int index =  I + strval_; 

	if(index < endval_)
	{
		TBL24_[index]= hop; 
	}
}

__global__ void routing_TBLtable_8_update_with_gpu(unsigned int range ,unsigned int hop, unsigned short* TBL24_, unsigned int strval_, unsigned int endval_)
{

	int I = blockIdx.y * range + blockIdx.x  * 20 + threadIdx.y * 10 + threadIdx.x * 1;

	unsigned int index =  I + strval_; 

	if(index < endval_)
	{
		TBL24_[index]= hop; 
	}
}


__global__ void routing_TBLlongtable_update_with_gpu(unsigned int range ,unsigned int hop, unsigned char* TBL24long_, unsigned int strval_, unsigned int endval_)
{

	int I = blockIdx.y * range + blockIdx.x  * 20 + threadIdx.y * 10 + threadIdx.x * 1;

	unsigned int index =  I + strval_; 

	if(index < endval_)
	{
		TBL24long_[index]= hop; 
	}
}




void _savanna_ipv4_insert(int dst_addr, int _prefix, unsigned int nexthop, int rt_update) /* Created based on DIR-24-8 Algorithm */
{

	if(dst_addr == NULL)
		return ;
		
	
//	u32 dst_network_max = dst_network_base | ((u64)0x0FFFFFFFF >> prefix); 		// Netmask applied maximum addr
//	u32 tbl24_base = dst_network_base >> 8;		// Index for tbl24 - base
//	u32 tbl24_max = dst_network_max >> 8;		// Index for tbl24 - largest
//	u16 pool_cell = pfinder_ipv4_pool_set( rt_entry );	// alloc current rt_entry into memory pool, cell

	static unsigned int index_count = 0;

	unsigned int i, prefix, temp, indextemp, hopbit, hoptemp ;
	prefix = temp = indextemp = hopbit = hoptemp = i = 0;

	int ntbl_fir_start, ntbl_range, ntbl_sec_start, ntbl_sec_range, flag, rare_flag, temp_flag;
	ntbl_fir_start = ntbl_range = ntbl_sec_start = ntbl_sec_range = flag = rare_flag = temp_flag = 0;

	int d, nprefix_position;
	d = nprefix_position = 0;
	
//	int *nsep_ip = NULL;
//	nsep_ip = &dst_addr;

	//d = dst_addr & (0x000000FF);
	//flag = ntohl(dst_addr) & ~(0xFFFFFFFF >> prefix);
	/*	
	unsigned char *t;	
	t = (unsigned char *)(&dst_addr);
	a = (int)t[3];
	b = (int)t[2];
	c = (int)t[1];
	d = (int)t[0];

	flag = f3 * a;
	flag+= f2 * b;
	flag+= f1 * c;
	*/
	
	rare_flag = ntohl(dst_addr);
//	d = rare_flag & (0x000000FF);
	d = rare_flag & (0x000000FF);

	//flag = rare_flag & (~(0xFFFFFFFF >> prefix));

//	flag = (rare_flag >> (32-_prefix)) & (0x00FFFFFF >> (32 - _prefix)) ;

	flag = (rare_flag >> 8) & 0x00FFFFFF;


//	rt_info->destination = daddr & ~(0x0FFFFFFFF << prefix);


	prefix = _prefix;
	hopbit = nexthop;
	
//	printf("%x, d %d,prefix %d, hopbit %d\n", rare_flag, d, prefix,hopbit); //getchar();
//	printf("%x, d %x,prefix %d, hopbit %d\n", flag, d, prefix,hopbit); //getchar();

		if(prefix <= 24)
		{
			nprefix_position = 1;
			flag = flag & mask[prefix];
			temp = pow(24 - prefix) + flag;
	//		ntbl_fir_start = flag;
		//	ntbl_range = temp - flag;
			for(i = flag; i < temp; i++)
				h_TBL24[i] = (unsigned short)hopbit;
		}
		else{ //Prefix is bigger than 24
			if(h_TBL24[flag] == 0)
			{ //Empty entry
				nprefix_position = 2;
				h_TBL24[flag] = 0x8000 | index_count;
				flag = d & charmask[prefix - 24];
				temp = pow(32 - prefix) + flag;	
			//	ntbl_fir_start = index_count * 256 + flag;
			//	ntbl_range = (index_count * 256 + temp) - ntbl_fir_start;
				for(i = index_count * 256 + flag; i < index_count * 256 + temp ; i++)
					h_TBLlong[i] = (unsigned char)hopbit;
				
				index_count++;
			}
			else{ //Something in entry
				if((h_TBL24[flag] & 0x8000) == 0x0000) //If TBL24 entry
				{ 
					nprefix_position = 3;
					hoptemp = h_TBL24[flag];
					temp_flag = flag;
					h_TBL24[flag] = 0x8000 | index_count;
					ntbl_fir_start = index_count * 256;
					ntbl_range = 256;
					for(i = index_count * 256; i < index_count * 256 + 256; i++) 
						h_TBLlong[i] = (unsigned char)hoptemp;
					
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag;			
					ntbl_sec_start = index_count * 256 + flag;
					ntbl_sec_range = (index_count * 256  + temp) - ntbl_sec_start;
					
					for(i = index_count * 256 + flag ; i < index_count * 256 + temp; i++)
						h_TBLlong[i] = (unsigned char)hopbit;

					index_count++;
				}
				else{ //If TBLlong entry
					nprefix_position = 4;	
						
					indextemp = h_TBL24[flag] & 0x7F;
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag;	
					ntbl_fir_start = indextemp * 256 + flag;
					ntbl_range = (indextemp * 256 + temp) - ntbl_fir_start;
					for( i = indextemp * 256 + flag; i < indextemp * 256 + temp; i++)
						h_TBLlong[i] = (unsigned char)hopbit;
				}
			}
		}
	/*
	//if(!rt_update)
	{
		int nloop  = 0; 
		switch(nprefix_position)
		{
			case 1:
				gettimeofday(&rt_copy_time[0], NULL); // <-- check point
			//	for(nloop = ntbl_fir_start; nloop < ntbl_fir_end; nloop++)	
					checkCudaErrors(cudaMemcpy(&d_TBL24[cur_rt][ntbl_fir_start], &h_TBL24[cur_rt][ntbl_fir_start],  ntbl_range * sizeof(unsigned short), cudaMemcpyHostToDevice));	

				gettimeofday(&rt_copy_time[1], NULL); // <-- check point
			break;
				
			case 2:
				
			//	for(nloop = ntbl_fir_start; nloop < ntbl_fir_end; nloop++)	
					checkCudaErrors(cudaMemcpy(&d_TBLlong[cur_rt][ntbl_fir_start], &h_TBLlong[cur_rt][ntbl_fir_start], ntbl_range * sizeof(unsigned char), cudaMemcpyHostToDevice));

				
			break;
				
			case 3:
				
				checkCudaErrors(cudaMemcpy(&d_TBL24[cur_rt][temp_flag], &h_TBL24[cur_rt][temp_flag], sizeof(unsigned short), cudaMemcpyHostToDevice));	
		
			//	for(nloop = ntbl_fir_start; nloop < ntbl_fir_end; nloop++)	
					checkCudaErrors(cudaMemcpy(&d_TBLlong[cur_rt][ntbl_fir_start], &h_TBLlong[cur_rt][ntbl_fir_start], ntbl_range * sizeof(unsigned char), cudaMemcpyHostToDevice));

			//	for(nloop = ntbl_sec_start; nloop < ntbl_sec_end; nloop++)	
					checkCudaErrors(cudaMemcpy(&d_TBLlong[cur_rt][ntbl_sec_start], &h_TBLlong[cur_rt][ntbl_sec_start], ntbl_sec_range* sizeof(unsigned char), cudaMemcpyHostToDevice));
			break;
			
			case 4:		
				gettimeofday(&rt_copy_time[2], NULL); // <-- check point
			//	for(nloop = ntbl_fir_start; nloop < ntbl_fir_end; nloop++)	
					checkCudaErrors(cudaMemcpy(&d_TBLlong[cur_rt][ntbl_fir_start], &h_TBLlong[cur_rt][ntbl_fir_start], ntbl_range * sizeof(unsigned char), cudaMemcpyHostToDevice));		

				gettimeofday(&rt_copy_time[3], NULL); // <-- check point
			break;
		}
		
	}
	
	//else{
		
	//	checkCudaErrors(cudaMemcpy(d_TBL24[cur_rt], h_TBL24[cur_rt], f4 * sizeof(unsigned short), cudaMemcpyHostToDevice));
	//	checkCudaErrors(cudaMemcpy(d_TBLlong[cur_rt], h_TBLlong[cur_rt], f3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

	//	printf(" d_TBL24 size = %d, cur_rt = %d\n", d_TBL24, cur_rt);
	//	printf(" d_TBLlong size = %d, cur_rt = %d\n", d_TBLlong, cur_rt);
	
//	}

	
	*/
	
	return;
}



/*
void _savanna_ipv4_insert(unsigned int dst_addr, int _prefix, unsigned int nexthop, int rt_update) 
{

	if(dst_addr == NULL)
		return ;
		
	
//	u32 dst_network_max = dst_network_base | ((u64)0x0FFFFFFFF >> prefix); 		// Netmask applied maximum addr
//	u32 tbl24_base = dst_network_base >> 8;		// Index for tbl24 - base
//	u32 tbl24_max = dst_network_max >> 8;		// Index for tbl24 - largest
//	u16 pool_cell = pfinder_ipv4_pool_set( rt_entry );	// alloc current rt_entry into memory pool, cell

	static unsigned int index_count = 0;

	unsigned int i, prefix, temp, indextemp, hopbit, hoptemp ;
	prefix = temp = indextemp = hopbit = hoptemp = i = 0;

	unsigned int ntbl_fir_start, ntbl_range, ntbl_sec_start, ntbl_sec_range, flag, rare_flag, temp_flag, pre32__pos2_flag, pre32_pos2_val, pre32_pos3_val;
	ntbl_fir_start = ntbl_range = ntbl_sec_start = ntbl_sec_range = flag = rare_flag = temp_flag = pre32__pos2_flag = pre32_pos2_val = pre32_pos3_val = 0;

	int d, nprefix_position;
	d = nprefix_position = 0;


	unsigned int blc_x, blc_y, th_x, th_y, thread_range;
	blc_x = blc_y = th_x = th_y = thread_range = 0;


//	int *nsep_ip = NULL;
//	nsep_ip = &dst_addr;

	//d = dst_addr & (0x000000FF);
	//flag = ntohl(dst_addr) & ~(0xFFFFFFFF >> prefix);
	
	unsigned char *t;	
	t = (unsigned char *)(&dst_addr);
	a = (int)t[3];
	b = (int)t[2];
	c = (int)t[1];
	d = (int)t[0];

	flag = f3 * a;
	flag+= f2 * b;
	flag+= f1 * c;
	


//	unsigned int rare_flag, flag, temp, thread_range, prefix, hopbit;
//	flag = temp = thread_range =  hopbit =  prefix= 0;



//	gettimeofday(&op_time[6], NULL);
	rare_flag = ntohl(dst_addr);
	d = rare_flag & (0x000000FF);

	//flag = rare_flag & (~(0xFFFFFFFF >> prefix));

	flag = (rare_flag >> 8) & 0x00FFFFFF;
//		gettimeofday(&op_time[7], NULL);
//	flag = (rare_flag >> 8) & 0x00FFFFFF;

//	rt_info->destination = daddr & ~(0x0FFFFFFFF << prefix);


	prefix = _prefix;
	hopbit = nexthop;
	
//	printf("%x, d %d,prefix %d, hopbit %d\n", rare_flag, d, prefix,hopbit); //getchar();
//	printf("%x, d %d,prefix %d, hopbit %d\n", flag, d, prefix,hopbit); //getchar();

		if(prefix <= 24)
		{
		//	gettimeofday(&op_time[8], NULL);
			nprefix_position = 1;
			flag = flag & mask[prefix];
			temp = pow(24 - prefix) + flag;
	
		//	routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], flag, temp);
			
		//	gettimeofday(&op_time[9], NULL);

		//	checkCudaErrors(cudaThreadSynchronize());

			ntbl_fir_start = flag;
			ntbl_range = temp;
			
			for(i = flag; i < temp; i++)
				h_TBL24[i] = (unsigned short)hopbit;


		//	gettimeofday(&op_time[10], NULL);
				
	}
		
		else{ //Prefix is bigger than 24
			if(h_TBL24[flag] == 0)
			{ //Empty entry
				nprefix_position = 2;
				h_TBL24[flag] = 0x8000 | index_count;
				pre32_pos2_val = h_TBL24[flag];
				pre32__pos2_flag = flag;
				flag = d & charmask[prefix - 24];
				temp = pow(32 - prefix) + flag;	
				ntbl_fir_start = index_count * 256 + flag;
				ntbl_range = index_count * 256 + temp;
		//		for(i = index_count * 256 + flag; i < index_count * 256 + temp ; i++)
		//			h_TBLlong[cur_rt][i] = (unsigned char)hopbit;
				
				index_count++;
			}
			else{ //Something in entry
				if((h_TBL24[flag] & 0x8000) == 0x0000) //If TBL24 entry
				{ 
					nprefix_position = 3;
					hoptemp = h_TBL24[flag];
					temp_flag = flag;
					h_TBL24[flag] = 0x8000 | index_count;
					pre32_pos3_val = h_TBL24[flag];
					ntbl_fir_start = index_count * 256;
					ntbl_range = index_count * 256 + 256;
					for(i = index_count * 256; i < index_count * 256 + 256; i++) 
						h_TBLlong[i] = (unsigned char)hoptemp;
					
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag;			
					ntbl_sec_start = index_count * 256 + flag;
					ntbl_sec_range = index_count * 256  + temp;
					
			//		for(i = index_count * 256 + flag ; i < index_count * 256 + temp; i++)
			//			h_TBLlong[cur_rt][i] = (unsigned char)hopbit;

					index_count++;
				}
				else{ //If TBLlong entry
					nprefix_position = 4;	
						
					indextemp = h_TBL24[flag] & 0x7F;
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag;	
					ntbl_fir_start = indextemp * 256 + flag;
					ntbl_range = indextemp * 256 + temp;
			//		for( i = indextemp * 256 + flag; i < indextemp * 256 + temp; i++)
			//			h_TBLlong[cur_rt][i] = (unsigned char)hopbit;
				}
			}
		}
		
		
		
		switch(prefix)
		{
			
			case 8:
				blc_x = 8, blc_y = 8;
				th_x = 32, th_y = 32;		
				thread_range = 8100;
			break;
			
			case 16:case 24:case 32:
				blc_x = 3, blc_y = 4;
				th_x = 6, th_y = 4;
				thread_range = 100;
			break;

		}

		dim3 rt_blocks(blc_x, blc_y);
		dim3 rt_threads(th_x, th_y);
		
		

	if(rt_update > 0)
	{
		switch(nprefix_position)
		{
			case 1:
			//	gettimeofday(&rt_copy_time[0], NULL); // <-- check point
		
		//		printf("positon 1:\n");
				routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24, ntbl_fir_start, ntbl_range);

			//	checkCudaErrors(cudaThreadSynchronize());
		//		gettimeofday(&rt_copy_time[1], NULL); // <-- check point
				
			break;
				
			case 2:
			//	printf("positon 2:\n");
			
				routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, pre32_pos2_val, d_TBL24, pre32__pos2_flag, pre32__pos2_flag + 1 );

				routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBLlong, ntbl_fir_start, ntbl_range);

			//	checkCudaErrors(cudaThreadSynchronize());
				
			break;
				
			case 3:
			//	printf("positon 3:\n");
				
			//	checkCudaErrors(cudaMemcpy(&d_TBL24[cur_rt][temp_flag], &h_TBL24[cur_rt][temp_flag], sizeof(unsigned short), cudaMemcpyHostToDevice));	
		//		printf("thread_range %d, pre32_pos3_val %d, temp_flag %d, temp_flag +1 %d \n", thread_range, pre32_pos3_val, temp_flag, temp_flag +1);
				
		//		printf("thread_range %d, hoptemp %d, ntbl_fir_start %d, ntbl_range %d \n", thread_range, hoptemp, ntbl_fir_start, ntbl_range);

		//		printf("thread_range %d, hopbit %d, ntbl_sec_start %d, ntbl_sec_range %d \n", thread_range, hopbit, ntbl_sec_start, ntbl_sec_range);
			
				routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, pre32_pos3_val, d_TBL24, temp_flag, temp_flag +1 );
		
				routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hoptemp, d_TBLlong, ntbl_fir_start, ntbl_range);

				routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBLlong, ntbl_sec_start, ntbl_sec_range);
			
			//	checkCudaErrors(cudaThreadSynchronize());
			break;
			
			case 4:		
		//		gettimeofday(&rt_copy_time[2], NULL); // <-- check point
			//	printf("positon 4:\n");

				routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hoptemp, d_TBL24, temp_flag, temp_flag + 1);
	
				routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBLlong, ntbl_fir_start, ntbl_range);

		//		checkCudaErrors(cudaThreadSynchronize());
		//		gettimeofday(&rt_copy_time[3], NULL); // <-- check point
			break;
		}
		
		
	}
		
		else{
			
			checkCudaErrors(cudaMemcpy(d_TBL24, h_TBL24, f4 * sizeof(unsigned short), cudaMemcpyHostToDevice));
	 		checkCudaErrors(cudaMemcpy(d_TBLlong, h_TBLlong, f3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	
		//	printf(" d_TBL24 size = %d, cur_rt = %d\n", d_TBL24, cur_rt);
		//	printf(" d_TBLlong size = %d, cur_rt = %d\n", d_TBLlong, cur_rt);
		
		}
	
	return;
}


*/



int savanna_ipv4_update(int rt_num)
{


	int rt_count = 0;
//	printf("savanna_ipv4_update \n");
	static int rt_update = 0;
	//unsigned int SAVANNA_BUF_NUM_HALF = SAVANNA_BUF_NUM /2;
	rt_count = rt_num;
//	gettimeofday(&op_time[0], NULL); // <-- check point
	
	//for(i = 0; i < SAVANNA_BUF_NUM_HALF ; i++)
//	memcpy((_rt_info *)rt_info, savanna_sm_list->ptr[0], sizeof(_rt_info));
	
//	savanna_Print_RT_info(rt_info);
	
//	int cur_rt = savanna_searchRT(pv_id);
	
//	gettimeofday(&op_time[1], NULL); // <-- check point

	//memset(h_hop, 0, SAVANNA_BUF_NUM/2 * 1024 * sizeof(unsigned int));

//	gettimeofday(&op_time[2], NULL); // <-- check point
	
//	gettimeofday(&op_time[5], NULL); // <-- check point

	int cmd = g_rt_info[rt_count].cmd;
	
	switch( cmd ){
		case INSERT_RT: 
		//	printf(  "Update Insertion Message!\n" );
			
			_savanna_ipv4_insert(g_rt_info[rt_count].dest_addr, g_rt_info[rt_count].prefix, g_rt_info[rt_count].nexthop, rt_update); /* Created based on DIR-24-8 Algorithm */
			
		//		savanna_print_mapping_pv_with_rt();
			
			
		//	pfinder_ipv4_insert( rt_info->destination, rt_info->prefix,  rt_info );
		
		break;
		case DELETE_RT:
		{
			
/*
	         1) pf & sa에서 해당하는 rt_entry는 삭제한다. 
		  2) routing_table에서 해당하는 rt_entry자리에 parent의 next_hop을 넣으면 된다. 

*/		 
			//struct pfinder_entry_ipv4* comp_entry;
	//		printf( "Update Deletion Message!\n" );
			
			_savanna_ipv4_insert(g_rt_info[rt_count].dest_addr, g_rt_info[rt_count].prefix, g_rt_info[rt_count].nexthop, rt_update); /* Created based on DIR-24-8 Algorithm */
					
		}
		break;
	}
	//rt_count ++;
	if(!rt_update)
		rt_update ++; 
	done_update_work = 1;
	return 0;
}


/*
void savanna_ipv4_pool_init( u8 pool ){
	pfinder_dir_pool[pool] = kcalloc( sizeof( struct pfinder_pool_entry ), DIR24_CELL_LENGTH, GFP_ATOMIC );
	active_pool = pool;
	active_cell = 0;
}

u16 savanna_ipv4_pool_set( struct pfinder_entry_ipv4* entry ){
	if( active_cell >= DIR24_CELL_LENGTH ){
		pfinder_ipv4_pool_init( active_pool +1 );
	}

	pfinder_dir_pool[ active_pool ][ active_cell ].pfinder_entry= entry;
	
	return ((active_pool & 0x07) << 12) | (active_cell++ & 0x0FFF);
}

*/





// 디버그용 테이블 출력 함수.
void print_table()
{
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

// 디버그용 데이터 출력 - 파일.
void print_output_file()
{
	int count;
	unsigned int *ip;
	unsigned char *t;
	FILE *fout = fopen("output_hvm.txt", "w");
	for(count = 0 ; count < SAVANNA_IP_NUM; count++){
		t = (unsigned char *)&(sample_IP[count]);
		fprintf(fout, "%5d %3d.%3d.%3d.%3d -> ", count, (int)t[3], (int)t[2], (int)t[1], (int)t[0]);
		ip = (unsigned int *)&(sample_IP[SAVANNA_IP_NUM]);
		fprintf(fout, "%d  hop : %2d\n", ip[count], h_hop[count]);
	}
	fclose(fout);
	printf("see output_hvm.txt for output\n");
}

void print_rt_output_file()
{
	int count;
	unsigned int *ip;
	unsigned char *t;
	
	int fir_time, sec_time;
	FILE *fout = fopen("output_hvm.txt", "w");
	int rt_count = 0 ;
	for(fir_time = 0 ; fir_time < MAX_RT_COUNT; fir_time = fir_time + 2){

		sec_time = fir_time;
		sec_time ++; 
		
		fprintf(fout, "count %d ,routing table udpate time   : %6d us ( %d - %d )\n ", count ,getusec_(op_time[fir_time], op_time[sec_time]), op_time[sec_time].tv_usec, op_time[fir_time].tv_usec);
		fprintf(fout, "rt_info : prefix  %d  hop : %2d cmd =%s \n", g_rt_info[count].prefix, g_rt_info[count].nexthop, g_rt_info[count].cmd);

		fprintf(fout, "\n");	
		rt_count  ++ ;
	}
	fclose(fout);
	printf("see output_hvm.txt for output\n");
}







void print_routing_result(int count ) 
{
	int fir_time, sec_time;
	//for (fir_time = 0; fir_time < MAX_RT_COUNT ; fir_time = fir_time + 2)
	{

	//	sec_time = fir_time;
	//	sec_time ++; 


		printf("------------- result -----------\n");
		printf("routing table udpate time       : %6d us ( %d - %d ) \n", getusec(op_time[0], op_time[1]), op_time[1].tv_usec, op_time[0].tv_usec);
		printf("--------------------------------\n");

		
	//printk(KERN_DEBUG "routing table update test : time = %7ld ns ( shared memory id : %5d ), Datainput time = %7ld ns", gntpg_list->gpuLaunch[1].tv_nsec-gntpg_list->gpuLaunch[0].tv_nsec, gntpg_list->listId, gntpg_list->datainput[1].tv_nsec-gntpg_list->datainput[0].tv_nsec);
	

	}
}


int create_sample_ip_to_testMemory(int ipnum) 
{

	unsigned int temp;
    unsigned long seed;
    unsigned long hop;
    unsigned char *t;
    int count;

	t = (unsigned char *)&temp;
    hop = IP_GEN_POOL / ipnum;
	seed = 10000;

	sample_IP[0] = 0x6601a8c0;
	sample_IP[1] = 0xc0a80166;
	
	for(count = 2 ; count < ipnum ; count++){
		t[3] = (unsigned char)(seed/16777216);
		t[2] = (unsigned char)((seed/65536) % 256);
		t[1] = (unsigned char)((seed/256) % 256);
		t[0] = (unsigned char)(seed%256);
		sample_IP[count] = temp;
		seed+= hop;

	}
    return 0;
}



int main()
{
	
	alloc_memory(); // 할당
	create_sample_ip_to_testMemory(SAVANNA_IP_NUM);
	printf("allocating memory...\n");
	
	create_rentry(); // 라우팅 테이블 입력
	printf("input rentry...\n");

	routing_table_entry rt_entry; 


	
	int count = 0;

	
//	int fir_time, sec_time;
//	FILE *fout = fopen("output_hvm.txt", "w");
//	int rt_count = 0 ;
//	for(fir_time = 0 ; fir_time < MAX_RT_COUNT; fir_time = fir_time + 2){

	//	sec_time = fir_time;
	//	sec_time ++; 
		
	//	rt_count  ++ ;
//	}
	



	
	int continue_ = 1;
	int rt_cmd = 0;
	while(count < 20)
	{

//		system("echo 3 > /proc/sys/vm/drop_caches");
//		system("echo 3 > /proc/sys/vm/drop_caches");

	//	printf("whether input more rt_info : ");
	//	scanf("%d", &continue_);

		if(continue_  == -1)
			break;
	
//	strcpy(ctestip, "192.168.1.102");
//	printk("ctestip %s\n", ctestip);
	
//	ntestip = in_aton(ctestip);
//	nlittle_edian_ip = ntohl(ntestip);
//	printk("ctestip %d, %x\n", ntestip, ntestip);
//	printk("nlittle_edian_ip %d, %x\n", nlittle_edian_ip, nlittle_edian_ip);

		gettimeofday(&op_time[0], NULL); // <- check point
		savanna_ipv4_update(count);
		gettimeofday(&op_time[1], NULL); // <- check point

/*
	int fir_time, sec_time;
	for (fir_time = 0; fir_time < MAX_RT_COUNT ; fir_time = fir_time + 2)
	{
		sec_time = fir_time;
		sec_time ++; 
		
		system("echo 3 > /proc/sys/vm/drop_caches");
		system("echo 3 > /proc/sys/vm/drop_caches");
				
		done_update_work  = 0;
		
		gettimeofday(&op_time[fir_time], NULL); // <- check point
		savanna_ipv4_update();
		gettimeofday(&op_time[sec_time], NULL); // <- check point

	//	while(!done_update_work)
	//		usleep(1);
			
//	create_table(); // TBL테이블 생성

	}
*/

	//	fprintf(fout, "count %d ,routing table udpate time   : %6d us ( %d - %d )\n ", count ,getusec_(op_time[0], op_time[1]), op_time[0].tv_usec, op_time[1].tv_usec);
	//	fprintf(fout, "rt_info : prefix  %d  hop : %2d cmd =%s \n", g_rt_info[count].prefix, g_rt_info[count].nexthop, g_rt_info[count].cmd);

	//	fprintf(fout, "\n");	
		count ++;


	print_routing_result();
//	print_rt_output_file();

	}
	
//	printf("creating table...\n");
	fclose(fout);
	printf("see output_hvm.txt for output\n");
//----------------------------------------------------------------------------------------------
	/*
	memcpy((void *)h_sm, sample_IP, sizeof(unsigned int) * SAVANNA_BUF_NUM/2 * 1024);
	
	dim3 blocks(10, 1);
	dim3 threads(32, 32);

	routing<<< blocks, threads >>>(d_sm, d_hop, d_TBL24, d_TBLlong);
	getLastCudaError("Routing kernel execution failed");

	*/
	
//	checkCudaErrors(cudaThreadSynchronize());

//	gettimeofday(&op_time[2], NULL); // <- check point
//	memcpy((void *)&(sample_IP[SAVANNA_IP_NUM]), (void *)h_hop, sizeof(unsigned int) * SAVANNA_BUF_NUM/2 * 1024);
//	gettimeofday(&op_time[3], NULL); // <- check point

//	print_routing_result(); 


//	print_rt_output_file();

//	print_rt_output_file()
//----------------------------------------------------------------------------------------------
	free_memory();
	printf("deallocating memory...\n");
	printf("Goodbye!!\n");
	return 0;

}
