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
#include <arpa/inet.h>

//intoosh
//#include <linux/list.h>  // app에서 linked-list를 사용하기 위해서 추가 
//

#define DEVICE_FILENAME  "/dev/savanna"

#define SAVANNA_IP_NUM 100000
#define SAVANNA_SIG_NUM   44 
#define SAVANNA_SIG_RT_NUM 45
#define SAVANNA_BUF_NUM   200    // 커널과 유저 사이의 mmap space의 개수.
#define SAVANNA_BUF_SIZE  4096   // 한 mmap space가 4KB임.
#define SAVANNA_ROUTE_NUM 64     // 라우팅 테이블의 규칙이 최대 몇 개인가를 나타냄. 필요하면 늘리면 됨. 별로 의미는 없음.

#define IOCTL_SETPID 13
#define IOCTL_SETPID_RT 14

#define IOCTL_MMAP 31
#define IOCTL_GPUCOMP 41
#define IOCTL_RTCOMP 43
#define IOCTL_IPLOOKUP_STATE 45
#define IOCTL_RTUPDATE_STATE 47




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

#define GET_PVID_SHIFT  16
#define MASK_SMLIST 0xFFFF 


struct mem_ioctl_data {
	unsigned long requested_size;
	unsigned long start_of_vma;
	unsigned long return_offset;
	unsigned int sm_listId;
};

typedef struct savanna_sm_list_t {
	unsigned int domid; 
	void *mmaprv[SAVANNA_BUF_NUM];
	void *ptr[SAVANNA_BUF_NUM];
	unsigned int listId;
	struct savanna_sm_list_t *next;
	struct savanna_sm_list_t *pre;
} savanna_sm_list_t;



int fd;
int state;
struct timeval op_time[20];
struct timeval total_time[6];
struct timeval gpu_launch_time[6];
struct timeval rt_copy_time[10];


float elapsed_time_ms=0.0f;
cudaEvent_t start, stop;

//intoosh

#define __be32  int


// GPU memory var
// a, b, c, d는 차례로 ip주소 a.b.c.d 에 대응됨
// hop은 결과 배열 즉 어디로 나갈지가 들어가는 배열
// TBL24, TBLlong은 DIR-24-8-BASIC 알고리즘에 사용되는 배열

#define MAX_PV_COUNT 5

unsigned int *h_hop, *d_hop;
unsigned short *h_TBL24[MAX_PV_COUNT] , *d_TBL24[MAX_PV_COUNT]; 
unsigned char *h_TBLlong[MAX_PV_COUNT] , *d_TBLlong[MAX_PV_COUNT]; 
unsigned char *h_sm, *d_sm;
unsigned int *h_strval, *d_strval;
unsigned int *h_TBL, *d_TBLbar;
unsigned int *h_endval, *d_endval;
unsigned int *g_temp_flag = NULL;
unsigned short *h_TBL24_flag ;
		


int run_iplookup ;

typedef struct 
{
	int 		 pv_id;
	__be32 		 dest_addr;
	int 		 prefix;
	int 		 cmd;
	unsigned int nexthop;
	
}_rt_info;

_rt_info *g_rt_info;


typedef struct 
{
	int nindex;
	int npv_id;
}_mapping_pv_with_rt;


_mapping_pv_with_rt *mapping_pv_with_rt[MAX_PV_COUNT];

static int g_currnet_pv_cout = 0;



#define MAX_RTCUDA_QUEUE 20

typedef struct 
{

	savanna_sm_list_t *_savanna_sm_list[MAX_RTCUDA_QUEUE];
	int pv_id[MAX_RTCUDA_QUEUE];
	int front ;
	int rear  ;
	
} cuda_task_queue;


cuda_task_queue *g_cuda_rtupdate_que;


savanna_sm_list_t *savanna_sm_list_head = NULL;

savanna_sm_list_t* savanna_CreateSM(int _listId, unsigned int pv_id);
void savanna_AddSM(savanna_sm_list_t **_head, savanna_sm_list_t *_sm);
savanna_sm_list_t* savanna_SearchSM(savanna_sm_list_t **_head, unsigned int _listId);
savanna_sm_list_t* savanna_DelSM(savanna_sm_list_t **_head, unsigned int _listId);
void savanna_PrintSM_info(savanna_sm_list_t **_head);
void savanna_print_mapping_pv_with_rt();
unsigned int savanna_searchRT(unsigned int pv_id);
void _savanna_ipv4_insert(unsigned int pv_id, unsigned int cur_rt, __be32 dst_addr, int _prefix, unsigned int nexthop, int rt_update);
void Add_mapping_pv_with_rt(unsigned int _nindex, unsigned int _pv_id);
void savanna_Print_RT_info(_rt_info *client_rt_info);



// 라우팅 테이블 규칙을 입력받는 포맷. 어떤 ip의 prefix 몇짜리가 오면 pointer로 보낸다는 뜻
struct route_entry{
	unsigned char prefix;
	unsigned int ip;
	unsigned int pointer;
};

// DIR-24-8-BASIC 알고리즘 구현 중 TBL 테이블 생성에 사용됨
#define f1 0x1
#define f2 0x100
#define f3 0x10000
#define f4 0x1000000

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


void init(cuda_task_queue *cuda_que)
{
	cuda_que->front = cuda_que->rear = 0;

}
int is_empty(cuda_task_queue *cuda_que)
{
	
	return(cuda_que->front == cuda_que->rear);
}

int is_full(cuda_task_queue *cuda_que)
{
	return ((cuda_que->rear + 1) % MAX_RTCUDA_QUEUE == cuda_que->front);

}

void enqueue(cuda_task_queue *cuda_que, savanna_sm_list_t *_savanna_sm_list, int _pv_id)
{
	if(is_full(cuda_que))
		printf("full_que\n");

	cuda_que->rear = (cuda_que->rear + 1) % MAX_RTCUDA_QUEUE;
	cuda_que->_savanna_sm_list[cuda_que->rear] = _savanna_sm_list;
	cuda_que->pv_id[cuda_que->rear] = _pv_id;
}

int dequeue(cuda_task_queue *cuda_que)
{
	if(is_empty(cuda_que))
		return 0;

	cuda_que->front = (cuda_que->front + 1)% MAX_RTCUDA_QUEUE;
	return cuda_que->front;
}



int savanna_ipv4_update(savanna_sm_list_t *_savanna_sm_list, int pv_id)
{

//	printf("savanna_ipv4_update \n");

	int i = 0;
	unsigned short *temp_TBL24 = NULL;
	savanna_sm_list_t *savanna_sm_list = _savanna_sm_list;
	
	//unsigned int SAVANNA_BUF_NUM_HALF = SAVANNA_BUF_NUM /2;

//	gettimeofday(&op_time[0], NULL); // <-- check point

	//for(i = 0; i < SAVANNA_BUF_NUM_HALF ; i++)
	memcpy((_rt_info *)g_rt_info, savanna_sm_list->ptr[0], sizeof(_rt_info));
	
//	savanna_Print_RT_info(g_rt_info);

	//gettimeofday(&op_time[1], NULL); // <-- check point


	int cur_rt = savanna_searchRT(pv_id);
	

	//memset(h_hop, 0, SAVANNA_BUF_NUM/2 * 1024 * sizeof(unsigned int));

//	gettimeofday(&op_time[2], NULL); // <-- check point
	
//	gettimeofday(&op_time[5], NULL); // <-- check point

	bool update_rt = 0;
	//int cur_rt = -1;
	int cmd = g_rt_info->cmd;
	
	switch( cmd ){
		case INSERT_RT: 
		//	printf(  "Update Insertion Message!\n" );

		//	gettimeofday(&op_time[3], NULL); // <-- check point
			if(cur_rt == -1)
			{
				update_rt = 1;
				cur_rt = g_currnet_pv_cout;
			}
			
			_savanna_ipv4_insert(g_rt_info->pv_id, cur_rt, g_rt_info->dest_addr, g_rt_info->prefix, g_rt_info->nexthop, update_rt); /* Created based on DIR-24-8 Algorithm */
		//	gettimeofday(&op_time[4], NULL);
			if(update_rt)
			{
				Add_mapping_pv_with_rt(cur_rt, pv_id);
				g_currnet_pv_cout ++;
				update_rt = 0;
		//		savanna_print_mapping_pv_with_rt();
			}
		//	gettimeofday(&op_time[5], NULL);	
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

			if(cur_rt != -1)
			{
				
				_savanna_ipv4_insert(g_rt_info->pv_id, cur_rt, g_rt_info->dest_addr, g_rt_info->prefix, g_rt_info->nexthop, 0); /* Created based on DIR-24-8 Algorithm */
				update_rt = 0;
			}		
		}
		break;
	}

	
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


__global__ void routing_TBLtable_get_flag_with_gpu(unsigned int range ,unsigned int hop, unsigned short* TBL24_, unsigned int* flag, unsigned int strval_, unsigned int endval_)
{

	int I = blockIdx.y;

	unsigned int index =  I + strval_; 

	if(index == endval_)
	{
		*flag = TBL24_[index]; 
	}
}

__global__ void routing_TBLtable8_update_with_gpu(unsigned int range ,unsigned int hop, unsigned short* TBL24_, unsigned int strval_, unsigned int endval_)
{
	int I = blockIdx.y * 10240 + blockIdx.x  * 1024 + threadIdx.y * 32 + threadIdx.x * 1;
		
//	int I = blockIdx.y * range + blockIdx.x  * 10 + threadIdx.y * 3 + threadIdx.x * 1;
	unsigned int index =  I + strval_; 

	if(index < endval_)
	{
		TBL24_[index]= hop;	
	}
}



/*
__global__ void routing_TBLtable_update_with_gpu(unsigned int range ,unsigned int hop, unsigned short* TBL24_, unsigned int strval_, unsigned int endval_)
{
	
	int I = blockIdx.y * range + blockIdx.x  * 10 + threadIdx.y * 3 + threadIdx.x * 1;
	unsigned int index =  I + strval_; 

	if(index < endval_)
	{
		TBL24_[index]= hop;	
	}
}

__global__ void routing_TBLtable8_update_with_gpu(unsigned int range ,unsigned int hop, unsigned short* TBL24_, unsigned int strval_, unsigned int endval_)
{
	int I = blockIdx.y * 10240 + blockIdx.x  * 1024 + threadIdx.y * 32 + threadIdx.x * 1;
		
//	int I = blockIdx.y * range + blockIdx.x  * 10 + threadIdx.y * 3 + threadIdx.x * 1;
	unsigned int index =  I + strval_; 

	if(index < endval_)
	{
		TBL24_[index]= hop;	
	}
}


__global__ void routing_TBLtable_get_flag_with_gpu(unsigned int range ,unsigned int hop, unsigned short* TBL24_, unsigned int* flag, unsigned int strval_, unsigned int endval_)
{

	int I = blockIdx.y;

	unsigned int index =  I + strval_; 

	if(index == endval_)
	{
		*flag = TBL24_[index]; 
	}
}




__global__ void routing_with_gpu(unsigned int strval_, unsigned int endval_)
{

	//int I = blockIdx.y * range + blockIdx.x  * (range/10) + threadIdx.y * ((range/100)/3) + threadIdx.x * 1;
	
	int I = 200;
	//blockIdx.y * range + blockIdx.x  * 10 + threadIdx.y * 3 + threadIdx.x * 1;
	unsigned int index =  I + strval_; 


}



__global__ void routing_TBLlongtable_update_with_gpu(unsigned int range ,unsigned int hop, unsigned char* TBL24long_, unsigned int strval_, unsigned int endval_)
{

	int I = blockIdx.y * range + blockIdx.x  * (range/10) + threadIdx.y * ((range/100)/3) + threadIdx.x * 1;

	unsigned int index =  I + strval_; 

	if(index < endval_)
	{
		TBL24long_[index]= hop;	
	}
}




__global__ void routing_table_update_case2_with_gpu(unsigned int* HOP,unsigned short* TBL24_, unsigned int *Strval_, unsigned int *Endval)
{

	int I = blockIdx.y * 500 + blockIdx.x  * 100 + threadIdx.y * 32 + threadIdx.x * 1;

	unsigned int index =  I + Strval_; 

	if(index < Endval)
	{
		TBL24_[index]= HOP;
	}
}


__global__ void routing_table_update_case3_with_gpu(unsigned int* HOP,unsigned short* TBL24_, unsigned int *Strval_, unsigned int *Endval)
{

	int I = blockIdx.y * 100 + blockIdx.x  * 10 + threadIdx.y * 4 + threadIdx.x * 1;

	unsigned int index =  I + Strval_; 

	if(index < Endval)
	{
		TBL24_[index]= HOP;
		
	}

}
*/







// 라우팅 규칙이 들어갈 배열
struct route_entry rentry[SAVANNA_ROUTE_NUM];

// DIR-24-8-BASIC 알고리즘이 실제로 수행되는 부분
// blockIdx.x 는 각 쓰레드가 속한 블록 번호
// 마찬가지로 threadIdx.x 및 threadIdx.y 는 각 쓰레드의 x, y좌표
__global__ void routing(unsigned char* _sm, unsigned int* HOP, unsigned short* TBL24_, unsigned char* TBLlong_){
	int I = blockIdx.y * 10240 + blockIdx.x  * 1024 + threadIdx.y * 32 + threadIdx.x * 1;
			
	//unsigned char *sm = (unsigned char *)&(_sm[I]);

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



	
	
	g_temp_flag = (unsigned int *)malloc(sizeof(unsigned int));
	memset(g_temp_flag, sizeof(unsigned int), 0);
			
	g_rt_info = (_rt_info*)malloc(1 * sizeof(_rt_info));
	g_cuda_rtupdate_que = (cuda_task_queue *) malloc(sizeof(cuda_task_queue));

	h_TBL24_flag = (unsigned short*)malloc(f4 * sizeof(unsigned short));
	
	
	
	//checkCudaErrors(cudaMalloc((void**)&d_TBL24[nloop],		f4 * sizeof(unsigned short)));
	//checkCudaErrors(cudaMalloc((void**)&d_TBLlong[nloop],		f3 * sizeof(unsigned char)));

	int nloop = 0;
	for (nloop = 0 ; nloop < MAX_PV_COUNT ; nloop ++)
	{	
		checkCudaErrors(cudaMalloc((void**)&d_TBL24[nloop],		f4 * sizeof(unsigned short)));
		checkCudaErrors(cudaMalloc((void**)&d_TBLlong[nloop],		f3 * sizeof(unsigned char)));


		h_TBL24[nloop] = (unsigned short*)malloc(f4 * sizeof(unsigned short));
		memset(h_TBL24[nloop], sizeof(f4 * sizeof(unsigned short)), 0);
			
		h_TBLlong[nloop] = (unsigned char*)malloc(f3 * sizeof(unsigned char));
		memset(h_TBLlong[nloop], sizeof(f4 * sizeof(unsigned char)), 0);
				
		mapping_pv_with_rt[nloop] = (_mapping_pv_with_rt*)malloc(sizeof(_mapping_pv_with_rt));

	//	checkCudaErrors(cudaMemcpy(d_TBL24[nloop], h_TBL24[nloop], f4 * sizeof(unsigned short), cudaMemcpyHostToDevice));
	//	checkCudaErrors(cudaMemcpy(d_TBLlong[nloop], h_TBLlong[nloop], f3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	}
	return;
	
}

// 종료시 해제
void free_memory(){
	checkCudaErrors(cudaFreeHost(h_sm));
	checkCudaErrors(cudaFreeHost(h_hop));



	
	int nloop = 0;
	for(nloop = 0 ; nloop < g_currnet_pv_cout; nloop ++)
	{
		checkCudaErrors(cudaFree(d_TBL24[nloop]));
		checkCudaErrors(cudaFree(d_TBLlong[nloop]));
		free(h_TBL24[nloop]);
		free(h_TBLlong[nloop]);
		free(mapping_pv_with_rt[nloop]);
	}

	free(g_temp_flag);
	free(g_rt_info);
	free(g_cuda_rtupdate_que);
	free(h_TBL24_flag);
	
	checkCudaErrors(cudaThreadExit());

	return;
}


/*

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
			temp = pow(24 - prefix) + flag;
			for(i=flag; i<temp; i++)
				h_TBL24[i] = (unsigned short)hopbit;
		}
		else{ //Prefix is bigger than 24
			if(h_TBL24[flag] == 0){ //Empty entry
				h_TBL24[flag] = 0x8000 | index_count;
				flag = d & charmask[prefix - 24];
				temp = pow(32 - prefix) + flag;			
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
					temp = pow(32 - prefix) + flag;			
					for(i=index_count*256+flag; i<index_count*256+temp; i++)
						h_TBLlong[i] = (unsigned char)hopbit;
					index_count++;
				}
				else{ //If TBLlong entry
					indextemp = h_TBL24[flag] & 0x7F;
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag;			
					for(i=indextemp*256+flag; i<indextemp*256+temp; i++)
						h_TBLlong[i] = (unsigned char)hopbit;
				}
			}
		}
		n++;
	}

	checkCudaErrors(cudaMemcpy(d_TBL24, h_TBL24, f4 * sizeof(unsigned short), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_TBLlong, h_TBLlong, f3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	return;
}

*/



void Add_mapping_pv_with_rt(unsigned int _nindex, unsigned int _pv_id)
{
	mapping_pv_with_rt[_nindex]->nindex = _nindex;
	mapping_pv_with_rt[_nindex]->npv_id = _pv_id;

}


/*
void Del_mapping_pv_with_rt(unsigend _pv_id)
{
	int nloop, t_index;
	nloop = t_index = 0;
	for(nloop = 0; nloop < g_currnet_pv_cout ; nloop ++)
	{
		if(mapping_pv_with_rt[nloop]->pv_id == _pv_id)
		{
			if(nloop == 0 || nloop == g_currnet_pv_cout )
			{
				mapping_pv_with_rt[nloop]->nindex = NULL;
				mapping_pv_with_rt[nloop]->pv_id = NULL;
				break;
			}
			else
			{
				while(nloop < g_currnet_pv_cout)
				{
				
					t_index = nloop + 1; 
					mapping_pv_with_rt[nloop]->nindex = mapping_pv_with_rt[t_index]->nindex;
					mapping_pv_with_rt[nloop]->pv_id = mapping_pv_with_rt[t_index]->pv_id;
					
				//	mapping_pv_with_rt[t_index]->nindex = NULL;
				//	mapping_pv_with_rt[t_index]->pv_id = NULL;
					nloop ++;
				}
				break;
			
			}
		}
	}

}

*/

unsigned int savanna_searchRT(unsigned int pv_id)
{
	int nloop = 0;
	for(nloop = 0; nloop < g_currnet_pv_cout ; nloop ++)
	{
		if(mapping_pv_with_rt[nloop]->npv_id == pv_id)
		{
			return nloop; 
		}
	}
	
	return -1;
}



/*

void _savanna_ipv4_insert(unsigned int pv_id, unsigned int cur_rt, int dst_addr, int _prefix, unsigned int nexthop, int rt_update)   //Created based on DIR-24-8 Algorithm 
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


	switch(prefix)
	{
		
		case 8:
			blc_x = 10, blc_y = 60;
			th_x = 32, th_y = 32;		
			thread_range = 7600;
		break;
		
		case 16:case 24:case 32:
			blc_x = 3, blc_y = 4;
			th_x = 6, th_y = 4;
			thread_range = 100;
		break;

	}

	dim3 rt_blocks(blc_x, blc_y);
	dim3 rt_threads(th_x, th_y);
	

if(rt_update)
{
		checkCudaErrors(cudaMemcpy(d_TBL24[cur_rt], h_TBL24[cur_rt], f4 * sizeof(unsigned short), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_TBLlong[cur_rt], h_TBLlong[cur_rt], f3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
		
}

		if(prefix <= 24)
		{
		//	gettimeofday(&op_time[8], NULL);
			nprefix_position = 1;
			flag = flag & mask[prefix];
			temp = pow(24 - prefix) + flag;
	
		//	routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], flag, temp);
			
		//	gettimeofday(&op_time[9], NULL);

		//	checkCudaErrors(cudaThreadSynchronize());


			if(prefix == 8)
			{
				routing_TBLtable8_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], flag, temp);
			}
			else
			{
				routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], flag, temp);
			}
		//	ntbl_fir_start = flag;
		//	ntbl_range = temp;
			
		//	for(i = flag; i < temp; i++)
		//		h_TBL24[cur_rt][i] = (unsigned short)hopbit;


		//	gettimeofday(&op_time[10], NULL);


				
	}
		else
		{ //Prefix is bigger than 24			
			cudaMemcpy(g_temp_flag, &d_TBL24[cur_rt][flag] , sizeof(unsigned int), cudaMemcpyDeviceToHost );
			if(*g_temp_flag == 0)
				
		//	if(h_TBL24[cur_rt][flag] == 0)
			{ //Empty entry
				
			//ih_TBL24[cur_rt][flag] = 0x8000 | index_count;
			//	routing_TBLtable_get_flag_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], d_flag, flag-1, flag);
			//	cudaMemcpy(g_temp_flag, &d_TBL24[cur_rt][flag] , sizeof(unsigned int), cudaMemcpyDeviceToHost );

				nprefix_position = 2;
				memset(g_temp_flag, sizeof(unsigned int), 0);
				*g_temp_flag = 0x8000 | index_count;
				cudaMemcpy( &d_TBL24[cur_rt][flag] , g_temp_flag, sizeof(unsigned int), cudaMemcpyHostToDevice ); 
				printf("h_flag %d", *g_temp_flag);
			//	printf("pre32_pos2_val %d", *pre32_pos2_val);
		
			//	pre32_pos2_val = h_TBL24[cur_rt][flag];
			//	pre32__pos2_flag = flag;
				flag = d & charmask[prefix - 24];
				temp = pow(32 - prefix) + flag;	
				ntbl_fir_start = index_count * 256 + flag;
				ntbl_range = index_count * 256 + temp;
		//		for(i = index_count * 256 + flag; i < index_count * 256 + temp ; i++)
		//			h_TBLlong[cur_rt][i] = (unsigned char)hopbit;

				routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], ntbl_fir_start, ntbl_range);
		
				index_count++;
			}
			else{ //Something in entry

			//	if((h_TBL24[cur_rt][flag] & 0x8000) == 0x0000) //If TBL24 entry
			//	{ 
			//	
				cudaMemcpy(g_temp_flag, &d_TBL24[cur_rt][flag] , sizeof(unsigned int), cudaMemcpyDeviceToHost );
		
			//	routing_TBLtable_get_flag_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], d_flag, flag-1, flag);
			//	printf("h_flag %d", *h_flag);
				if((*g_temp_flag & 0x8000) == 0x0000)
					
		//		if((h_TBL24[cur_rt][flag] & 0x8000) == 0x0000) //If TBL24 entry
				{ 
					nprefix_position = 3;
				//	routing_TBLtable_get_flag_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], d_flag, flag-1, flag);
				//	memcpy(&hoptemp, h_flag, sizeof(unsigned int));

				//	hoptemp = *h_flag;

				//	printf("h_flag %d", *h_flag);
				//	printf("hoptemp %d", *hoptemp);

					hoptemp = *g_temp_flag;
					temp_flag = flag;
					
					memset(g_temp_flag, sizeof(unsigned int), 0);
					*g_temp_flag = 0x8000 | index_count;
					cudaMemcpy( &d_TBL24[cur_rt][flag] , g_temp_flag, sizeof(unsigned int), cudaMemcpyHostToDevice ); 
			
				//	hoptemp = h_TBL24[cur_rt][flag];
				//	temp_flag = flag;
				//	pre32_pos3_val = 0x8000 | index_count;
				//	routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, pre32_pos3_val, d_TBL24[cur_rt], temp_flag, temp_flag+1 );
					
				//	h_TBL24[cur_rt][flag] = 0x8000 | index_count;
				//	pre32_pos3_val = h_TBL24[cur_rt][flag];
					ntbl_fir_start = index_count * 256;
					ntbl_range = index_count * 256 + 256;
				//	for(i = index_count * 256; i < index_count * 256 + 256; i++) 
				//		h_TBLlong[cur_rt][i] = (unsigned char)hoptemp;
				
					routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hoptemp, d_TBLlong[cur_rt], ntbl_fir_start, ntbl_range);
	
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag;			
					ntbl_sec_start = index_count * 256 + flag;
					ntbl_sec_range = index_count * 256  + temp;
					
			//		for(i = index_count * 256 + flag ; i < index_count * 256 + temp; i++)
			//			h_TBLlong[cur_rt][i] = (unsigned char)hopbit;
					routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hoptemp, d_TBLlong[cur_rt], ntbl_sec_start, ntbl_sec_range);
			
					index_count++;
				}
				else{ //If TBLlong entry
				
					nprefix_position = 4;	

					cudaMemcpy(g_temp_flag, &d_TBL24[cur_rt][flag] , sizeof(unsigned int), cudaMemcpyDeviceToHost );
	
				//	routing_TBLtable_get_flag_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], d_flag, flag-1, flag);
				
				//	indextemp = *h_flag;
					
				//	memcpy(&indextemp, h_flag, sizeof(unsigned int));
					indextemp = *g_temp_flag & 0x7F;
					//	= h_TBL24[cur_rt][flag] & 0x7F;				
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag;	
					ntbl_fir_start = indextemp * 256 + flag;
					ntbl_range = indextemp * 256 + temp;
			//		for( i = indextemp * 256 + flag; i < indextemp * 256 + temp; i++)
			//			h_TBLlong[cur_rt][i] = (unsigned char)hopbit;

					routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBLlong[cur_rt], ntbl_fir_start, ntbl_range);
			
				}
			}
			}
	


	return;
}

*/




void _savanna_ipv4_insert(unsigned int pv_id, unsigned int cur_rt, int dst_addr, int _prefix, unsigned int nexthop, int rt_update)   //Created based on DIR-24-8 Algorithm 
{

	if(dst_addr == NULL)
		return ;
		
	
//	u32 dst_network_max = dst_network_base | ((u64)0x0FFFFFFFF >> prefix);		// Netmask applied maximum addr
//	u32 tbl24_base = dst_network_base >> 8; 	// Index for tbl24 - base
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

//	gettimeofday(&op_time[6], NULL);

//intoosh
	rare_flag = ntohl(dst_addr);
//	rare_flag = dst_addr;
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


	switch(prefix)
	{
		
		case 8:
			blc_x = 10, blc_y = 60;
			th_x = 32, th_y = 32;		
			thread_range = 7600;
		break;
		
		case 16:case 24:case 32:
			blc_x = 3, blc_y = 4;
			th_x = 6, th_y = 4;
			thread_range = 100;
		break;

	}

	dim3 rt_blocks(blc_x, blc_y);
	dim3 rt_threads(th_x, th_y);
	

if(rt_update)
{
		checkCudaErrors(cudaMemcpy(d_TBL24[cur_rt], h_TBL24[cur_rt], f4 * sizeof(unsigned short), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_TBLlong[cur_rt], h_TBLlong[cur_rt], f3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
}

		if(prefix <= 24)
		{
		//	gettimeofday(&op_time[8], NULL);
			nprefix_position = 1;
			flag = flag & mask[prefix];
			temp = pow(24 - prefix) + flag;
	
		//	routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], flag, temp);
			
		//	gettimeofday(&op_time[9], NULL);

		//	checkCudaErrors(cudaThreadSynchronize());


			if(prefix == 8)
			{
				routing_TBLtable8_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], flag, temp);
			}
			else
			{
				routing_TBLtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBL24[cur_rt], flag, temp);
			}
		//	ntbl_fir_start = flag;
		//	ntbl_range = temp;
			
		//	for(i = flag; i < temp; i++)
		//		h_TBL24[cur_rt][i] = (unsigned short)hopbit;


		//	gettimeofday(&op_time[10], NULL);
		


		
	}
		else{ //Prefix is bigger than 24	
		//NUm _ 1
		gettimeofday(&op_time[9], NULL);
		checkCudaErrors(cudaMemcpy(&h_TBL24_flag[flag], &d_TBL24[cur_rt][flag] , sizeof(unsigned short), cudaMemcpyDeviceToHost ));
		gettimeofday(&op_time[10], NULL);

		if(h_TBL24_flag[flag] == 0)
			{
				nprefix_position = 2;
				h_TBL24_flag[flag] = 0x8000 | index_count;
				//NUm _ 2
				gettimeofday(&op_time[11], NULL);
				checkCudaErrors(cudaMemcpy( &d_TBL24[cur_rt][flag] , &h_TBL24_flag[flag], sizeof(unsigned short), cudaMemcpyHostToDevice )); 
				gettimeofday(&op_time[12], NULL);

			//	printf("h_flag %d", h_TBL24_flag[flag]);
				flag = d & charmask[prefix - 24];
				temp = pow(32 - prefix) + flag; 
				ntbl_fir_start = index_count * 256 + flag;
				ntbl_range = index_count * 256 + temp;
		
				routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBLlong[cur_rt], ntbl_fir_start, ntbl_range);
		
				index_count++;
			}
			else{ //Something in entry
				
				// NUm _ 3
				gettimeofday(&op_time[13], NULL);
				checkCudaErrors(cudaMemcpy(&h_TBL24_flag[flag], &d_TBL24[cur_rt][flag] , sizeof(unsigned short), cudaMemcpyDeviceToHost ));
				gettimeofday(&op_time[14], NULL);
				if((h_TBL24_flag[flag] & 0x8000) == 0x0000) 		
				{ 
					nprefix_position = 3;
				
					hoptemp = h_TBL24_flag[flag];
					temp_flag = flag;
					
					h_TBL24_flag[flag] = 0x8000 | index_count;
				// Num 4
					gettimeofday(&op_time[15], NULL);
					checkCudaErrors(cudaMemcpy( &d_TBL24[cur_rt][flag] , &h_TBL24_flag[flag], sizeof(unsigned short), cudaMemcpyHostToDevice )); 
					gettimeofday(&op_time[16], NULL);
					
					ntbl_fir_start = index_count * 256;
					ntbl_range = index_count * 256 + 256;
				
					routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hoptemp, d_TBLlong[cur_rt], ntbl_fir_start, ntbl_range);
		
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag; 		
					ntbl_sec_start = index_count * 256 + flag;
					ntbl_sec_range = index_count * 256	+ temp;
				//intoosh	
				//	routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hoptemp, d_TBLlong[cur_rt], ntbl_sec_start, ntbl_sec_range);
					routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBLlong[cur_rt], ntbl_sec_start, ntbl_sec_range);
			
									
					index_count++;
				}
				else{ //If TBLlong entry
				
					nprefix_position = 4;	
				// NUm 5
					gettimeofday(&op_time[17], NULL);
					checkCudaErrors(cudaMemcpy(&h_TBL24_flag[flag], &d_TBL24[cur_rt][flag] , sizeof(unsigned short), cudaMemcpyDeviceToHost ));
					gettimeofday(&op_time[18], NULL);
					indextemp = h_TBL24_flag[flag] & 0x7F;
							
					flag = d & charmask[prefix - 24];
					temp = pow(32 - prefix) + flag; 
					ntbl_fir_start = indextemp * 256 + flag;
					ntbl_range = indextemp * 256 + temp;
		
					routing_TBLlongtable_update_with_gpu<<< rt_blocks, rt_threads >>>(thread_range, hopbit, d_TBLlong[cur_rt], ntbl_fir_start, ntbl_range);
			
					}
				
				}

			}

}




void _savanna_ipv4_delete(unsigned int _pv_id, unsigned int cur_rt, __be32 dst_addr, int _prefix )
{
	if(dst_addr == NULL)
		return ;
		
	//unsigned int flag = (cpu_to_be32(dst_addr) & ~(0x0FFFFFFFF >> prefix)); 	// Netmask applied
	
	unsigned int i, prefix, temp, hoptemp ,indextemp , index_count, flag, rare_flag;
	prefix = temp = indextemp = hoptemp =index_count = i = flag = rare_flag = 0;

	
	int d = 0;
//	int *nsep_ip = NULL;

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
	
	//printf("%d, prefix %d, hopbit %d\n", flag,prefix,hopbit); //getchar();	
	
	rare_flag = ntohl(dst_addr);
	d = rare_flag & (0x000000FF);

	flag = (rare_flag >> 8) & 0x00FFFFFF;
	
	//printf("%d, prefix %d, hopbit %d\n", flag,prefix,hopbit); //getchar();	
	
	printf("%x, prefix %d\n", flag,prefix); //getchar();

	
	prefix = _prefix;
		
	if(prefix <= 24)
	{
		flag = flag & mask[prefix];
		temp = pow(24 - prefix) + flag;
		for(i = flag; i < temp; i++)
			h_TBL24[cur_rt][i] = (unsigned short)NULL;
	}
	else{ //Prefix is bigger than 24
		if(h_TBL24[cur_rt][flag] == 0){ //Empty entry
			h_TBL24[cur_rt][flag] = 0x8000 | index_count;
			flag = d & charmask[prefix - 24];
			temp = pow(32 - prefix) + flag;			
			for(i = index_count * 256 + flag; i < index_count * 256 + temp ; i++)
				h_TBLlong[cur_rt][i] = (unsigned char)NULL;
			
			index_count++;
		}
		else{ //Something in entry
			if((h_TBL24[cur_rt][flag] & 0x8000) == 0x0000) //If TBL24 entry
			{ 
				hoptemp = h_TBL24[cur_rt][flag];
				h_TBL24[cur_rt][flag] = 0x8000 | index_count;
				for(i = index_count * 256; i < index_count * 256 + 256; i++) 
					h_TBLlong[cur_rt][i] = (unsigned char)NULL;
				
				flag = d & charmask[prefix - 24];
				temp = pow(32 - prefix) + flag;			

				for(i = index_count * 256 + flag ; i < index_count * 256 + temp; i++)
					h_TBLlong[cur_rt][i] = (unsigned char)NULL;

				index_count++;
			}
			else{ //If TBLlong entry
				indextemp = h_TBL24[cur_rt][flag] & 0x7F;
				flag = d & charmask[prefix - 24];
				temp = pow(32 - prefix) + flag;			
				for( i = indextemp * 256 + flag; i < indextemp * 256 + temp; i++)
					h_TBLlong[cur_rt][i] = (unsigned char)NULL;
			}
		}
	}

	checkCudaErrors(cudaMemcpy(d_TBL24[cur_rt], h_TBL24[cur_rt], f4 * sizeof(unsigned short), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_TBLlong[cur_rt], h_TBLlong[cur_rt], f3 * sizeof(unsigned char), cudaMemcpyHostToDevice));
	
}




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

/* GPU를 사용하는 실제 라우팅 함수.
pckt 배열에 처리할 IP주소가 쌓여 있으면, (이때는 4바이트의 a.b.c.d 형태)
우선 a, b, c, d로 쪼개어 받고,
이를 routing 함수를 돌려 처리한 후 종료함.
결과를 따로 긁어올 필요 없이, 이미 h_hop에 들어 있으니 이걸 send udp로 다시 클라이언트에게 보내주면 됨.
*/

void route(savanna_sm_list_t *_savanna_sm_list, int ipnum, unsigned int pv_id)
{

	int i = 0;
	unsigned short *temp_TBL24 = NULL;
	savanna_sm_list_t *savanna_sm_list = _savanna_sm_list;
	unsigned int SAVANNA_BUF_NUM_HALF = SAVANNA_BUF_NUM /2;
	int cur_rt = savanna_searchRT(pv_id);

//	printf("cur_rt = %d, pv_id = %d\n", cur_rt, pv_id);
//	savanna_print_mapping_pv_with_rt();
	
	gettimeofday(&op_time[0], NULL); // <-- check point
	
	for(i = 0; i < SAVANNA_BUF_NUM_HALF ; i++)
		memcpy((void *)&(h_sm[i * 4096]), savanna_sm_list->ptr[i], sizeof(unsigned int) * 1024);
	
	gettimeofday(&op_time[1], NULL); // <-- check point

	//memset(h_hop, 0, SAVANNA_BUF_NUM/2 * 1024 * sizeof(unsigned int));

	gettimeofday(&op_time[2], NULL); // <-- check point
	/////////////////////////////////////////////////////////////////////////////////////
	// 블록은 가로로 길게 늘어서 있음.
	// 각 블록은 32*32개의 쓰레드로 이루어져 있음.
	// 각 쓰레드가 1개의 실행 단위 즉 1개의 IP를 처리함.
	// 1블록은 32*32 = 1024 쓰레드므로 한 블록은 IP 1024개를 처리함.
	// 차후 GPU 커널 코드 내의 blockIdx.x 는 차후 실행 시에 각 쓰레드가 속한 블록 번호로 대체됨.
	// 마찬가지로 threadIdx.x 및 threadIdx.y 는 각 쓰레드의 x, y좌표로 대체됨.
	//dim3 blocks((unsigned int)ceil(ipnum / 1024)+1);
	dim3 blocks(10, 10);
	dim3 threads(32, 32);

	routing<<< blocks, threads >>>(d_sm, d_hop, d_TBL24[cur_rt], d_TBLlong[cur_rt]);
	getLastCudaError("Routing kernel execution failed");
	/////////////////////////////////////////////////////////////////////////////////////

	gettimeofday(&op_time[3], NULL); // <-- check point

	checkCudaErrors(cudaThreadSynchronize());

	gettimeofday(&op_time[4], NULL); // <-- check point

	
	for(i = 0; i < SAVANNA_BUF_NUM_HALF ; i++) 
		memcpy(savanna_sm_list->ptr[SAVANNA_BUF_NUM_HALF+i], (void *)&(h_hop[i * 1024]), sizeof(unsigned int) * 1024);
	
	gettimeofday(&op_time[5], NULL); // <-- check point
	
	
	return;
}

void* dma_malloc(int fd, unsigned int count, unsigned int _size, savanna_sm_list_t *_savanna_sm_list, unsigned int _listId){
	savanna_sm_list_t *savanna_sm_list = _savanna_sm_list;
	int mmap_size;
	unsigned int retval;
	struct mem_ioctl_data mid;

	mmap_size = 1; // 1 page

	savanna_sm_list->mmaprv[count] = mmap(0, mmap_size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
	if(savanna_sm_list->mmaprv[count] == MAP_FAILED){
		//printf("mmap failed\n");
		perror("mmap failed");
		return NULL;
	}
	else{
		//printf("mmap succeeded (mmaprv[%d] = %p)\n", count, savanna_sm_list->mmaprv[count]);
	}

	mid.requested_size = _size;
	mid.start_of_vma = (unsigned long)savanna_sm_list->mmaprv[count];
	mid.return_offset = 0xffffffff;
	mid.sm_listId = _listId;

	retval = ioctl(fd, IOCTL_MMAP, &mid);
	if(retval){
		//printf("ioctl 1 failed\n");
		perror("ioctl 1 failed");
		return NULL;
	}
	else{
		//printf("ioctl 1 succeeded\n");
	}
	if(mid.return_offset == 0xffffffff){
		//printf("mid.return_offset = 0x%lx\n", mid.return_offset);
		return NULL;
	}
	return (void *)((unsigned long)savanna_sm_list->mmaprv[count] + mid.return_offset);
}

// 디버그용 데이터 출력 - 파일.
void print_output_file(savanna_sm_list_t *_savanna_sm_list, unsigned int ipnum)
{
	savanna_sm_list_t *savanna_sm_list = _savanna_sm_list;
	int count;
	unsigned char *t;
	unsigned int *ip;
	FILE *fout = fopen("output_hvm.txt", "w");
	for(count = 0 ; count < ipnum; count++){
		ip = (unsigned int *)savanna_sm_list->ptr[count/1024];
		t = (unsigned char *)&(ip[count%1024]);
		fprintf(fout, "%5d %3d.%3d.%3d.%3d -> ", count, (int)t[3], (int)t[2], (int)t[1], (int)t[0]);
		ip = (unsigned int *)savanna_sm_list->ptr[SAVANNA_BUF_NUM/2+count/1024];
		fprintf(fout, "%d (sm id : %4d)   hop : %2d\n", ip[count%1024], savanna_sm_list->listId, h_hop[count]);
	}
	fclose(fout);
	printf("see output_hvm.txt for output\n");
}

// 디버그용 데이터 출력 - 화면.
void print_output_screen(savanna_sm_list_t *_savanna_sm_list, unsigned int ipnum) 
{
	savanna_sm_list_t *savanna_sm_list = _savanna_sm_list;
	int count;
	unsigned char *t;
	unsigned int *ip;
	for(count = 0 ; count < ipnum; count++){
		ip = (unsigned int *)savanna_sm_list->ptr[count/1024];
		t = (unsigned char *)&(ip[count%1024]);
		printf("%3d %3d.%3d.%3d.%3d -> ", count, (int)t[3], (int)t[2], (int)t[1], (int)t[0]);
		ip = (unsigned int *)savanna_sm_list->ptr[SAVANNA_BUF_NUM/2+count/1024];
		printf("%d (sm id : %4d)   hop : %2d\n", ip[count%1024], savanna_sm_list->listId, h_hop[count]);
	}
}

void print_routing_result() 
{
	printf("------------- result -----------\n");
	printf("input data time    : %6d us ( %d - %d )\n", getusec(op_time[0], op_time[1]), op_time[1].tv_usec, op_time[0].tv_usec);
	printf("host to dev time   : %6d us ( %d - %d )\n", getusec(op_time[1], op_time[2]), op_time[2].tv_usec, op_time[1].tv_usec);
	printf("kernel create time : %6d us ( %d - %d )\n", getusec(op_time[2], op_time[3]), op_time[3].tv_usec, op_time[2].tv_usec);
	printf("kernel run time    : %6d us ( %d - %d )\n", getusec(op_time[3], op_time[4]), op_time[4].tv_usec, op_time[3].tv_usec);
	printf("output data time   : %6d us ( %d - %d )\n", getusec(op_time[4], op_time[5]), op_time[5].tv_usec, op_time[4].tv_usec);
	printf("---> total : %d us\n", getusec(op_time[0], op_time[5]));
	printf("--------------------------------\n");
}

void print_total_result()
{
	printf("------------- totoal time ----------\n");
	printf("target sm search time    : %6d us ( %d - %d )\n", getusec(total_time[0], total_time[1]), total_time[1].tv_usec, total_time[0].tv_usec);
	printf("test                     : %6d us ( %d - %d )\n", getusec(total_time[1], total_time[2]), total_time[2].tv_usec, total_time[1].tv_usec);
	printf("route() time             : %6d us ( %d - %d )\n", getusec(total_time[2], total_time[3]), total_time[3].tv_usec, total_time[2].tv_usec);
	printf("send ioctl               : %6d us ( %d - %d )\n", getusec(total_time[3], total_time[4]), total_time[4].tv_usec, total_time[3].tv_usec);
	printf("print result time        : %6d us ( %d - %d )\n", getusec(total_time[4], total_time[5]), total_time[5].tv_usec, total_time[4].tv_usec);
	printf("---> total : %d us\n", getusec(total_time[0], total_time[5]));
	printf("------------------------------------\n");
}

void print_routing_table_upate_result() 
{
	printf("------------- result -----------\n");

/*
	printf("data copy from shardmem    : %6d us ( %d - %d )\n", getusec(op_time[0], op_time[1]), op_time[1].tv_usec, op_time[0].tv_usec);
	printf("search pv 1 : %6d us ( %d - %d )\n", getusec(op_time[1], op_time[2]), op_time[2].tv_usec, op_time[1].tv_usec);
	printf("search pv 2 : %6d us ( %d - %d )\n", getusec(op_time[2], op_time[3]), op_time[3].tv_usec, op_time[2].tv_usec);
	printf("ip_insrt    : %6d us ( %d - %d )\n", getusec(op_time[3], op_time[4]), op_time[4].tv_usec, op_time[3].tv_usec);
	printf("add map  : %6d us ( %d - %d )\n", getusec(op_time[4], op_time[5]), op_time[5].tv_usec, op_time[4].tv_usec);


	
	printf("ip : big ladian -> little ladian : %6d us ( %d - %d )\n", getusec(op_time[6], op_time[7]), op_time[7].tv_usec, op_time[6].tv_usec); 
	printf("get start and end  : %6d us ( %d - %d )\n", getusec(op_time[7], op_time[8]), op_time[8].tv_usec, op_time[7].tv_usec);
		
	printf("run gpu for routing table to update  : %6d us ( %d - %d )\n", getusec(op_time[8], op_time[9]), op_time[9].tv_usec, op_time[8].tv_usec);	
	
	printf("sycr time	: %6d us ( %d - %d )\n", getusec(op_time[9], op_time[10]), op_time[10].tv_usec, op_time[9].tv_usec);
*/



	printf("data copy num 1 time   : %6d us ( %d - %d )\n", getusec(op_time[9], op_time[10]), op_time[10].tv_usec, op_time[9].tv_usec);

	
	printf("data copy num 2 time   : %6d us ( %d - %d )\n", getusec(op_time[11], op_time[12]), op_time[12].tv_usec, op_time[11].tv_usec);

	
	printf("data copy num 3  time   : %6d us ( %d - %d )\n", getusec(op_time[13], op_time[14]), op_time[14].tv_usec, op_time[13].tv_usec);

	printf("data copy num 4  time	: %6d us ( %d - %d )\n", getusec(op_time[15], op_time[16]), op_time[16].tv_usec, op_time[15].tv_usec);

	
	printf("data copy num 5  time   : %6d us ( %d - %d )\n", getusec(op_time[17], op_time[18]), op_time[18].tv_usec, op_time[17].tv_usec);

	int count;
	for (count = 9 ; count < 19 ; count ++)
		memset(&op_time[count], 0x00, sizeof(timeval));

	
				
//	printf("---> total : %d us\n", getusec(op_time[0], op_time[5]));
	printf("--------------------------------\n");
}



void print_total_result_rt_hander()
{
	printf("------------- totoal time ----------\n");
	printf("get pv_id time           : %6d us ( %d - %d )\n", getusec(total_time[0], total_time[1]), total_time[1].tv_usec, total_time[0].tv_usec);
	printf("target sm search time    : %6d us ( %d - %d )\n", getusec(total_time[1], total_time[2]), total_time[2].tv_usec, total_time[1].tv_usec);
	printf("test                     : %6d us ( %d - %d )\n", getusec(total_time[2], total_time[3]), total_time[3].tv_usec, total_time[2].tv_usec);
	printf("rt update() time             : %6d us ( %d - %d )\n", getusec(total_time[3], total_time[4]), total_time[4].tv_usec, total_time[3].tv_usec);
	printf("send ioctl               : %6d us ( %d - %d )\n", getusec(total_time[4], total_time[5]), total_time[5].tv_usec, total_time[4].tv_usec);
	printf("print result time        : %6d us ( %d - %d )\n", getusec(total_time[5], total_time[6]), total_time[6].tv_usec, total_time[5].tv_usec);
	printf("---> total : %d us\n", getusec(total_time[0], total_time[6]));
	printf("------------------------------------\n");
}



// Rouitng Table 을 생선하고, upate 하는 시그널  핸들러. 



void SavannaSignalHandlerAboutRT(int signo, siginfo_t *info, void *unused)
{
//	printf("receive the request \n");
	savanna_sm_list_t *savanna_sm_list = NULL;
	unsigned int count = 0;
	unsigned int pv_id = NULL;
	int temp = 0;
	int err = 0;
	int que_front = 0;
	int iplookup_state = 0;
	int command = info->si_value.sival_int;
//	printf("command %d\n", command);
//	gettimeofday(&total_time[0], NULL); // <- check point
	if(command > 1)
	{
		temp = command;
		command = temp & MASK_SMLIST ;
		pv_id = temp >> GET_PVID_SHIFT;
	//	printf("pv_id = %d, command = %d\n", pv_id , command);
	}
//	gettimeofday(&total_time[1], NULL); 

	
	if(command == 0) {
		printf("Savanna : Mission Complete!!\n");
		state = 0;
	} else if(command > 0) {

	

	//	gettimeofday(&total_time[2], NULL); // <- check point
		savanna_sm_list = savanna_SearchSM(&savanna_sm_list_head, command);
	//	gettimeofday(&total_time[3], NULL);

		if(savanna_sm_list) {
			//printf("Savanna : do GPU launch!!(ref : %d)\n", command);
			
		//	ioctl(fd, IOCTL_IPLOOKUP_STATE, &iplookup_state);
		//	printf("iplookup_state = %d \n", run_iplookup );
			/*
			while(run_iplookup > 0) 
			{
		//		printf("iplookup_state = %d \n", run_iplookup );
			//	enqueue(g_cuda_rtupdate_que, savanna_sm_list , pv_id);
				usleep(10); 
			//	ioctl(fd, IOCTL_IPLOOKUP_STATE, &iplookup_state);
			}
			*/
			
			run_iplookup = -1;
		//do{	
				/*
				que_front = dequeue(g_cuda_rtupdate_que);
				if(que_front)
				{
					savanna_sm_list = g_cuda_rtupdate_que->_savanna_sm_list[que_front];
					pv_id = g_cuda_rtupdate_que->pv_id[que_front];
				}
			 	*/
		//	 	gettimeofday(&total_time[4], NULL); // <- check point
		//		gettimeofday(&total_time[2], NULL); // <- check point

				savanna_ipv4_update(savanna_sm_list, pv_id);
		//		gettimeofday(&total_time[3], NULL); // <- check point
		//		gettimeofday(&total_time[5], NULL); // <- check point
		//		ioctl(fd, IOCTL_GPUCOMP, (unsigned int)getusec(total_time[0], total_time[3]));
				ioctl(fd, IOCTL_RTCOMP, 0);
		//		print_total_result_rt_hander();
				
				print_routing_table_upate_result();
				
		//	}while(!is_empty(g_cuda_rtupdate_que));
	//		gettimeofday(&total_time[4], NULL); // <- check point
			run_iplookup = 0;
	//		print_output_file(savanna_sm_list, SAVANNA_IP_NUM); // 파일 출력

		//	print_routing_result(); // routing 결과 출력

	//		gettimeofday(&total_time[5], NULL); // <- check point

	//		print_total_result(); // 총 처리 시간

		} else {
			//printf("Savanna : do mmap() to make shared memory (ref : %d)\n", command);
			savanna_sm_list = savanna_CreateSM(command, pv_id);

			for(count = 0 ; count < SAVANNA_BUF_NUM ; count++){
				//printf("Reserving memory (%u bytes)...\n", SAVANNA_BUF_SIZE);
				savanna_sm_list->ptr[count] = dma_malloc(fd, count, SAVANNA_BUF_SIZE, savanna_sm_list, savanna_sm_list->listId);
				if(savanna_sm_list->ptr[count] == NULL){
					printf("dma_malloc failed.\n[%d] size: %d\n", count, SAVANNA_BUF_SIZE);
					return;
				} else {
					//printf("dma_malloc succeeded.\n[%d] size: %d, ptr = %p\n", count, SAVANNA_BUF_SIZE, savanna_sm_list->ptr[count]);
				}
			}

			savanna_AddSM(&savanna_sm_list_head, savanna_sm_list);
			//savanna_PrintSM_info(&savanna_sm_list_head);
		}
	} else {
		//printf("Savanna : do unmmap() to release shred memory(ref : %d)\n", command);
		command *= -1;

		savanna_sm_list = savanna_DelSM(&savanna_sm_list_head, command);
		if(savanna_sm_list == NULL) {
			printf("CUDA's shared memory is null\n");
			return;
		}

		for(count = 0; count < SAVANNA_BUF_NUM ; count++){
			err = munmap(savanna_sm_list->mmaprv[count], 1);
			if(err == -1){ //실패
				printf("munmap failed. list id : %d, savanna_sm_list->mmaprv[%d] \n", savanna_sm_list->listId, count);
				return;
			} else { // 성공
				//printf("munmap succeeded. list id : %d, savanna_sm_list->mmaprv[%d] \n", savanna_sm_list->listId, count);
			}
		}

		free(savanna_sm_list);
		//savanna_PrintSM_info(&savanna_sm_list_head);
		}
		
		
}


// 시그널 핸들러. 커널 측에서 mmap space에 처리할 데이터를 채워넣고 이 시그널을 건드리면
// 클라이언트는 mmap spcae의 데이터를 서버에게 보내고, 서버로부터 처리된 내용을 받아오게 됨.
// 현재 처리되어 받은 내용을, 커널로 다시 전달하는 부분은 미구현. ioctl을 쓸까 했음. (변수 xebra가 그 흔적)
void SavannaSignalHandler(int signo, siginfo_t *info, void *unused){

//	printf("Savanna : good day:!!\n");
	savanna_sm_list_t *savanna_sm_list;
	unsigned int count = 0;
	unsigned int pv_id = NULL;
	int temp = 0;
	int err = 0;
	int rtupdate_state = 0;
	int command = info->si_value.sival_int;
//	printf("Savanna : command %d !!\n", command);

	if(command > 1)
	{
		temp = command;
		command = temp & MASK_SMLIST ;
		pv_id = temp >> GET_PVID_SHIFT;
//		printf("pv_id = %d, command = %d\n", pv_id , command);
	}

//	printf("Savanna : command %d !!\n", command);

	if(command == 0) {
		printf("Savanna : Mission Complete!!\n");
		state = 0;
	} else if(command > 0) {
		
	//	gettimeofday(&total_time[0], NULL); // <- check point
		savanna_sm_list = savanna_SearchSM(&savanna_sm_list_head, command);

	//	gettimeofday(&total_time[1], NULL);
		if(savanna_sm_list) {
	//		printf("Savanna : do GPU launch!!(ref : %d)\n", command);

	//		gettimeofday(&total_time[2], NULL); // <- check point
		//	run_iplookup = 1;
			
		//	ioctl(fd, IOCTL_RTUPDATE_STATE, &rtupdate_state);
		//	printf("run_iplookup = %d \n", run_iplookup );
			/*
			while(run_iplookup < 0) 
			{ 
		//		printf("run_iplookup = %d \n", run_iplookup ); 
				usleep(10); 
			//	ioctl(fd, IOCTL_RTUPDATE_STATE, &rtupdate_state);
			}
			*/
			
			run_iplookup = 1;
		
			route(savanna_sm_list, SAVANNA_IP_NUM, pv_id);

	//		gettimeofday(&total_time[3], NULL); // <- check point

	//		ioctl(fd, IOCTL_GPUCOMP, (unsigned int)getusec(total_time[0], total_time[3]));
			ioctl(fd, IOCTL_GPUCOMP, 0);
			run_iplookup = 0;
	//		gettimeofday(&total_time[4], NULL); // <- check point

	//		print_output_file(savanna_sm_list, SAVANNA_IP_NUM); // 파일 출력

	//		print_routing_result(); // routing 결과 출력

	//		gettimeofday(&total_time[5], NULL); // <- check point

	//		print_total_result(); // 총 처리 시간

		} else {
			//printf("Savanna : do mmap() to make shared memory (ref : %d)\n", command);
			savanna_sm_list = savanna_CreateSM(command, pv_id);

			for(count = 0 ; count < SAVANNA_BUF_NUM ; count++){
				//printf("Reserving memory (%u bytes)...\n", SAVANNA_BUF_SIZE);
				savanna_sm_list->ptr[count] = dma_malloc(fd, count, SAVANNA_BUF_SIZE, savanna_sm_list, savanna_sm_list->listId);
				if(savanna_sm_list->ptr[count] == NULL){
					printf("dma_malloc failed.\n[%d] size: %d\n", count, SAVANNA_BUF_SIZE);
					return;
				} else {
					//printf("dma_malloc succeeded.\n[%d] size: %d, ptr = %p\n", count, SAVANNA_BUF_SIZE, savanna_sm_list->ptr[count]);
				}
			}

			savanna_AddSM(&savanna_sm_list_head, savanna_sm_list);
			//savanna_PrintSM_info(&savanna_sm_list_head);
		}
	} else {
		printf("Savanna : do unmmap() to release shred memory(ref : %d)\n", command);
		command *= -1;

		savanna_sm_list = savanna_DelSM(&savanna_sm_list_head, command);
		if(savanna_sm_list == NULL) {
			printf("CUDA's shared memory is null\n");
			return;
		}

		for(count = 0; count < SAVANNA_BUF_NUM ; count++){
			err = munmap(savanna_sm_list->mmaprv[count], 1);
			if(err == -1){ //실패
				printf("munmap failed. list id : %d, savanna_sm_list->mmaprv[%d] \n", savanna_sm_list->listId, count);
				return;
			} else { // 성공
			//	printf("munmap succeeded. list id : %d, savanna_sm_list->mmaprv[%d] \n", savanna_sm_list->listId, count);
			}
		}

		free(savanna_sm_list);
		//savanna_PrintSM_info(&savanna_sm_list_head);
	}
}




int main(){
	
	system("mknod /dev/savanna c 250 32");

	//open device file
	fd = open(DEVICE_FILENAME, O_RDWR|O_NDELAY|O_SYNC);
	if(fd < 0){
		//printf("dev open error\nmake dev file (mknod /dev/savanna c 250 32) or insmod module (insmod shcomp_domU.ko)\n");
		return 1;
	} else {
		//printf("/dev/savanna open succeeded\n");
	}

	// setting signal
	struct sigaction sig_lookup_ip; 
	sig_lookup_ip.sa_sigaction = SavannaSignalHandler;
	sig_lookup_ip.sa_flags = SA_SIGINFO;
	
	sigfillset(&(sig_lookup_ip.sa_mask)); // 모든 시그널 포함, 모든 시그널 감지
	if(sigaction(SAVANNA_SIG_NUM, &sig_lookup_ip, NULL) == -1)
	{
		printf("Error : SavannaSignalHandler]\n");
		return -1;
	}

	// send pid to module
	ioctl(fd, IOCTL_SETPID, getpid());
	printf("PID: %d\nSAVANNA READY!\n", getpid());	

	struct sigaction sig_update_routingtable;
	sig_update_routingtable.sa_sigaction = SavannaSignalHandlerAboutRT;
	sig_update_routingtable.sa_flags = SA_SIGINFO;
	
	sigfillset(&(sig_update_routingtable.sa_mask)); // 모든 시그널 포함, 모든 시그널 감지
	if(sigaction(SAVANNA_SIG_RT_NUM, &sig_update_routingtable, NULL) == -1)
	{
		printf("Error : SavannaSignalHandlerAboutRT]\n");
		return -1;
	}

	ioctl(fd, IOCTL_SETPID_RT, getpid());
	printf("PID: %d\nSAVANNA READY!\n", getpid());	
	
	
	alloc_memory(); // 할당
	printf("allocating memory...\n");


	init(g_cuda_rtupdate_que);

	/*
	input_rentry(); // 라우팅 테이블 입력
	printf("input rentry...\n");
	create_table(); // TBL테이블 생성
	printf("creating table...\n");
	//print_table(); // 디버그 출력
	//printf("printing table...\n");

	*/
	
	run_iplookup = 0;

	state = 1;
	
	// pause
	while(state) {usleep(10);}

	free_memory();
	printf("deallocating memory...\n");

	printf("Goodbye!!\n");
	system("rm -rf /dev/savanna");

	return 0;
}

savanna_sm_list_t* savanna_CreateSM(int _listId, unsigned int pv_id) 
{
	savanna_sm_list_t *savanna_sm_list;

	savanna_sm_list = (savanna_sm_list_t *)malloc(sizeof(savanna_sm_list_t));

	if(!savanna_sm_list){
		printf("savanna : new shared memory allocation is filed\n");
		return NULL;
	}

	savanna_sm_list->listId = _listId;
	savanna_sm_list->domid = pv_id;
	savanna_sm_list->next = NULL;
	savanna_sm_list->pre = NULL;

	return savanna_sm_list;
}

void savanna_AddSM(savanna_sm_list_t **_head, savanna_sm_list_t *_sm)
{
	savanna_sm_list_t *temp;

	if(_sm == NULL)
	{
		printf("savanna : inputed shared memory is null\n");
		return;
	}

	if((*_head) == NULL)
		(*_head) = _sm;
	else {
		temp = (*_head);
		while(temp != NULL)
		{
			if(temp->next == NULL)
			{
				_sm->pre = temp;
				temp->next = _sm;
				break;
			}
			temp = temp->next;
		}
	}

	//printf("savanna : Add new shared memory\n");
}

savanna_sm_list_t* savanna_SearchSM(savanna_sm_list_t **_head, unsigned int _listId)
{
	savanna_sm_list_t *temp = (*_head);

	while(temp != NULL) 
	{
		if(temp->listId == _listId)
			return temp;
		temp = temp->next;
	}
	return NULL;
}

savanna_sm_list_t* savanna_DelSM(savanna_sm_list_t **_head, unsigned int _listId)
{
	savanna_sm_list_t *temp = (*_head);
	savanna_sm_list_t *pre = NULL;

	if((*_head) == NULL)
		return NULL;

	if((*_head)->listId == _listId)
	{
		(*_head) = temp->next;
		//printf("savanna : delete sm information & head point moved to the next sm (sm id : %d)\n", temp->listId);
		return temp;
	} else {
		while(temp != NULL)
		{
			if(temp->listId == _listId)
			{
				pre->next = temp->next;
				//printf("savanna : delete sm information (sm id : %d)\n", temp->listId);
				return temp;
			}
			pre = temp;
			temp = temp->next;
		}
	}

	printf("savanna : not exist sm information (sm id : %d)\n", temp->listId);
	return NULL;
}

void savanna_Print_RT_info(_rt_info *client_rt_info)
{
	if(!client_rt_info)
		return ;


	printf("------------------------------------print RT information-----------------------------------\n");
	
		printf("Dom U id = %d, ", client_rt_info->pv_id);
		printf("RT CMD : %d, ", client_rt_info->cmd);
		printf("RT dest_addr = %d, ", client_rt_info->dest_addr);
		printf("RT prefix = %d, ",client_rt_info->prefix);
		printf("RT nexthop = %d\n", client_rt_info->nexthop);
		
	
	printf("-------------------------------------------------------------------------------------------\n");
}


void savanna_print_mapping_pv_with_rt()
{
	int nmax_count = 0;

	printf("---------------print mapping_pv_with_rt information-------------\n");
	while(nmax_count != MAX_PV_COUNT)
	{

		printf("mapping_pv_with_rt[nmax_count]->nindex = %d\n", mapping_pv_with_rt[nmax_count]->nindex );
		printf("mapping_pv_with_rt[nmax_count]->npv_id = %d\n", mapping_pv_with_rt[nmax_count]->npv_id );
	
		nmax_count ++;
		
	}
	
	printf("------------------------------------------------\n");
	
}


void savanna_PrintSM_info(savanna_sm_list_t **_head)
{
	savanna_sm_list_t *temp = (*_head);

	printf("---------------print sm information-------------\n");
	while(temp != NULL)
	{
		printf("[ Shared memory id = %5d ]\n", temp->listId);
		temp = temp->next;
	}
	printf("------------------------------------------------\n");
}
