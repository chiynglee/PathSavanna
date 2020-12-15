#ifndef _SAVANNA_H_
#define _SAVANNA_H_

#include <linux/vmalloc.h>
#include <linux/spinlock.h>
#include <linux/freezer.h>
#include <linux/kthread.h>
#include <linux/time.h>

#include <linux/ktime.h>

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>          
#include <linux/errno.h>       
#include <linux/types.h>       
#include <linux/fcntl.h>       
#include <asm/uaccess.h>
#include <asm/io.h>
#include <linux/stat.h>
#include <linux/proc_fs.h>

#include <xen/xenbus.h>
#include <xen/grant_table.h>
#include <xen/xen.h>
#include <xen/events.h>
#include <xen/interface/io/ring.h>

#include <asm/xen/page.h>
#include <asm/siginfo.h>

#define CALL_DEV_NAME "savanna"
#define CALL_DEV_MAJOR 250

/* Response types */
#define RSP_ERROR       -1	// Operation failed for some unspecified reason (-EIO)
#define RSP_OKAY         0	 // Operation completed successfully.

/* Operation types */
#define REQ_COMTEST 1 // 통신 테스트
#define REQ_SETSM 3 // 공유 메모리 할당 요청
#define REQ_RELEASE 5 // 연결 해제
#define REQ_GPULAUNCH 7 // gpu 가동 요청
#define REQ_UPDATE_RT 9 // routing table update
#define REQ_LOOKUP_RT 11 // routing table lookup
#define REQ_SETSM_RT 13


#define RESP_COMTEST 2 // 통신 테스트 
#define RESP_SETSM 4 // 공유 메모리 할당 응답
#define RESP_RELEASE 6 // 연결 해제 -> 아직 사용 x
#define RESP_GPULAUNCH 8 // gpu 가동 응답
#define RESP_SETSM_RT 10
#define RESP_UPDATE_RT 16
#define RESP_LOOKUP_RT 18


#define RONLY 1
#define RW 0

#define REQ_RCV 1
#define RESP_RCV 2

#define INVALID_REF	0
#define INVALID_OFFSET	0

/* IOCTL types */
#define IOCTL_BIND 5 // hvm과의 연결
#define IOCTL_BUILD_TEST 7 // 통신 테스트
#define IOCTL_RELEASE 9 // hvm과의 연결 해제
#define IOCTL_GPULAUNCH 11 // gpu가동
#define IOCTL_SETSM 13 // 공유 메모리 할당



/* GY flags */

#define GY_DEBUG		0	//for print debugging info : 1 == print, 0 == not print
#define GY_TIME_ESTIMATE	0	//for estimate GPU response send to receive(or recieve to send) delay
								// : 1 == estimate, 0 == no estimate
#define NUMBER_SM 20	//# of Shared memory 
#define SAVANNA_IP_NUM 35000//Batch size


#define MBytes (1024*1024) // 1048576

#define REF_CNT_IN_1PAGE (PAGE_SIZE / sizeof(int)) //1024
#define RTAB_SIZE (50*MBytes) // 52428800
//#define RTAB_PAGE_CNT 2//((SAVANNA_IP_NUM/1024+1)*2)//DIV_ROUND_UP(RTAB_SIZE, PAGE_SIZE) // 12800      // 공유 메모리 페이지 개수.

#define BITMAP_LENGTH (PAGE_SIZE*8) // 32768

#define IP_GEN_POOL 0xFFFFFFFF

//routing_table_update

#define INSERT_RT 10
#define DELETE_RT 11
#define MAX_NETDEV_NAME 16


// pv를 등록하기 위해 user app과 동일하게 사용하는 구조체.
// user app을 이용해 otherend id와 ring page의 reference number, event channel의 포트 번호를 입력한다.
typedef struct savanna_bind_info {
	unsigned short otherend_id;
	int ring_ref;
	int evtchn_port;
} savanna_bind_info;

typedef struct gntpg_t { 
	grant_handle_t handle; // 있길래 쓰는데 왜 이유는 모른다.
	grant_ref_t ref; // grant table에 저장된 reference number.
	struct vm_struct *area; // area->addr이 실질적인 공유메모리 공간.
} gntpg_t;

// 공유 메모리 고주체
typedef struct gntpg_list_t {
	domid_t domid; // hvm Domain id.
	gntpg_t gntpg_bitmap; // hvm에서 grant table에 등록한 메모리들의 reference number 저장.
	gntpg_t *gntpg_shareMem; // 실질적인 공유메모리가 저장된다. 
	unsigned char state; // 현재 이 공유 메모리 공간이 사용 가능한지에 대한 정보 저장. 1은 사용가능, 0은 불가능
	unsigned int listId; // bitmap의 reference number이자 이 공유메모리의 고유 id. 이를 이용해 공유 메모리를 지정할 수 있다.
	unsigned int shareMem_num; // 할당받은 공유 메모리 개수. RTAB_PAGE_CNT과 동일하도록 구현되었다.
	unsigned int shareMem_minVal; // bitmap page의 표현범위를 벗어나는 일이 없도록 하기 위해 필요. bitmap으로 나온 수와 합친다.
	struct timespec gpuLaunch[6]; // 시간 측정용 변수. 보내기 전/후로 측정
	struct timespec datainput[2]; // 시간 측정용 변수. 보내기 전/후로 측정
	struct gntpg_list_t *next; // 다음 메모리로 ㄱㄱ
	struct gntpg_list_t *pre; // 난 뒤로도 움직인다.
} gntpg_list_t;

// request 구조체
typedef struct savanna_request_t {
	unsigned long id;
	unsigned short operation;
	unsigned int option[3];
} savanna_request_t;

// response 구조체
typedef struct savanna_response_t {
	unsigned long id;
	unsigned short operation;
	unsigned int option[3];
} savanna_response_t;

// ring 구조체를 쓰기위해 필요하다.
DEFINE_RING_TYPES(savanna, savanna_request_t, savanna_response_t);

#define RSHM_RING_SIZE __RING_SIZE((struct savanna_sring *)0, PAGE_SIZE)

// PV의 정보를 저장하는 구조체, Ring파일의 frontend.h에 저장된 구조체와 비슷하다.
// 매핑하는 입장이기에 약간의 수정이 이루어졌다.
typedef struct savannaClnt_info_t {
	// Unique identifier for this interface
	domid_t           domid; // hvm Domain 번호
	unsigned int      handle; // ring page의 handle

	// interrupt
	unsigned int     irq; // Event channel의 인터럽트 irq번호.. 맞나? ㅋ
	
	// Comms information
	struct savanna_front_ring ring; // ring 구조체 변수 
	struct vm_struct *ring_area; // ring page의 주소가 저장된 변수
	spinlock_t       ring_lock; // ring의 동기화를 위해

	wait_queue_head_t   wq;  //
	struct task_struct  *savanna_evt_d; // 쓰레드 저장
	unsigned int        waiting_msgs; //

	savanna_request_t sended_reqs[RSHM_RING_SIZE]; // request 저장
	unsigned long free_id; // 변수명 그대로 free한 공간의 id (request를 위한)

	gntpg_list_t *gntpg_head; // 공유 메모리의 대가리. 이건 건들면 ㄴㄴ해

	// fields for a grant mechanism
	grant_handle_t shmem_handle; // ring page 용
	grant_ref_t    shmem_ref; // ring page 용


} savannaClnt_info_t;


typedef struct _client_rt_info
{
	int 		 pv_id;
	__be32 		 dest_addr;
	u8 			 prefix;
	int 		 cmd;
	unsigned int nexthop;
	
}_client_rt_info;



// bitmap for the shared memory 
// that defined in savanna_clnt_comm.c
extern unsigned int bitTable[32];

// global variables ------------------
// defined in savanna_clnt_comm.c
extern savannaClnt_info_t *global_info;	// header of the shared memory

extern struct timespec comTest_S, comTest_E;
extern struct timespec setSMTest_S, setSMTest_E;
extern struct timespec upRTTest_S, upRTTest_E;
extern int GPUTestState;
extern int PfinderInit;	//gy add
extern int sync_flag;
extern struct task_struct* waiting_clean_queue;
#ifdef GY_TIME_ESTIMATE
extern ktime_t sr_calltime, sr_delta, sr_rettime, pf_sr_rettime, pf_sr_delta;
extern ktime_t rs_calltime, rs_delta, rs_rettime;
extern unsigned long long rs_duration, sr_duration, pf_duration;
#endif

// ------------------------------

// event channel 관련 함수 
int savanna_bind_evtchn_to_irqhandler(unsigned int , unsigned int , irq_handler_t , unsigned long , const char *, void *);
int savanna_bind_evtchn_to_irq(unsigned int , unsigned int );

// shared memory 관련 함수 
int trans_bitmap_to_refnum(gntpg_list_t *);
int mapping_space_bw_doms(gntpg_list_t *);
int unmapping_space_bw_doms(gntpg_list_t *);

// GPU test 관련 함수
int create_sample_ip(gntpg_list_t *, int );
int print_data(gntpg_list_t *, int );

// shared memory 관리 함수
gntpg_list_t* savanna_CreateSM(unsigned long , savanna_response_t *);
void savanna_AddSM(gntpg_list_t **, gntpg_list_t *);
gntpg_list_t* savanna_SearchSM(gntpg_list_t **, unsigned int);
gntpg_list_t* savanna_SearchEnabledSM(gntpg_list_t **);
gntpg_list_t* savanna_getEnabledSM(void);
gntpg_list_t* savanna_DelSM(gntpg_list_t **, unsigned int);
void savanna_PrintSM_info(gntpg_list_t **);

// cylee : move a function declaration from savanna_clnt_mod.c to support the call from pfinder
void send_request(uint16_t _operation, uint32_t _option_1, uint32_t _option_2, uint32_t _option_3);
int init_GPGPU_module(void);
void cleanup_GPGPU_module(void);


// cylee : change function names to prevent duplcatation by the same linux function
static inline void set_map_op(struct gnttab_map_grant_ref *map, 
					phys_addr_t addr, uint32_t flags, grant_ref_t ref, domid_t domid)
{
	if (flags & GNTMAP_contains_pte)
		map->host_addr = addr;
	else if (xen_feature(XENFEAT_auto_translated_physmap))
		map->host_addr = __pa(addr);
	else
		map->host_addr = addr;

	map->flags = flags;
	map->ref = ref;
	map->dom = domid;
}

// cylee : change function names to prevent duplcatation by the same linux function
static inline void set_unmap_op(struct gnttab_unmap_grant_ref *unmap, 
							phys_addr_t addr, uint32_t flags, grant_handle_t handle)
{
	if (flags & GNTMAP_contains_pte)
		unmap->host_addr = addr;
	else if (xen_feature(XENFEAT_auto_translated_physmap))
		unmap->host_addr = __pa(addr);
	else
		unmap->host_addr = addr;

	unmap->handle = handle;
	unmap->dev_bus_addr = 0;
}

#endif
