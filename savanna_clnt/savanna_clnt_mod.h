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
#define REQ_COMTEST 1 // ��� �׽�Ʈ
#define REQ_SETSM 3 // ���� �޸� �Ҵ� ��û
#define REQ_RELEASE 5 // ���� ����
#define REQ_GPULAUNCH 7 // gpu ���� ��û
#define REQ_UPDATE_RT 9 // routing table update
#define REQ_LOOKUP_RT 11 // routing table lookup
#define REQ_SETSM_RT 13


#define RESP_COMTEST 2 // ��� �׽�Ʈ 
#define RESP_SETSM 4 // ���� �޸� �Ҵ� ����
#define RESP_RELEASE 6 // ���� ���� -> ���� ��� x
#define RESP_GPULAUNCH 8 // gpu ���� ����
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
#define IOCTL_BIND 5 // hvm���� ����
#define IOCTL_BUILD_TEST 7 // ��� �׽�Ʈ
#define IOCTL_RELEASE 9 // hvm���� ���� ����
#define IOCTL_GPULAUNCH 11 // gpu����
#define IOCTL_SETSM 13 // ���� �޸� �Ҵ�



/* GY flags */

#define GY_DEBUG		0	//for print debugging info : 1 == print, 0 == not print
#define GY_TIME_ESTIMATE	0	//for estimate GPU response send to receive(or recieve to send) delay
								// : 1 == estimate, 0 == no estimate
#define NUMBER_SM 20	//# of Shared memory 
#define SAVANNA_IP_NUM 35000//Batch size


#define MBytes (1024*1024) // 1048576

#define REF_CNT_IN_1PAGE (PAGE_SIZE / sizeof(int)) //1024
#define RTAB_SIZE (50*MBytes) // 52428800
//#define RTAB_PAGE_CNT 2//((SAVANNA_IP_NUM/1024+1)*2)//DIV_ROUND_UP(RTAB_SIZE, PAGE_SIZE) // 12800      // ���� �޸� ������ ����.

#define BITMAP_LENGTH (PAGE_SIZE*8) // 32768

#define IP_GEN_POOL 0xFFFFFFFF

//routing_table_update

#define INSERT_RT 10
#define DELETE_RT 11
#define MAX_NETDEV_NAME 16


// pv�� ����ϱ� ���� user app�� �����ϰ� ����ϴ� ����ü.
// user app�� �̿��� otherend id�� ring page�� reference number, event channel�� ��Ʈ ��ȣ�� �Է��Ѵ�.
typedef struct savanna_bind_info {
	unsigned short otherend_id;
	int ring_ref;
	int evtchn_port;
} savanna_bind_info;

typedef struct gntpg_t { 
	grant_handle_t handle; // �ֱ淡 ���µ� �� ������ �𸥴�.
	grant_ref_t ref; // grant table�� ����� reference number.
	struct vm_struct *area; // area->addr�� �������� �����޸� ����.
} gntpg_t;

// ���� �޸� ����ü
typedef struct gntpg_list_t {
	domid_t domid; // hvm Domain id.
	gntpg_t gntpg_bitmap; // hvm���� grant table�� ����� �޸𸮵��� reference number ����.
	gntpg_t *gntpg_shareMem; // �������� �����޸𸮰� ����ȴ�. 
	unsigned char state; // ���� �� ���� �޸� ������ ��� ���������� ���� ���� ����. 1�� ��밡��, 0�� �Ұ���
	unsigned int listId; // bitmap�� reference number���� �� �����޸��� ���� id. �̸� �̿��� ���� �޸𸮸� ������ �� �ִ�.
	unsigned int shareMem_num; // �Ҵ���� ���� �޸� ����. RTAB_PAGE_CNT�� �����ϵ��� �����Ǿ���.
	unsigned int shareMem_minVal; // bitmap page�� ǥ�������� ����� ���� ������ �ϱ� ���� �ʿ�. bitmap���� ���� ���� ��ģ��.
	struct timespec gpuLaunch[6]; // �ð� ������ ����. ������ ��/�ķ� ����
	struct timespec datainput[2]; // �ð� ������ ����. ������ ��/�ķ� ����
	struct gntpg_list_t *next; // ���� �޸𸮷� ����
	struct gntpg_list_t *pre; // �� �ڷε� �����δ�.
} gntpg_list_t;

// request ����ü
typedef struct savanna_request_t {
	unsigned long id;
	unsigned short operation;
	unsigned int option[3];
} savanna_request_t;

// response ����ü
typedef struct savanna_response_t {
	unsigned long id;
	unsigned short operation;
	unsigned int option[3];
} savanna_response_t;

// ring ����ü�� �������� �ʿ��ϴ�.
DEFINE_RING_TYPES(savanna, savanna_request_t, savanna_response_t);

#define RSHM_RING_SIZE __RING_SIZE((struct savanna_sring *)0, PAGE_SIZE)

// PV�� ������ �����ϴ� ����ü, Ring������ frontend.h�� ����� ����ü�� ����ϴ�.
// �����ϴ� �����̱⿡ �ణ�� ������ �̷������.
typedef struct savannaClnt_info_t {
	// Unique identifier for this interface
	domid_t           domid; // hvm Domain ��ȣ
	unsigned int      handle; // ring page�� handle

	// interrupt
	unsigned int     irq; // Event channel�� ���ͷ�Ʈ irq��ȣ.. �³�? ��
	
	// Comms information
	struct savanna_front_ring ring; // ring ����ü ���� 
	struct vm_struct *ring_area; // ring page�� �ּҰ� ����� ����
	spinlock_t       ring_lock; // ring�� ����ȭ�� ����

	wait_queue_head_t   wq;  //
	struct task_struct  *savanna_evt_d; // ������ ����
	unsigned int        waiting_msgs; //

	savanna_request_t sended_reqs[RSHM_RING_SIZE]; // request ����
	unsigned long free_id; // ������ �״�� free�� ������ id (request�� ����)

	gntpg_list_t *gntpg_head; // ���� �޸��� �밡��. �̰� �ǵ�� ������

	// fields for a grant mechanism
	grant_handle_t shmem_handle; // ring page ��
	grant_ref_t    shmem_ref; // ring page ��


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

// event channel ���� �Լ� 
int savanna_bind_evtchn_to_irqhandler(unsigned int , unsigned int , irq_handler_t , unsigned long , const char *, void *);
int savanna_bind_evtchn_to_irq(unsigned int , unsigned int );

// shared memory ���� �Լ� 
int trans_bitmap_to_refnum(gntpg_list_t *);
int mapping_space_bw_doms(gntpg_list_t *);
int unmapping_space_bw_doms(gntpg_list_t *);

// GPU test ���� �Լ�
int create_sample_ip(gntpg_list_t *, int );
int print_data(gntpg_list_t *, int );

// shared memory ���� �Լ�
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
