#ifndef _SAVANNA_H_
#define _SAVANNA_H_
#include <linux/vmalloc.h>
#include <linux/spinlock.h>
#include <linux/kthread.h>
#include <linux/freezer.h>
#include <linux/time.h>
#include <linux/highmem.h>

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
#include <xen/interface/io/ring.h>
#include <xen/events.h>
#include <xen/xen.h>

#include <asm/xen/page.h>
#include <asm/siginfo.h>

#define CALL_DEV_NAME "savanna"
#define CALL_DEV_MAJOR 250

/* Response types */
#define RSP_ERROR       -1	// Operation failed for some unspecified reason (-EIO)
#define RSP_OKAY         0	 // Operation completed successfully.

/* Operation types */
#define REQ_RCV 1	
#define RESP_RCV 2

#define RONLY 1
#define RW 0

#define REQ_COMTEST 1
#define REQ_SETSM 3
#define REQ_RELEASE 5
#define REQ_GPULAUNCH 7


//intoosh

#define REQ_UPDATE_RT 9 // routing table update
#define REQ_LOOKUP_RT 11 // routing table lookup
#define REQ_SETSM_RT 13

//

// intoosh

#define RESP_SETSM_RT 10
#define RESP_UPDATE_RT 16
#define RESP_LOOKUP_RT 18

//



#define RESP_COMTEST 2
#define RESP_SETSM 4
#define RESP_RELEASE 6
#define RESP_GPULAUNCH 8

#define INVALID_REF	0
#define INVALID_OFFSET	0

/* IOCTL types */
#define IOCTL_BUILD 3
#define IOCTL_RELEASE 9
#define IOCTL_SETPID 13
#define IOCTL_SETPID_RT 14

#define IOCTL_CLOSE 15
#define IOCTL_MMAP 31
#define IOCTL_MUNMAP 33
#define IOCTL_GPUCOMP 41
#define IOCTL_RTCOMP 43
#define IOCTL_IPLOOKUP_STATE 45
#define IOCTL_RTUPDATE_STATE 47


#define IOCTL_RESP_DATA 21
#define IOCTL_SIGTEST_START 23
#define IOCTL_SIGTEST_END 25

/* Signal types */
#define SIG_SEND_DATA 27
#define SIG_SIGTEST_START 29
#define SIG_CLOSE 0


#define MBytes (1024*1024)

#define REF_CNT_IN_1PAGE (PAGE_SIZE / sizeof(int)) //1024
#define RTAB_SIZE (50*MBytes) // 52428800
#define RTAB_PAGE_CNT 200//DIV_ROUND_UP(RTAB_SIZE, PAGE_SIZE) // 12800

#define BITMAP_LENGTH (PAGE_SIZE*8) // 32768

typedef struct savanna_mem_mgr{
	unsigned long address;
	short size;
} savanna_mem_mgr;

typedef struct gntpg_t { 
	unsigned int ref;
	unsigned int *page;
	savanna_mem_mgr *savanna_tmp_mem;
} gntpg_t;

typedef struct gntpg_list_t { 
	domid_t domid;
	gntpg_t gntpg_bitmap;
	gntpg_t *gntpg_shareMem;
	unsigned int listId;
	unsigned int shareMem_num;
	unsigned int shareMem_minVal;
	struct gntpg_list_t *next;
	struct gntpg_list_t *pre;
}gntpg_list_t;

typedef struct savanna_request_t {
	unsigned long id;
	unsigned short operation;
	unsigned int option[3];
} savanna_request_t;

typedef struct savanna_response_t {
	unsigned long id;
	unsigned short operation;
	unsigned int option[3];
} savanna_response_t;

DEFINE_RING_TYPES(savanna, savanna_request_t, savanna_response_t);

typedef struct savannaServ_info_t {	
	domid_t           domid;
	atomic_t          refcnt;

	spinlock_t ring_lock;
	int ring_ref;
	struct savanna_back_ring ring;
	unsigned int evtchn;
	unsigned int irq;

	wait_queue_head_t   wq;
	struct task_struct  *savanna_evt_d;
	unsigned int waiting_reqs;
	wait_queue_head_t waiting_to_free;

	gntpg_list_t *gntpg_head;

	struct savannaServ_info_t *next;
	struct savannaServ_info_t *pre;
} savannaServ_info_t;

#define SAVANNA_SIG_NUM   44 
#define SAVANNA_SIG_RT_NUM 45
#define SAVANNA_BUF_NUM   10
#define SAVANNA_BUF_SIZE  4096
#define SAVANNA_MAXIP_NUM 100000

/*--------------------------Savanna 변수들 ---------------------------------*/
unsigned int *pages[RTAB_PAGE_CNT]; // 4KB * RTAB_PAGE_CNT 메모리 할당 - sorting
unsigned int ref_pages[RTAB_PAGE_CNT]; // 할당된 메모리의 reference number 저장

unsigned int bitTable[32] = { 0x80000000, 0x40000000, 0x20000000, 0x10000000,
	0x08000000, 0x04000000, 0x02000000, 0x01000000, 0x00800000, 0x00400000,
	0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000,
	0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400,
	0x00000200, 0x00000100, 0x00000080, 0x00000040, 0x00000020, 0x00000010,
	0x00000008, 0x00000004, 0x00000002, 0x00000001 };

#define mem_map_reserve(p)		set_bit(PG_reserved, &((p)->flags))
#define mem_map_unreserve(p)	clear_bit(PG_reserved, &((p)->flags))
#define NOPAGE_SIGBUS (NULL)

struct mem_ioctl_data{
	unsigned long requested_size;
	unsigned long start_of_vma;
	unsigned long return_offset;
	unsigned int sm_listId;
};

//int pid = -1;

struct siginfo sig_info, sig_info_rt; // signal info
struct task_struct *ts, *ts_rt; // signal info
struct pid *p, *p_rt; // signal info
int ret, ret_rt; // signal info
int pid, pid_rt; // signal info
/*-------------------------------------------------------------------------*/

struct timespec start, end, time[10];


// event channel관련 함수
int savanna_alloc_evtchn(domid_t *, int *);
int savanna_bind_evtchn_to_irqhandler(unsigned int , irq_handler_t , unsigned long , const char *, void *);

// shared memory 관련 함수
int set_grant_page(unsigned int, void *);
int offering_space_bw_doms(gntpg_list_t *);
int create_bitmap_ref(gntpg_list_t *);
void unmapping_space_bw_doms(gntpg_list_t *);

// pv list 관련 함수
void savanna_AddPV(savannaServ_info_t **, savannaServ_info_t *);
savannaServ_info_t* savanna_SearchPV(savannaServ_info_t **, unsigned short);
savannaServ_info_t* savanna_DelPV(savannaServ_info_t **, unsigned short);
void savanna_PrintPV(savannaServ_info_t **);

// shared memory 관리 함수
gntpg_list_t* savanna_CreateSM(unsigned long );
void savanna_AddSM(gntpg_list_t **, gntpg_list_t *);
gntpg_list_t* savanna_SearchSM(gntpg_list_t **, unsigned int);
gntpg_list_t* savanna_DelSM(gntpg_list_t **, unsigned int);
void savanna_PrintSM_info(gntpg_list_t **);


static inline void savanna_get(savannaServ_info_t *info)
{
	atomic_inc(&info->refcnt);
}

static inline void  savanna_put(savannaServ_info_t *info)
{
	if (atomic_dec_and_test(&info->refcnt))
		wake_up(&info->waiting_to_free);
}

int savanna_alloc_evtchn(domid_t *otherend_id, int *port) {
	struct evtchn_alloc_unbound alloc_unbound;
	int err;

	alloc_unbound.dom = DOMID_SELF;
	alloc_unbound.remote_dom = *otherend_id;

	err = HYPERVISOR_event_channel_op(EVTCHNOP_alloc_unbound, &alloc_unbound);

	if (err) {
		printk("event channel unbound error, err : %d\n", err);
	} else { 
		//printk("event channel unbound success, err : %d\n",err);
	}

	*port = alloc_unbound.port;
	//printk("port number : %d\n", *port);

	return err;
}

int savanna_bind_evtchn_to_irqhandler(unsigned int evtchn, irq_handler_t handler,
								  unsigned long irqflags, const char *devname, void *dev_id) {
	int irq, retval;

	irq = bind_evtchn_to_irq(evtchn);

	if (irq < 0) {
		printk("bind_evtchn_to_irq() error, irq : %d\n", irq);
		return irq;
	} else {
		//printk("bind_evtchn_to_irq() success, irq : %d\n", irq);
	}

	retval = request_irq(irq, handler, irqflags, devname, dev_id);

	if(retval < 0) {
		printk("request_irq() error, retval : %d\n", retval);
		//unbind_from_irq(irq);
		return retval;
	} else {
		//printk("request_irq() success, retval : %d\n", retval);
	}

	return irq;
}

int set_grant_page(unsigned int remote_domid, void *va) { 
	unsigned long frame;
	int ref = 0;

	frame = virt_to_mfn(va);

	ref = gnttab_grant_foreign_access(remote_domid, frame, RW);
	if(ref < 0) {
		printk("set_grant_page() : gnttab_grant_foreign_access() was failed (previous ref = %d)\n", ref);
		return ref;
	}

	return ref;
}

int offering_space_bw_doms(gntpg_list_t *_gntpg_list) 
{
	gntpg_list_t *gntpg_list = _gntpg_list;
	gntpg_t *gnt;
	unsigned short domid;
	int rtab_count;
	int err_count;
	int minval = -1;

	gntpg_list->gntpg_shareMem = (gntpg_t *)vmalloc(sizeof(gntpg_t)*RTAB_PAGE_CNT);
	if(!gntpg_list->gntpg_shareMem){
		printk("savanna(offering_space_bw_doms()) : vmalloc(gntpg_shareMem) is failed\n");
		vfree(gntpg_list->gntpg_shareMem);
		return -1;
	}

	gnt = gntpg_list->gntpg_shareMem;
	domid = gntpg_list->domid;

	for(rtab_count = 0; rtab_count < RTAB_PAGE_CNT; rtab_count++) {
		gnt[rtab_count].page = (void *)get_zeroed_page(GFP_KERNEL);
		if(!gnt[rtab_count].page) {
			printk("savanna(offering_space_bw_doms()) : get_zeroed_page() is failed\n"); 
			for(err_count=0; err_count < rtab_count; err_count++)
				free_page((unsigned long)gnt[rtab_count].page);
			return -1; 
		}

		gnt[rtab_count].ref = set_grant_page(domid, gnt[rtab_count].page);
		gnt[rtab_count].savanna_tmp_mem = NULL;


		if(minval == -1 || minval > gnt[rtab_count].ref)
			minval = gnt[rtab_count].ref;
	}

	gntpg_list->shareMem_num = rtab_count;
	gntpg_list->shareMem_minVal = minval;

	return 0;
}

int create_bitmap_ref(gntpg_list_t *_gntpg_list) 
{
	gntpg_list_t *gntpg_list = _gntpg_list;
	gntpg_t *gnt;
	gntpg_t gnt_temp;
	unsigned short domid;
	int minval;
	int temp;
	int bit;
	int count;
	int bitsetpoint;

	gnt = &gntpg_list->gntpg_bitmap;
	minval = gntpg_list->shareMem_minVal;
	domid = gntpg_list->domid;

	gnt->page = (void *)get_zeroed_page(GFP_KERNEL); // page 생성
	if(!gnt->page) {
		printk("create_bitmap_ref() - get_zeroed_page() is failed\n"); 
		free_page((unsigned long)gnt->page);
		return -1; 
	}

	gnt->ref =  set_grant_page(domid, gnt->page); // reference number 생성
	gntpg_list->listId = gnt->ref;
	if(gnt->ref < 0){
		printk("create_bitmap_ref -> set_grant_page() : gnttab_grant_foreign_access() is failed\n");
		gnttab_end_foreign_access(gnt->ref, RW, (unsigned long)gnt->page);
		free_page((unsigned long)gnt->page);
		return -1;
	}

	for(count = 0 ; count < gntpg_list->shareMem_num; count++) { 
		bitsetpoint = gntpg_list->gntpg_shareMem[count].ref - minval;
		gnt->page[bitsetpoint/32] = gnt->page[bitsetpoint/32] | bitTable[bitsetpoint % 32];
	}

	for(bit = 0, count = 0 ; bit < BITMAP_LENGTH; bit++){
		if ((gnt->page[bit / 32] & bitTable[bit % 32]) == bitTable[bit % 32]) {
			for(temp = 0; temp < gntpg_list->shareMem_num; temp++) {
				if(bit + minval == gntpg_list->gntpg_shareMem[temp].ref) {
					gnt_temp.page = gntpg_list->gntpg_shareMem[temp].page;
					gnt_temp.ref = gntpg_list->gntpg_shareMem[temp].ref;

					gntpg_list->gntpg_shareMem[temp].page = gntpg_list->gntpg_shareMem[count].page;
					gntpg_list->gntpg_shareMem[temp].ref = gntpg_list->gntpg_shareMem[count].ref;

					gntpg_list->gntpg_shareMem[count].page = gnt_temp.page;
					gntpg_list->gntpg_shareMem[count].ref = gnt_temp.ref;
				}
			}
			count++;
		}
		if (count > gntpg_list->shareMem_num) break;
	}

	return 0;
}

void unmapping_space_bw_doms(gntpg_list_t *_gntpg_list) 
{
	gntpg_list_t *gntpg_list = _gntpg_list;
	gntpg_t *gnt = gntpg_list->gntpg_shareMem;
	int count;

	if(gntpg_list == NULL)
		return ;

	if(gntpg_list->gntpg_bitmap.page)
		gnttab_end_foreign_access(gntpg_list->gntpg_bitmap.ref, RW, (unsigned long)gntpg_list->gntpg_bitmap.page);
	for(count = 0; count < gntpg_list->shareMem_num; count++) {
		if(gnt[count].page != NULL)
			gnttab_end_foreign_access(gnt[count].ref, RW, (unsigned long)gnt[count].page);
	}
	printk("delete all mapping pages\n");
}

void savanna_AddPV(savannaServ_info_t **_head, savannaServ_info_t *_pv)
{
	savannaServ_info_t *temp;

	if(_pv == NULL)
	{
		printk("pv is null\n");
		return;
	}

	if((*_head) == NULL)
		(*_head) = _pv;
	else {
		temp = (*_head);
		while(temp != NULL)
		{
			if(temp->next == NULL)
			{
				_pv->pre = temp;
				temp->next = _pv;
				break;
			}
			temp = temp->next;
		}
	}

	printk("savanna : Add new node\n");
}

savannaServ_info_t* savanna_SearchPV(savannaServ_info_t **_head, unsigned short domId)
{
	savannaServ_info_t *temp = (*_head);

	while(temp != NULL) 
	{
		if(temp->domid == domId)
			return temp;
		temp = temp->next;
	}
	return NULL;
}

savannaServ_info_t* savanna_DelPV(savannaServ_info_t **_head, unsigned short domId)
{
	savannaServ_info_t *temp = (*_head);
	savannaServ_info_t *pre = NULL;

	if((*_head) == NULL)
		return NULL;

	if((*_head)->domid == domId)
	{
		(*_head) = temp->next;
		return temp;
	} else {
		while(temp != NULL)
		{
			if(temp->domid == domId)
			{
				pre->next = temp->next;
				printk("savanna : deleted pv information (pv id : %d)\n", temp->domid);
				return temp;
			}
			pre = temp;
			temp = temp->next;
		}
	}
	
	//printk("savanna : not exist pv information (pv id : %d)\n", domId);
	return NULL;
}

void savanna_PrintPV(savannaServ_info_t **_head)
{
	savannaServ_info_t *temp = (*_head);

	if(temp == NULL) {
		printk("savanna : not exist PV information\n");
		return;
	}

	printk("-------------------print PV information------------------\n");
	while(temp != NULL)
	{
		printk("PV id = %d, ", temp->domid);
		printk("ring_ref = %d, ",temp->ring_ref);
		printk("event channel port = %d\n", temp->evtchn);
		temp = temp->next;
	}
	printk("---------------------------------------------------------\n");
}

gntpg_list_t* savanna_CreateSM(unsigned long _domid) 
{
	gntpg_list_t *gntpg_list;
	int err;

	gntpg_list = (gntpg_list_t *)vmalloc(sizeof(gntpg_list_t));

	if(!gntpg_list){
		printk("savanna : new shared memory allocation is filed\n");
		return NULL;
	}

	gntpg_list->domid = _domid; // Dom G(Back-end)의 Domain ID
	gntpg_list->gntpg_shareMem = NULL; // 일단 NULL, 공유 메모리 매핑하는 과정 필요
	gntpg_list->next = NULL; // 보나마나
	gntpg_list->pre = NULL; // 난 양쪽으로 이동할 수 있지

	err = offering_space_bw_doms(gntpg_list);
	err = create_bitmap_ref(gntpg_list);

	//printk("savanna : create new Shared Memory\n");
	//printk("Dom U ID : %d, ", gntpg_list->domid);
	//printk("Bitmap ref : %d, ", gntpg_list->gntpg_bitmap.ref);
	//printk("list ID : %d\n", gntpg_list->listId);
	//printk("Reference Count : %d, ", gntpg_list->shareMem_num);
	//printk("Minimum ref number : %d\n", gntpg_list->shareMem_minVal);

	return gntpg_list;
}

void savanna_AddSM(gntpg_list_t **_head, gntpg_list_t *_sm)
{
	gntpg_list_t *temp;

	if(_sm == NULL)
	{
		printk("savanna : inputed shared memory is null\n");
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
	
	//printk("savanna : Add new shared memory\n");
}

gntpg_list_t* savanna_SearchSM(gntpg_list_t **_head, unsigned int _listId)
{
	gntpg_list_t *temp = (*_head);

	while(temp != NULL) 
	{
		if(temp->listId == _listId)
			return temp;
		temp = temp->next;
	}
	return NULL;
}

gntpg_list_t* savanna_DelSM(gntpg_list_t **_head, unsigned int _listId)
{
	gntpg_list_t *temp = (*_head);
	gntpg_list_t *pre = NULL;

	if((*_head) == NULL)
		return NULL;

	if((*_head)->listId == _listId)
	{
		(*_head) = temp->next;
		printk("savanna : delete sm information (sm id : %d)\n", temp->listId);
		return temp;
	} else {
		while(temp != NULL)
		{
			if(temp->listId == _listId)
			{
				pre->next = temp->next;
				printk("savanna : delete sm information (sm id : %d)\n", temp->listId);
				return temp;
			}
			pre = temp;
			temp = temp->next;
		}
	}
	
	printk("savanna : not exist sm information (sm id : %d)\n", temp->listId);
	return NULL;
}



void savanna_PrintSM_info(gntpg_list_t **_head)
{
	gntpg_list_t *temp = (*_head);

	if(temp == NULL) {
		printk("savanna : not exist shared memory\n");
		return;
	}

	printk("------------------------------------print sm information-----------------------------------\n");
	while(temp != NULL)
	{
		printk("Dom U id = %d, ", temp->domid);
		printk("SM list ID : %d, ", temp->listId);
		printk("Bitmap ref = %d, ",temp->gntpg_bitmap.ref);
		printk("SM counter = %d, ", temp->shareMem_num);
		printk("SM minimum value = %d\n", temp->shareMem_minVal);
		temp = temp->next;
	}
	printk("-------------------------------------------------------------------------------------------\n");
}



#endif
