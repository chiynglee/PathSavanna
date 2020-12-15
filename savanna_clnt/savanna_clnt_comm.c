#include "savanna_clnt_mod.h"

// bitmap for the shared memory
unsigned int bitTable[32] = { 0x80000000, 0x40000000, 0x20000000, 0x10000000,
	0x08000000, 0x04000000, 0x02000000, 0x01000000, 0x00800000, 0x00400000,
	0x00200000, 0x00100000, 0x00080000, 0x00040000, 0x00020000, 0x00010000,
	0x00008000, 0x00004000, 0x00002000, 0x00001000, 0x00000800, 0x00000400,
	0x00000200, 0x00000100, 0x00000080, 0x00000040, 0x00000020, 0x00000010,
	0x00000008, 0x00000004, 0x00000002, 0x00000001 };

// global variable ------------------
// header of the shared memory
savannaClnt_info_t *global_info;

struct timespec comTest_S, comTest_E;
struct timespec setSMTest_S, setSMTest_E;
struct timespec upRTTest_S, upRTTest_E;
int GPUTestState = 1;
int PfinderInit = 1;	//gy add
int sync_flag = 0;

#ifdef GY_DEBUG
ktime_t sr_calltime, sr_delta, sr_rettime, pf_sr_rettime,pf_sr_delta;
ktime_t rs_calltime, rs_delta, rs_rettime;
unsigned long long rs_duration, sr_duration, pf_duration;
#endif

// ------------------------------

// hvm과의 도메인간 통신을 위한 event channel 및 핸들로 등록을 진행한다.
int savanna_bind_evtchn_to_irqhandler(unsigned int remote_domain, unsigned int remote_port, irq_handler_t handler,
										  unsigned long irqflags, const char *devname, void *dev_id) {
	int irq, retval;

	irq = savanna_bind_evtchn_to_irq(remote_domain, remote_port);
	if (irq < 0) {
		printk("event channel bind error, irq : %d\n", irq);
		return irq;
	} else {
		//printk("event channel bind success, irq : %d\n", irq);
	}

	retval = request_irq(irq, handler, irqflags, devname, dev_id);
	if (retval != 0) {
		printk("request_irq() error : %d\n", retval);
		//unbind_from_irq(irq);
	} else {
		//printk("request_irq() success : %d\n", retval);
	}

	return irq;
}

// hvm Domain의 id롸 event channal의 포트 번호를 이용해 매핑하여 irq 번호를 반환받는다.
int savanna_bind_evtchn_to_irq(unsigned int remote_domain, unsigned int remote_port) {
	struct evtchn_bind_interdomain bind_interdomain;
	int err;

	bind_interdomain.remote_dom  = remote_domain;
	bind_interdomain.remote_port = remote_port;

	err = HYPERVISOR_event_channel_op(EVTCHNOP_bind_interdomain, &bind_interdomain);

	return err ? : bind_evtchn_to_irq(bind_interdomain.local_port);
}

// bitmap으로부터 공유 메모리의 reference number를 축출하는 함수
// 오름차순으로 정렬되며, hvm도 이를 지원한다.
int trans_bitmap_to_refnum(gntpg_list_t *_gntpg_list) {
	struct gnttab_map_grant_ref map_op;
	gntpg_t *gnt = &(_gntpg_list->gntpg_bitmap);
	gntpg_list_t *gntpg_list = _gntpg_list;
	unsigned int remote_domid = _gntpg_list->domid;
	unsigned int shareMem_minVal = _gntpg_list->shareMem_minVal;
	unsigned int shareMem_num = _gntpg_list->shareMem_num;
	unsigned int count;
	unsigned int bit;
	unsigned int *page;

	//printk("--------------------------------------------------\n");
	//printk("Dom G id: %5d, ref = %d, minVal = %d, count = %d\n", remote_domid, gntpg_list->gntpg_bitmap.ref, shareMem_minVal, shareMem_num);
	//printk("--------------------------------------------------\n");

	// bitmap page 매핑
	gnt->area = alloc_vm_area(PAGE_SIZE);
	if(gnt->area)
		// cylee : change the function name
		set_map_op(&map_op, (unsigned long)gnt->area->addr, GNTMAP_host_map, gnt->ref, remote_domid);

	if (HYPERVISOR_grant_table_op(GNTTABOP_map_grant_ref, &map_op, 1))
		BUG();
	gnt->handle = map_op.handle;

	// 매핑 실패 시 호출
	if(map_op.status != GNTST_okay)
	{
		xen_free_vm_area(gnt->area);
		printk("map_op.status = %d\n", map_op.status);
		return -1;
	}

	// bitmap page 접근
	page = (unsigned int*)(gnt->area->addr);

	// 매핑하기 위한 메모리 공간 확보
	gntpg_list->gntpg_shareMem = (gntpg_t *)vmalloc(sizeof(gntpg_t)*shareMem_num);

	// refernece number만 저장, 매핑은 mapping_space_bw_doms()에서 진행.
	for(bit = 0, count = 0 ; bit < BITMAP_LENGTH; bit++){
		if ((page[bit / 32] & bitTable[bit % 32]) == bitTable[bit % 32]) {
			gntpg_list->gntpg_shareMem[count].ref = shareMem_minVal + bit;
			count++;
		}
		if (count == shareMem_num) break;
	}

	return 0;
}

// 공유 메모리를 매핑하는 함수
int mapping_space_bw_doms(gntpg_list_t *_gntpg_list) {
	struct gnttab_map_grant_ref *map_op;
	gntpg_t *gnt = _gntpg_list->gntpg_shareMem;
	unsigned int remote_domid = _gntpg_list->domid;
	unsigned int shareMem_num = _gntpg_list->shareMem_num;
	unsigned int count;

	map_op = (struct gnttab_map_grant_ref *)kzalloc(sizeof(struct gnttab_map_grant_ref)*shareMem_num, GFP_KERNEL);

	for(count = 0; count < shareMem_num; count++) {
		gnt[count].area = alloc_vm_area(PAGE_SIZE);
		if (!gnt[count].area) {
			printk("mapping_space_bw_doms() : vm area allocation is failed(count : %d)\n", count);
			goto err_out;
		}

		// cylee : change the function name
		set_map_op(&(map_op[count]), (unsigned long)(gnt[count].area->addr), GNTMAP_host_map, gnt[count].ref, remote_domid);
	}

	HYPERVISOR_grant_table_op(GNTTABOP_map_grant_ref, map_op, shareMem_num);

	for(count = 0; count < shareMem_num; count++) {
		gnt[count].handle = map_op[count].handle;
		if(map_op[count].status != GNTST_okay) {
			printk("mapping_space_bw_doms() : hypercall failed (%d)\n", map_op[count].status);
			goto err_out;
		}
	}

	kfree(map_op);
	return 0;
err_out:

	unmapping_space_bw_doms(_gntpg_list);

	kfree(map_op);
	return -3;
}

// 공유 메모리 공간을 해제하는 함수다.
int unmapping_space_bw_doms(gntpg_list_t *_gntpg_list) {
	struct gnttab_unmap_grant_ref *unmap_op_sm;
	struct gnttab_unmap_grant_ref unmap_op_bm;
	gntpg_list_t *gntpg_list = _gntpg_list;
	gntpg_t *gnt_bm;
	gntpg_t *gnt_sm;
	unsigned int remote_domid;
	unsigned int bitmep_ref;
	unsigned int shareMem_num;
	unsigned int count;

	if(gntpg_list == NULL || gntpg_list->gntpg_shareMem == NULL)
	{
		printk("savanna : shared memory is null\n");
		return -1;
	}

	gnt_bm = &(gntpg_list->gntpg_bitmap); // bitmap page도 해제하기 위해
	gnt_sm = gntpg_list->gntpg_shareMem; // 공유 메모리 해제하기 위해
	remote_domid = gntpg_list->domid; // hvm의 domain id
	shareMem_num = gntpg_list->shareMem_num; //매핑된 공유 메모리 수
	bitmep_ref = gntpg_list->gntpg_bitmap.ref; // bitmap과 공유 메모리의 id

	printk("--------------------- unmapped page ----------------------\n");
	printk("Remote domain id : %d, ", remote_domid);
	printk("Shared momery count : %d\n", shareMem_num);

	// bitmap page 해제
	// cylee : change the function name
	set_unmap_op(&unmap_op_bm, (unsigned long)gnt_bm->area->addr, GNTMAP_host_map, gnt_bm->handle);
	gnt_bm->handle = ~0;

	HYPERVISOR_grant_table_op(GNTTABOP_unmap_grant_ref, &unmap_op_bm, 1);
	xen_free_vm_area(gnt_bm->area);
	printk("savanna : memory unmapping(bitmap), bitmep ref = %d\n", bitmep_ref);

	// shared memory 해제
	unmap_op_sm = (struct gnttab_unmap_grant_ref *)kzalloc(sizeof(struct gnttab_unmap_grant_ref)*shareMem_num, GFP_KERNEL);

	for(count = 0; count < shareMem_num; count++) {
		if(gnt_sm[count].area->addr != NULL) {
			// cylee : change the function name
			set_unmap_op(&(unmap_op_sm[count]), (unsigned long)(gnt_sm[count].area->addr), GNTMAP_host_map, gnt_sm[count].handle);
			gnt_sm[count].handle = ~0;
		}
	}

	// 하이퍼 콜
	HYPERVISOR_grant_table_op(GNTTABOP_unmap_grant_ref, unmap_op_sm, shareMem_num);

	// 제거
	for(count = 0; count < shareMem_num; count++)
		xen_free_vm_area(gnt_sm[count].area);
	printk("savanna : number of unmapped pages : %d\n", count);

	kfree(unmap_op_sm);
	vfree(gnt_sm);
	vfree(gntpg_list);

	printk("savanna : unmapped all pages\n");
	printk("----------------------------------------------------------\n");

	return 0;
}

// 샘플로 ip를 생성해 저장하는 함수.
int create_sample_ip(gntpg_list_t *_gntpg_list, int ipnum) 
{
	gntpg_list_t *gntpg_list = _gntpg_list;
	gntpg_t *gnt_sm;
    unsigned int temp;
    unsigned long seed;
    unsigned long hop;
    unsigned char *t;
	unsigned int *ip;
    int count;

	if(gntpg_list == NULL || gntpg_list->gntpg_shareMem == NULL)
	{
		printk("savanna(unmapping_space_bw_doms()) : shared memory is null\n");
		return -1;
	}

	gntpg_list->state = 0;

	gnt_sm = gntpg_list->gntpg_shareMem;
	t = (unsigned char *)&temp;
    hop = IP_GEN_POOL / ipnum;
	seed = 10000;

	for(count = 0 ; count < gntpg_list->shareMem_num; count++)
		memset(gntpg_list->gntpg_shareMem[count].area->addr, 0,	sizeof(int) * 1024);

	//printk("Creating IP data %d... seed = %lu, hop = %lu(%lx)\n", ipnum, seed, hop, hop);
	for(count = 0 ; count < ipnum ; count++){
        ip = (unsigned int *)gnt_sm[count/1024].area->addr;
		t[3] = (unsigned char)(seed/16777216);
		t[2] = (unsigned char)((seed/65536) % 256);
		t[1] = (unsigned char)((seed/256) % 256);
		t[0] = (unsigned char)(seed%256);
		ip[count%1024] = temp;
		seed+= hop;
	}
    return 0;
}

// 공유 메모리에 저장되어 있는 ip 및 hop정보를 호출한다.
// _gntpg_list : 공유 메모리 주소, sm_num : 공유한 페이지 개수
int print_data(gntpg_list_t *_gntpg_list, int sm_num) 
{
	gntpg_list_t *gntpg_list = _gntpg_list;
	gntpg_t *gnt_sm;
	int count;
	unsigned char *t;
	unsigned int *ip;

	if(gntpg_list == NULL || gntpg_list->gntpg_shareMem == NULL)
	{
		printk("savanna() : shared memory is null\n");
		return -1;
	}

	gnt_sm = gntpg_list->gntpg_shareMem;

	for(count = 0; count < 300; count++) { // 100은 임의로 정한 값으로 조정 가능
        ip = (unsigned int *)gnt_sm[count/1024].area->addr;
	    t = (unsigned char *)&(ip[count%1024]);
        printk("%3d   %3d.%3d.%3d.%3d - ", count, (int)t[3], (int)t[2], (int)t[1], (int)t[0]);
		ip = (unsigned int *)gnt_sm[sm_num/2+count/1024].area->addr; // +100페이지 이후의 hop 정보 출력
		printk("%d\n", ip[count%1024]);
	}
	return 0;
}

// hvm으로부터 받은 response로부터 bitmap page의 reference number를 꺼내 메모리 매핑을 시도한다.
// 매핑된 공유 메모리를 반환한다. savanna_AddSM()를 호출하면 list에 추가된다.
gntpg_list_t* savanna_CreateSM(unsigned long _domid, savanna_response_t *_resp) 
{
	gntpg_list_t *gntpg_list;
	savanna_response_t *resp;
	int err;

	gntpg_list = (gntpg_list_t *)vmalloc(sizeof(gntpg_list_t));
	resp = _resp;

	if(!gntpg_list){
		printk("savanna : new shared memory allocation is filed\n");
		return NULL;
	}

	gntpg_list->domid = _domid; // Dom G(Back-end)의 Domain ID
	gntpg_list->gntpg_bitmap.ref = resp->option[0]; // 공유 메모리들의 ref가 저장된 Bitmap 페이지의 ref값 ?!?!
	gntpg_list->gntpg_shareMem = NULL; // 일단 NULL, 공유 메모리 매핑하는 과정 필요
	gntpg_list->state = 1; // 1은 사용 가능, 0은 사용 불가능
	gntpg_list->listId = resp->option[0]; // 공유 메모리의 고유 식별 값 = Bitmap 페이지의 ref값
	gntpg_list->shareMem_num = resp->option[1]; // 공유 메모리 패밍 후 매핑한 페이지 수 설정
	gntpg_list->shareMem_minVal = resp->option[2]; // 공유 메모리 ref의 최소값
	gntpg_list->next = NULL; // 보나마나
	gntpg_list->pre = NULL; // 난 양쪽으로 이동할 수 있지

	err = trans_bitmap_to_refnum(gntpg_list); // bitmap에서 ref 값 축출
	if(err < 0)
	{
		unmapping_space_bw_doms(gntpg_list);
		return NULL;
	}

	err = mapping_space_bw_doms(gntpg_list); // 공유 메모리 매핑

	//printk("savanna : mapping new Shared Memory\n");
	//printk("Dom G ID : %d, ", gntpg_list->domid);
	//printk("Bitmap ref : %d, ", gntpg_list->gntpg_bitmap.ref);
	//printk("list ID : %d\n", gntpg_list->listId);
	//printk("Reference Count : %d, ", gntpg_list->shareMem_num);
	//printk("Minimum ref number : %d\n", gntpg_list->shareMem_minVal);

	return gntpg_list;
}

// 생성된 공유 메모리를 리스트의 가장 끝에 연결한다.
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

// 공유 메모리의 고유 id를 이용해 해당 공유 메모리를 검색한다.
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

// 사용가능한 공유 메모리 검색.
// 없으면 null 반환
gntpg_list_t* savanna_SearchEnabledSM(gntpg_list_t **_head)
{
	gntpg_list_t *temp = (*_head);

	while(temp != NULL) 
	{
		if(temp->state == 1){
			temp->state = 0; //intoosh
			return temp;
		}
		temp = temp->next;
	}
	return NULL;
}

// 사용가능한 공유메모리 반환.
// 없을경우 hvm으로부터 할당받아 제공, 따라서 경우에 따라 오래걸릴 수 있다.
gntpg_list_t* savanna_getEnabledSM()
{
	gntpg_list_t *gntpg_list = savanna_SearchEnabledSM(&global_info->gntpg_head); // 현재 가용 가능한 공유 메모리 공간 검사, 없으면 null 반환

	if(gntpg_list == NULL) { // null. 즉, 사용 가능한 공유 메모리가 없을 때
		GPUTestState = 1; // 조건 초기화
		getnstimeofday(&setSMTest_S); // 공유 메모리 확보까지 시간 체크, 보통 20ms(이건 좀 더 확인해봐야함)
		send_request(REQ_SETSM, 0, 0, 0); // hvm으로 요청
		while(GPUTestState) { udelay(1); } // response가 올 때까지 대기
		gntpg_list = savanna_SearchEnabledSM(&global_info->gntpg_head); // 다시 검사, 없을리가 읍지.
	}

	return gntpg_list;
}

// 공유메모리 제거 시 사용.
// 당장 없애는건 아니다. 해당 공유리를 리스트로
gntpg_list_t* savanna_DelSM(gntpg_list_t **_head, unsigned int _listId)
{
	gntpg_list_t *temp = (*_head);
	gntpg_list_t *pre = NULL;

	if((*_head) == NULL)
		return NULL;

	if((*_head)->listId == _listId)
	{
		(*_head) = temp->next;
		return temp;
	} else {
		while(temp != NULL)
		{
			if(temp->listId == _listId)
			{
				pre->next = temp->next;
				printk("savanna : deleted sm information (sm id : %d)\n", temp->listId);
				return temp;
			}
			pre = temp;
			temp = temp->next;
		}
	}
	
	printk("savanna : not exist sm information (sm id : %d)\n", temp->listId);
	return NULL;
}

// 현재 할당받은 공유메모리의 정보를 호출한다. 
void savanna_PrintSM_info(gntpg_list_t **_head)
{
	gntpg_list_t *temp = (*_head);

	if(temp == NULL) {
		printk("savanna : not exist shared memory\n");
		return;
	}

	printk("------------------------------------------print sm information-----------------------------------------\n");
	while(temp != NULL)
	{
		printk("Dom G id = %d, ", temp->domid);
		printk("SM ID : %d, ", temp->listId);
		printk("SM state : %d, ", temp->state);
		printk("Bitmap ref = %d, ",temp->gntpg_bitmap.ref);
		printk("SM counter = %d, ", temp->shareMem_num);
		printk("SM minimum value = %d\n", temp->shareMem_minVal);
		temp = temp->next;
	}
	printk("--------------------------------------------------------------------------------------------------------\n");
}

