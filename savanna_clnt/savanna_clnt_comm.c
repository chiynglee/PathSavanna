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

// hvm���� �����ΰ� ����� ���� event channel �� �ڵ�� ����� �����Ѵ�.
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

// hvm Domain�� id�� event channal�� ��Ʈ ��ȣ�� �̿��� �����Ͽ� irq ��ȣ�� ��ȯ�޴´�.
int savanna_bind_evtchn_to_irq(unsigned int remote_domain, unsigned int remote_port) {
	struct evtchn_bind_interdomain bind_interdomain;
	int err;

	bind_interdomain.remote_dom  = remote_domain;
	bind_interdomain.remote_port = remote_port;

	err = HYPERVISOR_event_channel_op(EVTCHNOP_bind_interdomain, &bind_interdomain);

	return err ? : bind_evtchn_to_irq(bind_interdomain.local_port);
}

// bitmap���κ��� ���� �޸��� reference number�� �����ϴ� �Լ�
// ������������ ���ĵǸ�, hvm�� �̸� �����Ѵ�.
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

	// bitmap page ����
	gnt->area = alloc_vm_area(PAGE_SIZE);
	if(gnt->area)
		// cylee : change the function name
		set_map_op(&map_op, (unsigned long)gnt->area->addr, GNTMAP_host_map, gnt->ref, remote_domid);

	if (HYPERVISOR_grant_table_op(GNTTABOP_map_grant_ref, &map_op, 1))
		BUG();
	gnt->handle = map_op.handle;

	// ���� ���� �� ȣ��
	if(map_op.status != GNTST_okay)
	{
		xen_free_vm_area(gnt->area);
		printk("map_op.status = %d\n", map_op.status);
		return -1;
	}

	// bitmap page ����
	page = (unsigned int*)(gnt->area->addr);

	// �����ϱ� ���� �޸� ���� Ȯ��
	gntpg_list->gntpg_shareMem = (gntpg_t *)vmalloc(sizeof(gntpg_t)*shareMem_num);

	// refernece number�� ����, ������ mapping_space_bw_doms()���� ����.
	for(bit = 0, count = 0 ; bit < BITMAP_LENGTH; bit++){
		if ((page[bit / 32] & bitTable[bit % 32]) == bitTable[bit % 32]) {
			gntpg_list->gntpg_shareMem[count].ref = shareMem_minVal + bit;
			count++;
		}
		if (count == shareMem_num) break;
	}

	return 0;
}

// ���� �޸𸮸� �����ϴ� �Լ�
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

// ���� �޸� ������ �����ϴ� �Լ���.
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

	gnt_bm = &(gntpg_list->gntpg_bitmap); // bitmap page�� �����ϱ� ����
	gnt_sm = gntpg_list->gntpg_shareMem; // ���� �޸� �����ϱ� ����
	remote_domid = gntpg_list->domid; // hvm�� domain id
	shareMem_num = gntpg_list->shareMem_num; //���ε� ���� �޸� ��
	bitmep_ref = gntpg_list->gntpg_bitmap.ref; // bitmap�� ���� �޸��� id

	printk("--------------------- unmapped page ----------------------\n");
	printk("Remote domain id : %d, ", remote_domid);
	printk("Shared momery count : %d\n", shareMem_num);

	// bitmap page ����
	// cylee : change the function name
	set_unmap_op(&unmap_op_bm, (unsigned long)gnt_bm->area->addr, GNTMAP_host_map, gnt_bm->handle);
	gnt_bm->handle = ~0;

	HYPERVISOR_grant_table_op(GNTTABOP_unmap_grant_ref, &unmap_op_bm, 1);
	xen_free_vm_area(gnt_bm->area);
	printk("savanna : memory unmapping(bitmap), bitmep ref = %d\n", bitmep_ref);

	// shared memory ����
	unmap_op_sm = (struct gnttab_unmap_grant_ref *)kzalloc(sizeof(struct gnttab_unmap_grant_ref)*shareMem_num, GFP_KERNEL);

	for(count = 0; count < shareMem_num; count++) {
		if(gnt_sm[count].area->addr != NULL) {
			// cylee : change the function name
			set_unmap_op(&(unmap_op_sm[count]), (unsigned long)(gnt_sm[count].area->addr), GNTMAP_host_map, gnt_sm[count].handle);
			gnt_sm[count].handle = ~0;
		}
	}

	// ������ ��
	HYPERVISOR_grant_table_op(GNTTABOP_unmap_grant_ref, unmap_op_sm, shareMem_num);

	// ����
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

// ���÷� ip�� ������ �����ϴ� �Լ�.
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

// ���� �޸𸮿� ����Ǿ� �ִ� ip �� hop������ ȣ���Ѵ�.
// _gntpg_list : ���� �޸� �ּ�, sm_num : ������ ������ ����
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

	for(count = 0; count < 300; count++) { // 100�� ���Ƿ� ���� ������ ���� ����
        ip = (unsigned int *)gnt_sm[count/1024].area->addr;
	    t = (unsigned char *)&(ip[count%1024]);
        printk("%3d   %3d.%3d.%3d.%3d - ", count, (int)t[3], (int)t[2], (int)t[1], (int)t[0]);
		ip = (unsigned int *)gnt_sm[sm_num/2+count/1024].area->addr; // +100������ ������ hop ���� ���
		printk("%d\n", ip[count%1024]);
	}
	return 0;
}

// hvm���κ��� ���� response�κ��� bitmap page�� reference number�� ���� �޸� ������ �õ��Ѵ�.
// ���ε� ���� �޸𸮸� ��ȯ�Ѵ�. savanna_AddSM()�� ȣ���ϸ� list�� �߰��ȴ�.
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

	gntpg_list->domid = _domid; // Dom G(Back-end)�� Domain ID
	gntpg_list->gntpg_bitmap.ref = resp->option[0]; // ���� �޸𸮵��� ref�� ����� Bitmap �������� ref�� ?!?!
	gntpg_list->gntpg_shareMem = NULL; // �ϴ� NULL, ���� �޸� �����ϴ� ���� �ʿ�
	gntpg_list->state = 1; // 1�� ��� ����, 0�� ��� �Ұ���
	gntpg_list->listId = resp->option[0]; // ���� �޸��� ���� �ĺ� �� = Bitmap �������� ref��
	gntpg_list->shareMem_num = resp->option[1]; // ���� �޸� �й� �� ������ ������ �� ����
	gntpg_list->shareMem_minVal = resp->option[2]; // ���� �޸� ref�� �ּҰ�
	gntpg_list->next = NULL; // ��������
	gntpg_list->pre = NULL; // �� �������� �̵��� �� ����

	err = trans_bitmap_to_refnum(gntpg_list); // bitmap���� ref �� ����
	if(err < 0)
	{
		unmapping_space_bw_doms(gntpg_list);
		return NULL;
	}

	err = mapping_space_bw_doms(gntpg_list); // ���� �޸� ����

	//printk("savanna : mapping new Shared Memory\n");
	//printk("Dom G ID : %d, ", gntpg_list->domid);
	//printk("Bitmap ref : %d, ", gntpg_list->gntpg_bitmap.ref);
	//printk("list ID : %d\n", gntpg_list->listId);
	//printk("Reference Count : %d, ", gntpg_list->shareMem_num);
	//printk("Minimum ref number : %d\n", gntpg_list->shareMem_minVal);

	return gntpg_list;
}

// ������ ���� �޸𸮸� ����Ʈ�� ���� ���� �����Ѵ�.
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

// ���� �޸��� ���� id�� �̿��� �ش� ���� �޸𸮸� �˻��Ѵ�.
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

// ��밡���� ���� �޸� �˻�.
// ������ null ��ȯ
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

// ��밡���� �����޸� ��ȯ.
// ������� hvm���κ��� �Ҵ�޾� ����, ���� ��쿡 ���� �����ɸ� �� �ִ�.
gntpg_list_t* savanna_getEnabledSM()
{
	gntpg_list_t *gntpg_list = savanna_SearchEnabledSM(&global_info->gntpg_head); // ���� ���� ������ ���� �޸� ���� �˻�, ������ null ��ȯ

	if(gntpg_list == NULL) { // null. ��, ��� ������ ���� �޸𸮰� ���� ��
		GPUTestState = 1; // ���� �ʱ�ȭ
		getnstimeofday(&setSMTest_S); // ���� �޸� Ȯ������ �ð� üũ, ���� 20ms(�̰� �� �� Ȯ���غ�����)
		send_request(REQ_SETSM, 0, 0, 0); // hvm���� ��û
		while(GPUTestState) { udelay(1); } // response�� �� ������ ���
		gntpg_list = savanna_SearchEnabledSM(&global_info->gntpg_head); // �ٽ� �˻�, �������� ����.
	}

	return gntpg_list;
}

// �����޸� ���� �� ���.
// ���� ���ִ°� �ƴϴ�. �ش� �������� ����Ʈ��
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

// ���� �Ҵ���� �����޸��� ������ ȣ���Ѵ�. 
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

