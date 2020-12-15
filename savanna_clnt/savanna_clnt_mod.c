#include "savanna_clnt_mod.h"

irqreturn_t savanna_interrupt_pv(int irq, void *dev_id);
static int get_id_from_freelist(savannaClnt_info_t *info);
static void add_id_to_freelist(savannaClnt_info_t *info, unsigned long id);
static void create_savanna_kthread(savannaClnt_info_t *info);
int savanna_front_schedule(void *arg);
static int do_savanna_op(savannaClnt_info_t *info);
static void do_response_op(savanna_response_t *_resp);
static int map_ring_page(savannaClnt_info_t *info, unsigned long ring_ref);
void unmap_ring_page(savannaClnt_info_t *info);
int connect_ring(savannaClnt_info_t *info, unsigned long ring_ref, unsigned int evtchn);
int savannaFront_probe(domid_t otherend_id, unsigned long ring_ref, unsigned int evtchn);
static void savanna_front_free(savannaClnt_info_t *info);
static int savanna_front_remove(savannaClnt_info_t *info);

// cylee : add function declaration
extern int pfinder_init(void);
//extern void export_test(savanna_response_t *resp); //gybak : pfinder_main.c:export_test
extern void export_test(gntpg_list_t* target_gntpg, int req_dev_ifindex, int xmit_size);	//gybak : pfinder_main.c:export_test


int call_open(struct inode *inode, struct file *filp){
	//printk("test module call_open\n");
	return 0;
}

int call_release(struct inode *inode, struct file *filp){
	//printk("test module call_release\n");
	return 0;
}

irqreturn_t savanna_interrupt_pv(int irq, void *dev_id)
{
	savannaClnt_info_t *info;

	info = (savannaClnt_info_t *) dev_id;
	info->waiting_msgs = 1;
	wake_up(&info->wq);

	return IRQ_HANDLED;
}

static int get_id_from_freelist(savannaClnt_info_t *info)
{
	unsigned long free = info->free_id;
	BUG_ON(free >= RSHM_RING_SIZE);
	info->free_id = info->sended_reqs[free].id;
	info->sended_reqs[free].id = 0x0fffffee; /* debug */
	return free;
}

static void add_id_to_freelist(savannaClnt_info_t *info, unsigned long id)
{
	info->sended_reqs[id].id  = info->free_id;
	info->free_id = id;
}

// cylee : remove "static" for external calling from pfinder
// hvm로의 요청
void send_request(uint16_t _operation, uint32_t _option_1, uint32_t _option_2, uint32_t _option_3)
{
	savanna_request_t *req;
	int id;
	int notify;

	if (global_info == NULL) {
		printk("globla_info is NULL\n");
		return;
	}

	//printk("send_request\n");
	spin_lock_irq(&global_info->ring_lock);

	req = RING_GET_REQUEST(&global_info->ring, global_info->ring.req_prod_pvt);

	id = get_id_from_freelist(global_info);

	// Request 설정
	req->id = id;
	req->operation = _operation;
	req->option[0] = _option_1;
	req->option[1] = _option_2;
	req->option[2] = _option_3;

	memcpy(&global_info->sended_reqs[id], req, sizeof(req));

	global_info->ring.req_prod_pvt++;

	spin_unlock_irq(&global_info->ring_lock);

	RING_PUSH_REQUESTS_AND_CHECK_NOTIFY(&global_info->ring, notify);

	if (notify)
		notify_remote_via_irq(global_info->irq);

	//printk("notify to backend\n");
	return;
}

static void create_savanna_kthread(savannaClnt_info_t *info)
{
	int err;
	char name[TASK_COMM_LEN];

#ifdef GNT_DEBUG
	printk(KERN_DEBUG "connect_ring_evt\n");
#endif

	if (info == NULL)
		return;

	// Not ready to connect?
	if (!info->irq)
		return;

	// Already connected?
	if (info->savanna_evt_d)
		return;

	//gybak : thread binding
	snprintf(name, TASK_COMM_LEN, "savannafront.to.%d", info->domid);
	info->savanna_evt_d = kthread_create(savanna_front_schedule, info, name);
	kthread_bind(info->savanna_evt_d, 2);
	wake_up_process(info->savanna_evt_d);


/*
	snprintf(name, TASK_COMM_LEN, "savannafront.to.%d", info->domid);
	info->savanna_evt_d = kthread_run(savanna_front_schedule, info, name);
*/

	if (IS_ERR(info->savanna_evt_d)) {
		printk("create_savanna_kthread fail\n");
		err = PTR_ERR(info->savanna_evt_d);
		info->savanna_evt_d = NULL;
	}
}

int savanna_front_schedule(void *arg)
{
	savannaClnt_info_t *info = arg;

	while (!kthread_should_stop()) {
		if (try_to_freeze())
			continue;

		wait_event_interruptible(
			info->wq,
			info->waiting_msgs || kthread_should_stop());

		info->waiting_msgs = 0;
		smp_mb(); // clear flag

		if (do_savanna_op(info))
			info->waiting_msgs = 1;
	}

	info->savanna_evt_d = NULL;
	
	return 0;
}

static int do_savanna_op(savannaClnt_info_t *info)
{
	savanna_response_t *resp;
	RING_IDX cons, rp;
	//int error = 0;
	int more_to_do = 0;

	cons = info->ring.rsp_cons;
	rp = info->ring.sring->rsp_prod;
	rmb(); // Ensure we see queued responses up to 'rp'.

	while(cons != rp) {
		if (kthread_should_stop()) {
			more_to_do = 1;
			break;
		}

		resp = RING_GET_RESPONSE(&info->ring, cons);
		info->ring.rsp_cons = ++cons;

		add_id_to_freelist(info, resp->id);

		do_response_op(resp);

		cond_resched();
	}

	if (cons != info->ring.req_prod_pvt) {
		RING_FINAL_CHECK_FOR_RESPONSES(&info->ring, more_to_do);
	}
	else
		info->ring.sring->rsp_event = cons + 1;

	return more_to_do;
}

// hvm으로부터 전송되는 응답에 대한 동작 실행.
// option[]을 통해 추가 데이터를 전달 받을 수 있다.
static void do_response_op(savanna_response_t *_resp)
{
	savanna_response_t *resp = _resp;
	gntpg_list_t *gntpg_list = NULL;

	switch (resp->operation) {
	case RESP_COMTEST:
		getnstimeofday(&comTest_E);
		printk(KERN_DEBUG "Communication test : time = %ld us\n", comTest_E.tv_nsec-comTest_S.tv_nsec);
		break;
	case RESP_SETSM:
		gntpg_list = savanna_CreateSM(global_info->domid, resp);
		savanna_AddSM(&global_info->gntpg_head, gntpg_list);
		getnstimeofday(&setSMTest_E);
		
		//intoosh
		if(PfinderInit <= NUMBER_SM)
		{
			if(PfinderInit == NUMBER_SM) 	//gy add
				pfinder_init();
			//	PfinderInit = 0;	//gy add
			
			PfinderInit ++;

		}
	
		/*
		// cylee: pfinder initialization and state change
		if(PfinderInit != 0) {	//gy add
			pfinder_init();
			PfinderInit = 0;	//gy add
		}
		*/
		GPUTestState = 0;	//gy add
		//printk(KERN_DEBUG "Setup SM test : time = %ld\n", setSMTest_E.tv_nsec-setSMTest_S.tv_nsec);
		//savanna_PrintSM_info(&global_info->gntpg_head);
		break;
	case RESP_RELEASE:
		if(resp->option[0] == 1) {
			//printk("savanna : pv release success\n");
		} else {
			printk("savanna : pv release fail\n");
		}
		break;

	case RESP_UPDATE_RT:
		gntpg_list = savanna_SearchSM(&global_info->gntpg_head, resp->option[0]);
		getnstimeofday(&(gntpg_list->gpuLaunch[1]));
		printk(KERN_DEBUG "routing table update test : time = %7ld ns ( shared memory id : %5d ), Datainput time = %7ld ns", gntpg_list->gpuLaunch[1].tv_nsec-gntpg_list->gpuLaunch[0].tv_nsec, gntpg_list->listId, gntpg_list->datainput[1].tv_nsec-gntpg_list->datainput[0].tv_nsec);
		gntpg_list->state = 1;
	
		break;
		
	case RESP_GPULAUNCH:
#if GY_TIME_ESTIMATE
		sr_rettime = ktime_get();
		sr_delta = ktime_sub(sr_rettime, sr_calltime);
		sr_duration = (unsigned long long) ktime_to_ns(sr_delta) >> 10;
//		printk(KERN_DEBUG "send to savanna_clnt_recieve : %lld usecs\n", duration);
		rs_calltime = ktime_get();
#endif
		gntpg_list = savanna_SearchSM(&global_info->gntpg_head, resp->option[0]);
		getnstimeofday(&(gntpg_list->gpuLaunch[1]));
//		print_data(gntpg_list, gntpg_list->shareMem_num);
		gntpg_list->state = 1;
//		printk(KERN_DEBUG "GPU Launch test : time = %7ld ns ( shared memory id : %5d )\n", gntpg_list->gpuLaunch[1].tv_nsec-gntpg_list->gpuLaunch[0].tv_nsec, gntpg_list->listId);

		sync_flag = 1;	//gy : sync_flag nnhet(e pfinder can get the control)

//		print_data(gntpg_list, gntpg_list->listId );
#if GY_DEBUG
		printk(KERN_DEBUG "response options, 0 : %d, 1 : %d, 2 : %d\n", resp->option[0], resp->option[1], resp->option[2]);
#endif
		export_test(gntpg_list, resp->option[1], resp->option[2]);
//		export_test(resp);
//		sync_flag=1;  //gy : sync_flag set(the pfinder can get the control)

		break;
	default:
		BUG();
	}
}


static int map_ring_page(savannaClnt_info_t *info, unsigned long ring_ref)
{
	struct gnttab_map_grant_ref op;

	// cylee : change the function name
	set_map_op(&op, (unsigned long)info->ring_area->addr,
					GNTMAP_host_map, ring_ref, info->domid);

	if (HYPERVISOR_grant_table_op(GNTTABOP_map_grant_ref, &op, 1))
		BUG();

	if (op.status) {
		printk("map_ring_page() : Grant table operation failure !\n");
		return op.status;
	}

	info->shmem_ref = ring_ref;
	info->shmem_handle = op.handle;

	return 0;
}

void unmap_ring_page(savannaClnt_info_t *info) 
{
	struct gnttab_unmap_grant_ref unmap_op;

	// cylee : change the function name
	set_unmap_op(&unmap_op, (unsigned long)info->ring_area->addr, GNTMAP_host_map, info->shmem_handle);
	info->shmem_handle = ~0;

	if (HYPERVISOR_grant_table_op(GNTTABOP_unmap_grant_ref, &unmap_op, 1))
		BUG();

	printk("ring page unmmaping\n");
}

int connect_ring(savannaClnt_info_t *info, unsigned long ring_ref, unsigned int evtchn)
{
	int err;
	struct savanna_sring *sring;

	if (info->irq)
		return 0;

	if ( (info->ring_area = alloc_vm_area(PAGE_SIZE)) == NULL )
		return -ENOMEM;

	err = map_ring_page(info, ring_ref);
	if (err) {
		free_vm_area(info->ring_area);
		return err;
	}

	sring = (struct savanna_sring *)info->ring_area->addr;
	FRONT_RING_INIT(&info->ring, sring, PAGE_SIZE);

	err = savanna_bind_evtchn_to_irqhandler(info->domid, evtchn, savanna_interrupt_pv, 0, "savanna-frontend", info);
	if (err < 0)
	{
		unmap_ring_page(info);
		free_vm_area(info->ring_area);
		info->ring.sring = NULL;
		return err;
	}
	info->irq = err;

	return 0;
}

// hvm과의 연결 및 ring 초기화 & 쓰레드 생성.
int savannaFront_probe(domid_t otherend_id, unsigned long ring_ref, unsigned int evtchn)
{
	int i;
	savannaClnt_info_t *info;

	info = kzalloc(sizeof(savannaClnt_info_t), GFP_KERNEL);
	if (!info) {
		printk("savannaFront_probe() : kzalloc failed\n");
		return -ENOMEM;
	}

	global_info = info;

	info->domid = otherend_id;
	spin_lock_init(&info->ring_lock);
	init_waitqueue_head(&info->wq);
	info->savanna_evt_d = NULL;

	for (i = 0; i < RSHM_RING_SIZE; i++)
		info->sended_reqs[i].id = i+1;
	info->sended_reqs[RSHM_RING_SIZE-1].id = 0x0fffffff;

	connect_ring(info, ring_ref, evtchn);

	create_savanna_kthread(info);

	return 0;
}

static void savanna_front_free(savannaClnt_info_t *info)
{
	if (info->savanna_evt_d) {
		kthread_stop(info->savanna_evt_d);
		info->savanna_evt_d = NULL;
	}

	if (info->irq) {
		unbind_from_irqhandler(info->irq, info);
		info->irq = 0;
	}
	
	spin_lock_irq(&info->ring_lock);

	unmap_ring_page(info);
	info->ring.sring = NULL;

	spin_unlock_irq(&info->ring_lock);

#ifdef GNT_DEBUG
	printk("savanna frontend driver freed\n");
#endif

}

static int savanna_front_remove(savannaClnt_info_t *info)
{
	gntpg_list_t *gntpg_list_temp;

	if(!info)
		return 0;
	
	while(info->gntpg_head != NULL)
	{
		gntpg_list_temp = savanna_DelSM(&info->gntpg_head, info->gntpg_head->listId);
		unmapping_space_bw_doms(gntpg_list_temp);
	}

	savanna_front_free(info);

	global_info = NULL;

	kfree(info);
	return 0;
}

long call_ioctl(struct file *filp, unsigned int cmd, unsigned long arg){
	savanna_bind_info *info = NULL;
	gntpg_list_t *gntpg_list = NULL;
	int result = 0;
	int err = 0;
	int count;

	switch (cmd) {
	case IOCTL_BIND: // hvm과의 연결
		info = (savanna_bind_info *)arg;
		if(info == NULL) {
			printk("info is null\n");
			result = -1;
			break;
		}

		result = savannaFront_probe(info->otherend_id, info->ring_ref, info->evtchn_port);
		getnstimeofday(&setSMTest_S);
		for(count = 0 ; count < NUMBER_SM ; count++)
			send_request(REQ_SETSM, 0, 0, 0); // 처음 pv연결 시 공유 메모리가 없기 때문에 공유 메모리를 요청한다.

		break;
	case IOCTL_BUILD_TEST: // hvm과의 통신 테스트
		getnstimeofday(&comTest_S);
		send_request(REQ_COMTEST, 0, 0, 0);
		result = 0;
		break;
	case IOCTL_SETSM: // hvm과의 공유 메모리 확보 테스트
		getnstimeofday(&setSMTest_S);
		send_request(REQ_SETSM, 0, 0, 0);
		result = 0;
		break;
	case IOCTL_GPULAUNCH: // GPU 구동 테스트
		/*gntpg_list = savanna_SearchEnabledSM(&global_info->gntpg_head);
		if(gntpg_list == NULL) {
			GPUTestState = 1;
			getnstimeofday(&setSMTest_S);
			send_request(REQ_SETSM, 0, 0, 0);
			while(GPUTestState) { udelay(1); }
			gntpg_list = savanna_SearchEnabledSM(&global_info->gntpg_head);
		}*/
		gntpg_list = savanna_getEnabledSM(); // 사용가능한 공유 메모리 검색
		
		err = create_sample_ip(gntpg_list, SAVANNA_IP_NUM);
		//err = print_data(gntpg_list, 100);

		getnstimeofday(&(gntpg_list->gpuLaunch[0]));
		send_request(REQ_GPULAUNCH, gntpg_list->listId, 0, 0);
		result = 0;
		break;
	case IOCTL_RELEASE: // 연결 해제 -> 아직 사용 x
		result = savanna_front_remove(global_info);
		//send_request(REQ_RELEASE, 0, 0, 0);
		break;
	default:
		printk("insert noncorrect command\n");
		result = -1;
		break;
	}

	return result;
}

struct file_operations call_fops = {
	.owner          = THIS_MODULE,
	.unlocked_ioctl = call_ioctl,
	.open           = call_open,
	.release        = call_release,
};

// cylee: module integration
//int my_init_module(void)
int init_GPGPU_module(void)
{
	int result;

	result = register_chrdev(CALL_DEV_MAJOR, CALL_DEV_NAME, &call_fops);

	if (result < 0){
		printk("fail to init module: %d\n", result);
		return result;
	}

	if (!xen_domain())
		return -ENODEV;

	printk("Savanna.. init module(savanna_front_init)\n");
	return 0; 
}

// cylee: module integration
//void my_cleanup_module(void)
void cleanup_GPGPU_module(void)
{
	if(global_info)
		savanna_front_remove(global_info);

	unregister_chrdev(CALL_DEV_MAJOR, CALL_DEV_NAME);
	printk("Savanna.. Mission Complete!!... (^0^)v\n");
}

// cylee: module integration
/*
module_init(my_init_module);
module_exit(my_cleanup_module);

MODULE_AUTHOR("Korea University Xebra Team <xebra@os.korea.ac.kr>");
MODULE_DESCRIPTION("Savanna Kernel Module - PV");
MODULE_LICENSE("GPL");
*/
