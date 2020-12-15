#include "savanna_serv_mod.h"

int send_signal_to_user(int sigpid, int command, int domid);
int send_signal_to_user_about_rouingtable(int sigpid, int command, int domid);
irqreturn_t savanna_interrupt_hvm(int irq, void *dev_id);
void create_savanna_kthread(savannaServ_info_t *info);
int savanna_back_schedule(void *arg);
int do_savanna_op(savannaServ_info_t *info);
static int do_request_op(savannaServ_info_t *_info, savanna_request_t *_req, savanna_response_t *_resp);
static void send_response(savannaServ_info_t *info, savanna_response_t *resp);
int setup_ring(savannaServ_info_t *info);
int savannaBack_probe(int otherend_id);
static void savanna_disconnect(savannaServ_info_t *info);
static void savanna_back_free(savannaServ_info_t *info);
static int savanna_back_remove(savannaServ_info_t *info);
int savanna_back_mmap(struct mem_ioctl_data *mid);
void savanna_back_unmmap(gntpg_list_t *gntpg_list, unsigned int num);

savannaServ_info_t *Head;
int gpuState;
int routingState;
unsigned int gpuLaunchTime;
struct timespec start, end;
struct timespec gpu_launch_S, gpu_launch_S2,gpu_launch_S3, gpu_launch_E;

int savanna_open(struct inode *inode, struct file *filp){
	//printk("test module call_open\n");
	return 0;
}

int savanna_release(struct inode *inode, struct file *filp){
	//printk("test module call_release\n");
	return 0;
} 

void savanna_vm_open(struct vm_area_struct* area){
	//printk("Savanna: savanna_vm_open... vma = 0x%p\n", area);
	return;
}

void savanna_vm_close(struct vm_area_struct* area){
	//printk("Savanna: savanna_vm_close\n");
	return;
}

struct page* savanna_nopage(struct vm_area_struct* vma, unsigned long address, int unused){
	//printk("Savanna: ERROR - savanna_nopage called\n");
	return NOPAGE_SIGBUS;
}


#define GET_PVID_SHIFT  16
#define MASK_SMLIST 0xFFFF 

int send_signal_to_user_about_rouingtable(int sigpid, int command, int domid)
{

	int combined_command, temp ; 
	combined_command = temp = 0;

	if(command > 1)
	{
		temp = domid << GET_PVID_SHIFT;
		combined_command = command | temp;
	}
	else
	{
		combined_command = command;
		printk("combined_command = %d \n", combined_command); 
	}

	
//	getnstimeofday(&(time[1]));
	
//	iprintk("sigpid %d, command %d , domid %d", sigpid, command, domid);
//	printk(" combined_command  %d\n", combined_command);
	
//	getnstimeofday(&(time[2]));

	sig_info_rt.si_value.sival_int = combined_command;

//	getnstimeofday(&(time[3]));
	
//	printk("send signal to user app2 = %d\n", combined_command);
	ret_rt = send_sig_info(SAVANNA_SIG_RT_NUM, &sig_info_rt, ts_rt);

//	getnstimeofday(&(time[4]));	
	
	//iprintk("send signal to user app 3 = %d\n", ret_rt);
	if(ret_rt< 0) {
		printk(KERN_NOTICE "error sending signal\n");
		return ret_rt;
	}
	cond_resched();
	
	
	return 0;

}



int send_signal_to_user(int sigpid, int command, int domid) 
{
	int combined_command, temp ; 
	combined_command = temp = 0;

	if(command > 1)
	{
		temp = domid << GET_PVID_SHIFT;
		combined_command = command | temp;
	}
	else
	{
		combined_command = command;
	//	printk("combined_command = %d \n", combined_command); 
	}
	
	sig_info.si_value.sival_int = combined_command;
	ret = send_sig_info(SAVANNA_SIG_NUM, &sig_info, ts);
	if(ret < 0) {
		printk(KERN_NOTICE "error sending signal\n");
		return ret;
	}
	cond_resched();
	return 0;
}

irqreturn_t savanna_interrupt_hvm(int irq, void *dev_id)
{
	savannaServ_info_t *info;

	info = (savannaServ_info_t *) dev_id;
	
	//printk("savanna : occur event from pv(id : %d)\n", info->domid);
	
	info->waiting_reqs = 1;
	wake_up(&info->wq);

	return IRQ_HANDLED;
}

void create_savanna_kthread(savannaServ_info_t *info) 
{
	char name[TASK_COMM_LEN];
	long ptr;

	// Not ready to connect?
	if (!info->irq) 
		return;

	snprintf(name, TASK_COMM_LEN, "savannaback.%d", info->domid);

	info->savanna_evt_d = kthread_run(savanna_back_schedule, info, name);
	if (IS_ERR(info->savanna_evt_d)) {
		printk("create_savanna_kthread fail\n");
		ptr = PTR_ERR(info->savanna_evt_d);
		info->savanna_evt_d = NULL;
	}
}

int savanna_back_schedule(void *arg)
{
	savannaServ_info_t *info = arg;

	savanna_get(info);
	
	while (!kthread_should_stop()) {
		//printk("while kthread_should_stop\n");
		if (try_to_freeze())
			continue;

		wait_event_interruptible(
			info->wq,
			info->waiting_reqs || kthread_should_stop());
		//printk("wait_event_interruptible\n");
		info->waiting_reqs = 0;
		smp_mb(); // clear flag
		//printk("smp_mb\n");
		if (do_savanna_op(info))
			info->waiting_reqs = 1;
	}

	info->savanna_evt_d = NULL;
	//printk("info_evt_d\n");
	savanna_put(info); 
	//printk("info_put\n");
	
	return 0;
}

int do_savanna_op(savannaServ_info_t *info)
{
	struct savanna_back_ring *ring = &info->ring;
	savanna_request_t req;
	savanna_response_t *resp;
	RING_IDX rc, rp;
	int more_to_do = 0;
	int err = 0;

	rc = ring->req_cons;
	rp = ring->sring->req_prod;
	rmb(); // Ensure we see queued requests up to 'rp'.

	//printk("do_savanna_op\n");
	while (rc != rp) {
		if (RING_REQUEST_CONS_OVERFLOW(ring, rc)) {
			//printk("RING_REQUEST_CONS_OVERFLOW\n");
			break;
		}

		if (kthread_should_stop()) {
			more_to_do = 1;
			break;
		}

		memcpy(&req, RING_GET_REQUEST(ring, rc), sizeof(req));
		ring->req_cons = ++rc;

		// Apply all sanity checks to /private copy/ of request.
		barrier();

		err = do_request_op(info, &req, resp);

		send_response(info, resp);

		// Yield point for this unbounded loop.
		cond_resched();
	}

	return more_to_do;
}

static int do_request_op(savannaServ_info_t *_info, savanna_request_t *_req, savanna_response_t *_resp)
{
	savannaServ_info_t *info = _info;
	savanna_request_t *req = _req;
	savanna_response_t *resp = _resp;
	struct savanna_back_ring *ring = &info->ring;
	gntpg_list_t *gntpg_list = NULL;
	int err = 0;

	switch (req->operation) {			
	case REQ_COMTEST:
		//printk("Do communication test with PV(id = %d)\n", info->domid);
		resp = RING_GET_RESPONSE(ring, ring->rsp_prod_pvt);
		ring->rsp_prod_pvt++;

		resp->id = req->id;
		resp->operation = RESP_COMTEST;
		return 0;
		break;
	case REQ_SETSM:
		printk("Do setup shared memory with PV(id = %d)\n", info->domid);
		resp = RING_GET_RESPONSE(ring, ring->rsp_prod_pvt);
		ring->rsp_prod_pvt++;

		gntpg_list = savanna_CreateSM(info->domid);
		send_signal_to_user(pid, gntpg_list->listId, info->domid);
		savanna_AddSM(&info->gntpg_head, gntpg_list);
		//savanna_PrintSM_info(&info->gntpg_head);

		resp->id = req->id;
		resp->operation = RESP_SETSM;
		resp->option[0] = gntpg_list->gntpg_bitmap.ref;
		resp->option[1] = gntpg_list->shareMem_num;
		resp->option[2] = gntpg_list->shareMem_minVal;
		return 0;
		break;
	case REQ_RELEASE:
		printk("Do release(PV id = %d)\n", info->domid);
		//resp = RING_GET_RESPONSE(ring, ring->rsp_prod_pvt);
		//ring->rsp_prod_pvt++;

		err = savanna_back_remove(info);

		//resp->id = req->id;
		//resp->operation = RESP_RELEASE;
		//resp->option[0] = 1;
		return -1;
		break;

		
	case REQ_LOOKUP_RT:
		printk("Do setup shared memory with PV(id = %d)\n", info->domid);
		resp = RING_GET_RESPONSE(ring, ring->rsp_prod_pvt);
		ring->rsp_prod_pvt++;
		


	case REQ_UPDATE_RT:
		
	//	printk("Do setup shared memory with PV(id = %d)\n", info->domid);
	//	printk("Get routing update table request 1\n");
		resp = RING_GET_RESPONSE(ring, ring->rsp_prod_pvt);
		ring->rsp_prod_pvt++;

		routingState = 1;
	//	getnstimeofday(&(time[0]));
			
	//	printk("Get routing update table request 2\n");
		send_signal_to_user_about_rouingtable(pid, req->option[0], info->domid);

	//	getnstimeofday(&(time[5]));
		
		
		while(routingState > 0) { udelay(1); }


	//	printk(KERN_DEBUG "sig merge command and pv_id : time = %7ld ns ", time[1].tv_nsec-time[0].tv_nsec);
	//	printk(KERN_DEBUG "printk test : time = %7ld ns ", time[2].tv_nsec-time[1].tv_nsec);
	//	printk(KERN_DEBUG "input combined_command test : time = %7ld ns ", time[3].tv_nsec-time[2].tv_nsec);
	//	printk(KERN_DEBUG "sigal to a test : time = %7ld ns ", time[4].tv_nsec-time[3].tv_nsec);
	//	printk(KERN_DEBUG "total time  : time = %7ld ns ", time[5].tv_nsec-time[0].tv_nsec);
		

		resp->id = req->id;
		resp->operation = RESP_UPDATE_RT;
		resp->option[0] = req->option[0]; 

				

		return 0;
		break;
		
		/*
		resp = RING_GET_RESPONSE(ring, ring->rsp_prod_pvt);
		ring->rsp_prod_pvt++;

		
		
		break;
		*/
		
		
	case REQ_GPULAUNCH:
	//	printk("Do GPU launch(PV id = %d)\n", info->domid);
		resp = RING_GET_RESPONSE(ring, ring->rsp_prod_pvt);
		ring->rsp_prod_pvt++;

		gpuState = 1;

	//	getnstimeofday(&gpu_launch_S); // <-- check point

		send_signal_to_user(pid, req->option[0], info->domid);

	//	getnstimeofday(&gpu_launch_S2); // <-- check point

		while(gpuState > 0) { udelay(1); }

	//	getnstimeofday(&gpu_launch_E); // <-- check point		

		resp->id = req->id;
		resp->operation = RESP_GPULAUNCH;
		resp->option[0] = req->option[0]; 
	//	resp->option[1] = gpu_launch_E.tv_nsec-gpu_launch_S.tv_nsec; 
	//	resp->option[2] = gpuLaunchTime; 
		return 0;
		break;
	default:
		msleep(1);
		//printk("error: unknown io operation [%d]\n", req->operation);
		return -1;
		break;
	}

	return -1;
}

static void send_response(savannaServ_info_t *info, savanna_response_t *resp)
{
	unsigned long     flags;
	struct savanna_back_ring *ring = &info->ring;
	int more_to_do = 0;
	int notify;

	//printk("send_response\n");
	spin_lock_irqsave(&info->ring_lock, flags);
	
	RING_PUSH_RESPONSES_AND_CHECK_NOTIFY(ring, notify);
	if (ring->rsp_prod_pvt == ring->req_cons) {
		// Tail check for pending requests. Allows frontend to avoid
		// notifications if requests are already in flight (lower
		// overheads and promotes batching).
		RING_FINAL_CHECK_FOR_REQUESTS(ring, more_to_do);
	} else if (RING_HAS_UNCONSUMED_REQUESTS(ring)) {
		more_to_do = 1;
	}

	spin_unlock_irqrestore(&info->ring_lock, flags);

	if (more_to_do) {
		info->waiting_reqs = 1;
		wake_up(&info->wq);
	}
	if (notify)
		notify_remote_via_irq(info->irq);
}

int setup_ring(savannaServ_info_t *info)
{
	struct savanna_sring *sring;
	int err;

	info->ring_ref = INVALID_REF;

	sring = (struct savanna_sring *)__get_free_page(GFP_NOIO | __GFP_HIGH);
	if (!sring) {
		printk("setup_ring() : shared ring allocation fail\n");
		return -ENOMEM; 
	}

	SHARED_RING_INIT(sring);
	BACK_RING_INIT(&info->ring, sring, PAGE_SIZE);

	err = set_grant_page(info->domid, info->ring.sring);
	if (err < 0) {
		free_page((unsigned long)sring);
		info->ring.sring = NULL;
		goto fail;
	}
	info->ring_ref = err;
	printk("setup_ring() : ring's reference number is %d\n", info->ring_ref);

	err = savanna_alloc_evtchn(&info->domid, &info->evtchn);
	if (err)
		goto fail;

	err = savanna_bind_evtchn_to_irqhandler(info->evtchn, savanna_interrupt_hvm, IRQF_SAMPLE_RANDOM, "savanna-backend", info);
	if (err <= 0) {
		printk("setup_ring() : bind evtchn to irq fail\n");
		goto fail;
	}
	info->irq = err;

	return 0;
fail:
	printk(" setup_ring () fail\n");

	savanna_back_free(info);
	return err;
}

int savannaBack_probe(int otherend_id)
{
	int err;
	savannaServ_info_t *info;

	if(savanna_SearchPV(&Head, otherend_id) == NULL)
		info = kzalloc(sizeof(savannaServ_info_t), GFP_KERNEL); // kzalloc = kmalloc + memset
	else 
		return -1;

	if(!info) {
		printk("savannaBack_probe() : kzalloc failed\n");
		return -ENOMEM; 
	}


	// 초기화
	memset(info, 0, sizeof(*info));
	info->domid = otherend_id;
	spin_lock_init(&info->ring_lock);
	atomic_set(&info->refcnt, 1);
	init_waitqueue_head(&info->wq);
	info->savanna_evt_d = NULL;
	info->gntpg_head = NULL;
	init_waitqueue_head(&info->waiting_to_free);
	info->next = NULL;
	info->pre = NULL;

	// I/O ring 셋팅
	err = setup_ring(info);
	if(err) {
		kfree(info);
		return err;
	}
	
	// I/O ring 감시용 쓰레드 생성 및 동작
	create_savanna_kthread(info);

	// PV 데이터 저장
	savanna_AddPV(&Head, info);
	savanna_PrintPV(&Head);

	return err;
}

static void savanna_disconnect(savannaServ_info_t *info)
{
	if(atomic_read(&info->refcnt) <= 0)
		return ;

	if (info->savanna_evt_d) {
		kthread_stop(info->savanna_evt_d);
		info->savanna_evt_d = NULL;
	}

	atomic_dec(&info->refcnt);
	wait_event(info->waiting_to_free, atomic_read(&info->refcnt) == 0);
	atomic_inc(&info->refcnt);

	if (info->irq) {
		unbind_from_irqhandler(info->irq, info);
		info->evtchn = info->irq = 0;
		printk("savanna : unbind_from_irqhandler()\n");
	}

	spin_lock_irq(&info->ring_lock);
	
	if (info->ring_ref != INVALID_REF) {
		gnttab_end_foreign_access(info->ring_ref, RW, (unsigned long)info->ring.sring);
		info->ring_ref = INVALID_REF;
		info->ring.sring = NULL;
	}

	spin_unlock_irq(&info->ring_lock);
}

static void savanna_back_free(savannaServ_info_t *info)
{
	if (!atomic_dec_and_test(&info->refcnt))
		BUG();

	kfree(info);
	printk("savanna : kfree()\n");
}

static int savanna_back_remove(savannaServ_info_t *info)
{
	gntpg_list_t *gntpg_list;
	gntpg_list_t *gntpg_temp;

	if(!info)
		return -1;
	
	gntpg_list = info->gntpg_head;

	//printk("savanna : released pv information(pv id : %d)\n", info->domid);

	while(info->gntpg_head != NULL) {
		gntpg_temp = savanna_DelSM(&info->gntpg_head, info->gntpg_head->listId);
		savanna_back_unmmap(gntpg_temp, gntpg_temp->shareMem_num);
		send_signal_to_user(pid, -1 * gntpg_temp->listId, 1);
		unmapping_space_bw_doms(gntpg_temp);
		//savanna_PrintSM_info(&info->gntpg_head);
	}
	
	printk("savanna : released all sm \n");
	
	savanna_disconnect(info);
	savanna_back_free(info);
	info = NULL;
	savanna_PrintPV(&Head);

	if(Head == NULL) 
	{
		printk("savanna : send_signal_to_user for close \n");
		send_signal_to_user(pid, SIG_CLOSE, 1);
		
	}

	return 0;
}

int savanna_back_mmap(struct mem_ioctl_data *mid)
{
	struct vm_area_struct* vmalist = current->mm->mmap;
	unsigned long requested_size = mid->requested_size;
	unsigned long start_of_vma = mid->start_of_vma;
	unsigned int listId = mid->sm_listId;
	savannaServ_info_t *temp_info = Head;
	gntpg_list_t *temp_sm_list;
	gntpg_t *temp_sm;
	int savanna_buf_count = 0;

	if(vmalist == NULL) {
		//printk("Savanna : vmalist is NULL\n");
		return -1;
	}

	while(temp_info != NULL) 
	{
		temp_sm_list = savanna_SearchSM(&temp_info->gntpg_head, listId);
		if(temp_sm_list != NULL)
			break;
		temp_info = temp_info->next;
	}

	if(temp_sm_list == NULL) {
		//printk("Savanna : shared memory is NULL (list id : %d)\n", listId);
		return -1;
	} else 
		temp_sm = temp_sm_list->gntpg_shareMem;

	for(; vmalist != NULL; vmalist = vmalist->vm_next){
		if(vmalist->vm_start == start_of_vma){
			if(vmalist->vm_private_data != NULL){
				int *unaligned_kptr;
				int *kptr;
				struct vm_area_struct* vma = vmalist;
				unsigned long size = vma->vm_end - vma->vm_start;
				int rv;

				while(temp_sm[savanna_buf_count].savanna_tmp_mem != NULL)
					savanna_buf_count++;

				unaligned_kptr = temp_sm[savanna_buf_count].page;
				if(unaligned_kptr == 0) {
					printk("Savanna: kmalloc failed (size = %ld)\n", requested_size);
					return -1;
				}

				kptr=(int *)(((unsigned long)unaligned_kptr) & PAGE_MASK);
				mem_map_reserve(virt_to_page((unsigned long)kptr));

				rv = remap_pfn_range(vma, vma->vm_start, virt_to_phys(kptr) >> PAGE_SHIFT, size, PAGE_SHARED);
				if(rv) {
					printk("Savanna: remap_pfn_range failed (rv = %d)\n", rv);
					return -5;
				}

				if(vma->vm_private_data == NULL){
					printk("Savanna: vm_private_data is null\n");
					return -1;
				} 

				temp_sm[savanna_buf_count].savanna_tmp_mem = (savanna_mem_mgr *)(vmalist->vm_private_data);
				temp_sm[savanna_buf_count].savanna_tmp_mem->address = (unsigned long)unaligned_kptr;
				temp_sm[savanna_buf_count].savanna_tmp_mem->size = requested_size;

				mid->return_offset = (((unsigned long)unaligned_kptr) & (PAGE_SIZE -1));
				return 0;
			} else {
				printk("Savanna: vmalist->vm_private_data is null\n");
				return -1;
			}
		}
	}

	printk("Savanna: couldn't find the vma\n");

	return -1;
}

void savanna_back_unmmap(gntpg_list_t *gntpg_list, unsigned int num)
{
	gntpg_list_t *temp_sm_list = gntpg_list;
	gntpg_t *temp_sm;
	unsigned int count;

	temp_sm = temp_sm_list->gntpg_shareMem;

	if(temp_sm == NULL)
		return;

	for(count = 0 ; count < num ; count++){
		mem_map_unreserve(virt_to_page((unsigned long)temp_sm[count].page));
		kfree(temp_sm[count].savanna_tmp_mem);
		//printk("shared memory unmap, pages[%d]\n", count);
	}

	return;
}

long savanna_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	int err = 0;
	int result = 0;
	int otherend_id;
	int *rtupdate_state = 0;
	int *iplookup_state = 0;
		
	switch (cmd) {
	case IOCTL_BUILD:
		otherend_id = (int)arg;
		err = savannaBack_probe(otherend_id);
		if(err < 0) {
			printk("IOCTL_BUILD fail\n");
			result = err;
			break;
		} else if(err == -1) {
			printk("domain(%d) is exist\n", otherend_id);
			result = err;
			break;
		}
		result = 0;
		break;
	case IOCTL_RELEASE:
		otherend_id = (int)arg;
		result = savanna_back_remove(savanna_DelPV(&Head, otherend_id));
		break;
	case IOCTL_SETPID:
		pid = (int)arg;
		printk("ioctl IOCTL_SET_PID : pid = %d\n", pid);
		memset(&sig_info, 0, sizeof(struct siginfo));
		sig_info.si_signo = SAVANNA_SIG_NUM;
		sig_info.si_code = SI_QUEUE;
		p = find_get_pid(pid);
		ts = pid_task(p, PIDTYPE_PID);
		if(ts == NULL ) {
			printk("no such pid %d\n", pid);
			return -ENODEV;
		}
		result = 0;
		break;
			
	case IOCTL_SETPID_RT:
		
		pid_rt = (int)arg;
		printk("ioctl IOCTL_SETPID_RT : pid = %d\n", pid_rt);
		memset(&sig_info_rt, 0, sizeof(struct siginfo));
		sig_info_rt.si_signo = SAVANNA_SIG_RT_NUM;
		sig_info_rt.si_code = SI_QUEUE;
		p_rt = find_get_pid(pid_rt);
		ts_rt = pid_task(p_rt, PIDTYPE_PID);
		if(ts_rt == NULL ) {
			printk("no such pid %d\n", pid_rt);
			return -ENODEV;
		}
		result = 0;
		
		break;
	case IOCTL_SIGTEST_START:
		getnstimeofday(&start);
		send_signal_to_user(pid, SIG_SIGTEST_START, 1);
		result = 0;
		break;
	case IOCTL_SIGTEST_END:
		getnstimeofday(&end);
		printk(KERN_DEBUG "signal test : start = %ld end = %ld diff = %ld\n", start.tv_nsec, end.tv_nsec, end.tv_nsec-start.tv_nsec);
		result = 0;
		break;
	case IOCTL_GPUCOMP:
	//	getnstimeofday(&gpu_launch_S3); // <-- check point
		gpuLaunchTime = (unsigned int)arg;
		gpuState = -1;
		result = 0;
		break;
	case IOCTL_RTCOMP:
		
		routingState = -1;
		result = 0;
		break;

	case IOCTL_IPLOOKUP_STATE:
		iplookup_state = (int *)arg;
		if(gpuState)
			*iplookup_state = 1;
		else 
			*iplookup_state = -1;

		break;
	case IOCTL_RTUPDATE_STATE:
		rtupdate_state = (int *)arg;
		if(routingState)
			*rtupdate_state = 1;
		else
			*rtupdate_state = -1;
		
		break;
	case IOCTL_CLOSE:
		send_signal_to_user(pid, SIG_CLOSE, 1);
		result = 0;
		break;
	case IOCTL_MMAP:
		result = savanna_back_mmap((struct mem_ioctl_data *)arg);
		break;
	default:
		//printk("insert noncorrect command\n");
		result = -1;
		break;
	}

	return result;
}

struct vm_operations_struct savanna_vm_ops = {
open: &savanna_vm_open,
close: &savanna_vm_close,
};

int savanna_mmap(struct file * filp, struct vm_area_struct *vma){
	struct savanna_mem_mgr* savanna_tmp_mem;

	savanna_tmp_mem = kmalloc(sizeof(struct savanna_mem_mgr), GFP_KERNEL);
	if(savanna_tmp_mem == NULL){
		printk("Savanna: temp buf kmalloc failed (size = %d)\n", (int)sizeof(struct savanna_mem_mgr));
		return -1;
	}

	if (vma->vm_private_data != NULL){
		printk("Savanna: error in vm_private_data (0x%p)\n", vma->vm_private_data);
		return -1;
	}

	vma->vm_private_data = savanna_tmp_mem;
	vma->vm_ops = &savanna_vm_ops;
	return 0;
}

struct file_operations call_fops = {
	.owner          = THIS_MODULE,
	.unlocked_ioctl = savanna_ioctl,
	.open           = savanna_open,
	.release        = savanna_release,
	.mmap           = savanna_mmap,
};

int savanna_init_module(void)
{
	int result;

	result = register_chrdev(CALL_DEV_MAJOR, CALL_DEV_NAME, &call_fops);

	if (result < 0){
		printk("fail to init module: %d\n", result);
		return result;
	}

	if (!xen_domain())
		return -ENODEV;

	gpuState = 0;
	routingState = 0;


	printk("Savanna.. init module(savanna_back_init)\n");

	return 0; 
}

void savanna_cleanup_module(void)
{
	unregister_chrdev(CALL_DEV_MAJOR, CALL_DEV_NAME);
	printk("Savanna.. Mission Complete!!... (^0^)v\n"); 
}

module_init(savanna_init_module);
module_exit(savanna_cleanup_module);

MODULE_AUTHOR("Korea University Xebra Team <xebra@os.korea.ac.kr>");
MODULE_DESCRIPTION("Savanna Kernel Module - HVM");
MODULE_LICENSE("GPL");
