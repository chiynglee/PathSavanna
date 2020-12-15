#include <linux/types.h>
#include <linux/kthread.h>
#include <linux/smp_lock.h>
#include <linux/rwsem.h>
#include <linux/list.h>
#include <linux/skbuff.h>
#include <linux/netdevice.h>
#include <linux/rtnetlink.h>
#include <linux/inet.h>
#include <linux/if.h>	//struct ifreq
#include <linux/sched.h>  //sched_setaffinity
#include <linux/ktime.h>

//header files for vlink
#include <linux/etherdevice.h>

#include <linux/ip.h>
#include <net/arp.h>
#include <net/ip.h>

#include "xbroute_compat.h"

//savanna_clnt_mod.h : gy add
#include "../savanna_clnt/savanna_clnt_mod.h"
#define NOTIFY_NAME		"Path Finder: "
#define VERSION			"0.0.5"

#define KU_OSL_VLINK
//#define PF_DEBUG

//gybak : define -> GPU use / not define -> GPU unuse
#define GPU_USE

#ifdef GPU_USE
#define PFINDER_CLEAN_BATCH_SIZE SAVANNA_IP_NUM
#define PFINDER_CLEAN_BATCH_MAX 2
#endif

#define CLEANTASK_CALL_INTERVAL 32

//for allocating xmit buffers
#define PFINDER_MAX_INTERFACES 3//12

#define PFINDER_MAX_QUEUES		255
#define PFINDER_QUEUE_LEN		102400
#define PFINDER_TX_LEN			65536//65536

#define PFINDER_DEV_ADV_BUFFER	0x01
#define PFINDER_DEV_RX_POLL		0x02
#define PFINDER_DEV_DD_FWD		0x04
#define PFINDER_DEV_RT_QUEUE	0x08

#ifdef KU_OSL_VLINK

#define ENCAP_BASE_UDP_PORT	50000
#define list_walk(pos, start)	\
	for (pos = start;				\
	       pos;				\
               pos = (pos)->next)

#endif

#define PFINDER_IOCTL_INTERRUPT		SIOCDEVPRIVATE+3
#define PFINDER_IOCTL_POLL			SIOCDEVPRIVATE+4
#define PFINDER_IOCTL_SLOWPATH	SIOCDEVPRIVATE+5

struct pfinder_dev_list;

struct ixgbevf_dma_buffer {
	u16	len;
	dma_addr_t dma;
	void* data;
};

struct ixgbevf_tx_dma_buffer {
	u16	 len;
	u32* status;
	dma_addr_t dma;
	void* data;
};

struct pfinder_buffer {
	union{
		struct ixgbevf_dma_buffer*  dma_buffer;
		struct sk_buff* skb;
	};
};

struct pfinder_queue {
	struct pfinder_buffer			rx_buffers[PFINDER_QUEUE_LEN];

#ifndef GPU_USE
	struct ixgbevf_tx_dma_buffer*	tx_buffers[PFINDER_TX_LEN];
	u64 		tx_head;
#endif

#ifdef GPU_USE
	struct ixgbevf_tx_dma_buffer*	batch_buffer[PFINDER_CLEAN_BATCH_MAX][PFINDER_CLEAN_BATCH_SIZE];	//gybak
	u32			bbuf_xmit_use;
	u32			bbuf_clean_use;
#endif
	struct pfinder_dev_list*		dev;
	u32			head;
	u32			tail;
	ktime_t		start_enqueue, last_enqueue;
	u64			total_packets;
	u64			drops;
};

struct pfinder_dev_list {
	struct list_head 		dev_list;
	struct net_device 		*netdev;
	struct task_struct		*clean_task;
	struct task_struct		*poll_task;
	struct pfinder_queue	*queue;
	u16					flag;
	int 					(*xmit)(struct net_device*, struct ixgbevf_tx_dma_buffer**, u64);
	void					(*slowpath_rx)(struct net_device* netdev, void* data, u32 len);
//	void					(*slowpath_rx)(struct net_device* netdev, void* data, u32 len, struct sk_buff* rxskb);
						
	//kllaf
	const struct net_device_ops* prev_netdev_ops;
	struct net_device_ops* new_netdev_ops;
	rx_handler_func_t* prev_rx_handler;

	__be32 represent_ipv4_addr;
	u8 vnet_id;
};

struct pfinder_ptype_set {
	struct list_head		node;
	__be16				ptype;
	struct sk_buff*		(*handler)(struct sk_buff*, struct pfinder_ptype_set*);
	void					(*init)(void);
	void*				(*flush)(struct pfinder_ptype_set*);
};

#ifdef KU_OSL_VLINK

struct encaphdr {
	struct ethhdr etherh;
	struct iphdr iph;
	struct udphdr udph;
} __attribute__((packed));

struct varp_entry {
	struct varp_entry* prev;						// prev entry on vARP table
	struct varp_entry* next;						// next entry on vARP table
	__be32 represent_dst_ipv4_addr;			// destination authoritative ipv4 address of which peer domain 0
	u8 nexthop_ethaddr[ETH_ALEN];				// next hop ethernet address for forwarding to next hop from source
	u8 dst_ethaddr[ETH_ALEN];					// ethernet address of domain U
};

struct varp_table {
	struct varp_entry* head;	  	// first entry of vARP
	struct varp_entry* tail;		  	// last entry of vARP
	u32 tbl_size;				 	// number of vARP entry
};

//struct varp_table varp_tbl;	 			// vARP table

struct varp_proto_t {
	__be32	ipaddr;
	u8		ethaddr[ETH_ALEN];
	union {
		u16		vnet_id;
	} other_info;
};

void pfinder_device_register( struct net_device* netdev, __be32 represent_ipv4, u8 vnet_id );
void pfinder_device_unregister( struct net_device* netdev, struct pfinder_dev_list* pf_dev );

static inline int pfinder_sysfs_show(struct seq_file *seq, void *v);
inline int pfinder_sysfs_open(struct inode *inode, struct file *file);
inline ssize_t pfinder_sysfs_write(struct file *file, const char __user * user_buffer, size_t count, loff_t *ppos);


//function definition for sysfs_write()
int virtlink_encap_skb(struct sk_buff* skb, struct net_device* netdev);
int virtlink_encap_dma_buffer(struct ixgbevf_tx_dma_buffer* tx_buffer, struct net_device* netdev);
struct sk_buff* virtlink_decap_skb(struct sk_buff* skb);
int virtlink_decap_dma_buffer(struct ixgbevf_tx_dma_buffer* dma_buffer, struct net_device* netdev); 

int virtlink_varp_add(__be32 dst_ipaddr, u8* nexthop_ethaddr, u8* dst_ethaddr);
int virtlink_varp_remove(u8* dst_ethaddr, struct varp_entry* entry);
int virtlink_varp_flush(void);

#endif
