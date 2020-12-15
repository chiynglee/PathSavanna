#include <linux/inetdevice.h>
#include <linux/ip.h>
#include <net/arp.h>
#include <net/ip.h>
//nclude <net/dst.h>
#include <net/route.h>    /* struct rtentry */
#include <linux/in.h>	  /* sockaddr_in */
#include <linux/icmp.h>
#include <linux/sched.h>
#include <linux/proc_fs.h>


#define INADDR(a,b,c,d) (d<<24 | c<<16 | b<<8 | a)
#define IPV4_FORWARD_DAEMON	"pfinder:ipv4"

#define DIR24_POOL_COUNT		4
#define DIR24_CELL_LENGTH		4096

#define IPV4_ROUTE_THREADS		1

#define pfinder_graze_savanna(ip_list, tbl24) pass_to_gpgpu( ip_list, tbl24 );


struct pfinder_neigh_info {
	struct list_head	node;
	struct neighbour*	neigh;
};

struct pfinder_entry_ipv4 {
	struct list_head		node;
	__be32 				destination;
	__be32				gateway;
	u8					prefix;
	bool					is_local;
	struct neighbour*		neigh;

	struct net_device* netdev;

	int netdev_index;
};

struct pfinder_dir_entry {
	u16		tbl24:1,
			pool:3,
			cell:12;
};

struct pfinder_pool_entry{
	union {
		struct pfinder_entry_ipv4* pfinder_entry;
		struct pfinder_entry_ipv4** pfinder_long_tbl;
	};
};

// Function Forwarder
static inline int pfinder_ipv4_sysfs_show(struct seq_file *seq, void *v);
inline int pfinder_ipv4_sysfs_open(struct inode *inode, struct file *file);
inline ssize_t pfinder_ipv4_sysfs_write(struct file *file, const char __user * user_buffer, size_t count, loff_t *ppos);
long pfinder_ipv4_sysfs_ioctl(struct file *file, unsigned int cmd, unsigned long arg);
void pfinder_ipv4_register( void );
struct sk_buff* pfinder_ipv4_handler( struct sk_buff* skb, struct pfinder_ptype_set* ptype_set );
struct list_head* pfinder_ipv4_lookup( void* rt_info, struct list_head* rtable );
u16 pfinder_ipv4_pool_set( struct pfinder_entry_ipv4* entry );
struct pfinder_entry_ipv4* pfinder_ipv4_entry_alloc( __be32 daddr, __be32 gateway, u8 prefix, const char* if_name );
void* pfinder_ipv4_flush(struct pfinder_ptype_set* ptype_set);
struct sk_buff* pfinder_arp_handler( struct sk_buff* skb, struct pfinder_ptype_set* ptype_set );
u8 pfinder_ipv4_update( u16 cmd, struct pfinder_entry_ipv4* rt_info );

struct list_head* pfinder_get_ipv4_rtable(void);

void rt_copy_to_sm(gntpg_list_t* _target_gntpg, _client_rt_info *client_rt_info);
int get_next_hop_from_netdev(char *_dev_name);
int build_next_hop(_client_rt_info *dest_info, struct pfinder_entry_ipv4* source_info);
int make_sender_packet(_client_rt_info *dest_info, struct pfinder_entry_ipv4* source_info, unsigned int cmd );
u8 savanna_ipv4_update( u16 cmd, struct pfinder_entry_ipv4* rt_info );


// External Functions
extern int pfinder_ptype_register( struct pfinder_ptype_set* handler );
extern int pfinder_ptype_unregister( struct pfinder_ptype_set* handler );
//extern int pfinder_enqueue( struct sk_buff* skb, struct pfinder_queue* queue );
//extern struct sk_buff* pfinder_dequeue( struct pfinder_queue* queue );
extern struct pfinder_queue* pfinder_get_queue(void);

extern void icmp_send(struct sk_buff *skb_in, int type, int code, __be32 info);
extern struct pfinder_entry_ipv4** pass_to_gpgpu(__be32**, struct pfinder_dir_entry* );

// External Variables
extern struct proc_dir_entry	*pfinder_proc;
extern struct pfinder_queue	slowpath_queue;
extern struct list_head		queue_store;
extern struct list_head		pfinder_devices;

static struct proc_dir_entry		*ipv4_pe;
static struct list_head			ipv4_rtable;
static struct pfinder_dir_entry		tbl24[0x1000000];

static struct pfinder_pool_entry	*pfinder_dir_pool[DIR24_POOL_COUNT];
static u8 active_pool = 0;
static u8 active_cell = 0;
