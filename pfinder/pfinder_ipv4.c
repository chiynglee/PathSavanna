#include "pfinder.h"
#include "pfinder_ipv4.h"
#include "pfinder_sysfs.h"


static const struct file_operations pfinder_ipv4_fops = {
	.owner   = THIS_MODULE,
	.open    = pfinder_ipv4_sysfs_open,
	.write   = pfinder_ipv4_sysfs_write,
	.unlocked_ioctl = pfinder_ipv4_sysfs_ioctl,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

static struct pfinder_ptype_set ipv4_ptype_set = {
	.ptype = cpu_to_be16(ETH_P_IP),
	.init = pfinder_ipv4_register,
	.handler = pfinder_ipv4_handler,
	.flush = pfinder_ipv4_flush
};

static struct pfinder_ptype_set arp_ptype_set = {
	.ptype = cpu_to_be16(ETH_P_ARP),
	.handler = pfinder_arp_handler
};

static void pfinder_hh_init( struct neighbour *n, __be16 protocol )
{
	struct hh_cache *hh;
	struct net_device *dev = n->dev;

	for (hh = n->hh; hh; hh = hh->hh_next)
	        if (hh->hh_type == protocol)
	                break;

	if (!hh && (hh = kzalloc(sizeof(*hh), GFP_ATOMIC)) != NULL) {
		seqlock_init(&hh->hh_lock);
	        hh->hh_type = protocol;
	        atomic_set(&hh->hh_refcnt, 0);
	        hh->hh_next = NULL;

	        if (dev->header_ops->cache(n, hh)) {
	                kfree(hh);
	                hh = NULL;
	        } else {
	                atomic_inc(&hh->hh_refcnt);
	                hh->hh_next = n->hh;
	                n->hh       = hh;
	                if (n->nud_state & NUD_CONNECTED)
	                        hh->hh_output = n->ops->hh_output;
	                else
	                        hh->hh_output = n->ops->output;
	        }
	}
	if (hh) {
	        atomic_inc(&hh->hh_refcnt);
	}
}

struct list_head* pfinder_get_ipv4_rtable(){ return &ipv4_rtable; };

struct sk_buff* pfinder_ipv4_handler( struct sk_buff* skb, struct pfinder_ptype_set* ptype_set ){
	return NULL;
}

extern void* pfinder_direct_ipv4( void* data )
{
	struct ethhdr*			eth = (struct ethhdr*)data;
	struct iphdr*			iph = (struct iphdr*)((u8*)data + ETH_HLEN);
	struct list_head*		rt_node;
	struct pfinder_entry_ipv4* entry;
	struct neighbour* neigh = NULL;

#ifdef PF_DEBUG
printk("<1>" "destination addr : %02X:%02X:%02X:%02x:%02x:%02x\n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
printk("<1>" "ip daddr : %d", iph->daddr);
#endif

/*
	//decrease ttl
	if( --(iph->ttl) <= 0) {
		return NULL;
	}
*/	
	rt_node = pfinder_ipv4_lookup( &iph->daddr, &ipv4_rtable );

	if( likely(rt_node) ){
		entry = list_entry( rt_node, struct pfinder_entry_ipv4, node );
	} else {

#ifdef PF_DEBUG
		printk("<1>" "routing failure");
#endif
		return NULL;
	}
	

	if(entry->is_local == true) {
		//slow path packet
#ifdef PF_DEBUG
		printk("<1>" "to local slow path \n");
#endif
		goto out;
	}
	
	if(entry->gateway == 0) {
		neigh = __neigh_lookup(&arp_tbl, &iph->daddr, entry->netdev, true);
	} else {
		neigh = __neigh_lookup(&arp_tbl, &iph->daddr, entry->netdev, true);	
	}

	if(neigh != NULL) {
		if( !neigh->hh ){
			printk("<1>" "no hh cache\n");
			pfinder_hh_init( neigh, eth->h_proto );
		
		} else if( neigh->hh->hh_data ){
			int hh_len, hh_alen;
			hh_len = neigh->hh->hh_len;
			hh_alen = HH_DATA_ALIGN(hh_len);
		
			memcpy(data,((u8 *) neigh->hh->hh_data + 2), hh_len);
			struct ethhdr* ethx = (struct ethhdr*)data;
			goto out;
		}
	} else {
		printk("<1>" "no arp entry\n");
		return NULL;
	}

	memcpy(eth->h_source, neigh->dev->dev_addr, ETH_ALEN);
/*
	unsigned char next_mac[ETH_ALEN];
	next_mac[0] = 0xf2;
	next_mac[1] = 0x29;
        next_mac[2] = 0x9d;
        next_mac[3] = 0xbe;
        next_mac[4] = 0x12;
        next_mac[5] = 0x3a;
	memcpy(neigh->ha, next_mac, ETH_ALEN);
*/
	memcpy(eth->h_dest, neigh->ha, ETH_ALEN);
	neigh->dev->header_ops->cache(neigh, neigh->hh);

//	printk("<1>" "<eth header> \n");
//	printk("<1>" "%02X:%02x:%02x:%02x:%02x:%02x\n", neigh->ha[0], neigh->ha[1], neigh->ha[2], neigh->ha[3], neigh->ha[4], neigh->ha[5]);
//	printk("<1>" "destination addr : %02X:%02x:%02x:%02x:%02x:%02x\n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
//	printk("<1>" "source : %02x:%02x:%02x:%02x:%02x:%02x\n", eth->h_source[0], eth->h_source[1], eth->h_source[2], eth->h_source[3], eth->h_source[4], eth->h_source[5]);

out:
	return entry;
}

#ifdef RX_THREAD_UNUSED_SECTION
struct sk_buff* pfinder_ipv4_handler( struct sk_buff* skb, struct pfinder_ptype_set* ptype_set )
{
	
	u16 thread_id = (IPV4_ROUTE_THREADS==1)? 0 :
				   skb->dev->ifindex % IPV4_ROUTE_THREADS;
	
	struct pfinder_queue* queue;

	// Check if rx queue is currently available or not, this is caused by
	// possibility to change receive queue address by passing them to Savanna.
	queue = rt_threads[thread_id].rx_queue;
	if( pfinder_enqueue( skb, queue ) ){
		wake_up_process(rt_threads[thread_id].task);
	}
	else {
		kfree_skb(skb);
	}

	// Return slow path packet to kernel
	return pfinder_dequeue( &slowpath_queue );
}

int pfinder_ipv4_forward( void* info )
{
	struct pfinder_thread*	thread_info = info;
	struct sk_buff* 		skb;
	struct iphdr* 			iph;
	struct list_head*		rt_node;
	struct pfinder_queue*	tx_queue = thread_info->tx_queue;
	
	while( !signal_pending(current) && !kthread_should_stop() ){
		set_current_state(TASK_INTERRUPTIBLE);

		skb = pfinder_dequeue(thread_info->rx_queue);
		
		if( likely(skb) ){
			struct pfinder_entry_ipv4* entry;
			iph = ip_hdr(skb);
			
			rt_node = pfinder_ipv4_lookup( &iph->daddr, &ipv4_rtable );

			if( likely(rt_node) ) entry = list_entry( rt_node, struct pfinder_entry_ipv4, node );
			else {
				// Fail to lookup routing table
				kfree_skb(skb);
				goto do_tx;
			}
			
			// It is destined to local, we forward it to linux kernel stack otherwise,
			// it should be passed into fast routing part

			// Slow Path (heading to kernel)
			if( unlikely( entry->is_local ) ){
				pfinder_enqueue(skb, &slowpath_queue);
			}
			
			// Fast Path (Need to be forwarded to other interface)
			else if( likely(iph->ttl > 2) ){
				struct neighbour* neigh = entry->neigh;
				u8 send_tid =	 (IPV4_ROUTE_THREADS==1)? 0 : neigh->dev->ifindex % IPV4_ROUTE_THREADS;

				///////
				//	L3 Header processing
				///////
				iph->ttl--;
				iph->check = 0;	// checksum will be offloaded

/*
				///////
				//	L2 Device and hwaddr lookup
				///////
				if( unlikely(entry->gateway == 0) ){
					struct net_device*	dev = neigh->dev;
					u32 hash_val;
					struct neigh_hash_table *nht;
					nht = arp_tbl.nht;
					hash_val = arp_tbl.hash(&iph->daddr, neigh->dev, nht->hash_rnd) & nht->hash_mask;
					neigh = nht->hash_buckets[hash_val];
					
					if( unlikely(!neigh) ){
						neigh = __neigh_lookup(&arp_tbl, &iph->daddr, dev, 1);
						if( unlikely(!neigh) ){
							kfree_skb(skb);
							goto queue_flush;
						}
					}
				}

				// neighbour entry used time update
				neigh->used = jiffies;
				__neigh_event_send(neigh, NULL);

				///////
				//	Settings for IP checksum offload (if hw available)
				///////
				if( likely(neigh->dev->features & NETIF_F_IP_CSUM) ){
					// hardware checksum offload is available
					skb_set_transport_header(skb, iph->ihl << 2);
					skb->ip_summed = CHECKSUM_PARTIAL;
					if( iph->protocol == IPPROTO_TCP ) {
						tcp_hdr(skb)->check = 
							~csum_tcpudp_magic(iph->saddr, iph->daddr,
			                                         cpu_to_be16(iph->tot_len),  IPPROTO_TCP, 0);
					}
				} else {
					// hardware does not support xsum offload
					iph->check = ip_fast_csum(iph, iph->ihl);
					skb->ip_summed = CHECKSUM_COMPLETE;
				}
*/
				///////
				//	L2 Header Generation
				///////
				if( !neigh->hh ){
					pfinder_hh_init( neigh, skb->protocol );
 				} else if( neigh->hh->hh_data ){
					int hh_len, hh_alen;
					hh_len = neigh->hh->hh_len;
					hh_alen = HH_DATA_ALIGN(hh_len);
					memcpy(skb->data - hh_alen, neigh->hh->hh_data, hh_alen);
					skb_push(skb, hh_len);
					goto l2hdr_complete;
				}
				
				dev_hard_header(skb, neigh->dev, ntohs(skb->protocol), neigh->ha, NULL, skb->len );
				neigh->dev->header_ops->cache(neigh, neigh->hh);
				
l2hdr_complete:
				skb->pkt_type = PACKET_OUTGOING;
				skb->next = NULL;

				skb_set_dev(skb, neigh->dev);
				skb_set_queue_mapping(skb, 0);

				
				///////
				//	Queue for transmit skb
				///////
				pfinder_enqueue(skb, rt_threads[send_tid].tx_queue);
			}
			else {	
				// TTL over
				icmp_send(skb, ICMP_TIME_EXCEEDED, ICMP_EXC_TTL, 0);
				kfree_skb(skb);
			}
		}

do_tx:
		// Forwarding tx queue packets that are results of routing
		skb = pfinder_dequeue(tx_queue);
		if ( likely(skb) ) {
			do {
				int ret = NETDEV_TX_OK;
				struct netdev_queue *txq = netdev_get_tx_queue(skb->dev, 0);
				
				// Test if net device is up
				if ( likely(!netif_tx_queue_stopped(txq)) ) {
					// If it has a socket buffer to transmit
					while( skb && !netif_tx_queue_frozen(txq) ) {
						// Accquire a lock for device
						if( __netif_tx_trylock(txq) ){
							rcu_read_lock();
							set_bit(__QUEUE_STATE_FROZEN, &txq->state);
							ret = skb->dev->netdev_ops->ndo_start_xmit(skb, skb->dev);	
							clear_bit(__QUEUE_STATE_FROZEN, &txq->state);
							__netif_tx_unlock(txq);
							rcu_read_unlock();

							if( likely(ret == NETDEV_TX_OK) ) {
								txq_trans_update(txq);
							} else {
								dev_kfree_skb_any(skb);
							}
							
							skb = NULL;
						}
					}
				}
			}while( likely(skb = pfinder_dequeue(tx_queue)) );
		}
		else {
			usleep_range(1,1);
		 }
	}

	printk(KERN_INFO NOTIFY_NAME "IPv4 Forwarding Thread was stopped!\n");
	
	return 0;
}
#endif

u8 pfinder_ipv4_insert( __be32 dst_addr, u8 prefix, struct pfinder_entry_ipv4* rt_entry ) /* Created based on DIR-24-8 Algorithm */
{
printk("<1>" "ipv4_insert func called");
	u32 dst_network_base = cpu_to_be32(dst_addr) & ~((u64)0x0FFFFFFFF >> prefix); 	// Netmask applied
	u32 dst_network_max = dst_network_base | ((u64)0x0FFFFFFFF >> prefix); 		// Netmask applied maximum addr
	u32 tbl24_base = dst_network_base >> 8;		// Index for tbl24 - base
	u32 tbl24_max = dst_network_max >> 8;		// Index for tbl24 - largest
	u16 pool_cell = pfinder_ipv4_pool_set( rt_entry );	// alloc current rt_entry into memory pool, cell
	u32 i;

printk("<1>" "dst_network_base : %d\n", dst_network_base);
printk("<1>" "dst_network_max : %d\n", dst_network_max);
printk("<1>" "tbl24_base : %d\n", tbl24_base);
printk("<1>" "tble24_max : %d\n", tbl24_max);
printk("<1>" "pool_cell : %d\n", pool_cell);
	
	switch( prefix ){
		case 0 ... 24:		// When it can use tbl24, then use it.
			for( i = tbl24_base; i <= tbl24_max; i++ )
			{
				if(
					 *((u16*)&tbl24[i]) == 0xFFFF /* Initial step */ ||
					(tbl24[i].tbl24 == 1 && pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_entry->prefix < prefix ) 
				)
				{ // Current prefix is longer than before
					tbl24[i].tbl24 = 1;
					tbl24[i].pool = pool_cell >> 12;
					tbl24[i].cell = pool_cell & 0x0FFF;
					//printk(KERN_INFO NOTIFY_NAME "TBL24 Updated!\n");
				}
				else if(tbl24[i].tbl24 == 0 && pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl)
				{ // If the table is fragmented already, we should fill blank cells with our rule
					u8 tbl_idx = 0;
					for( tbl_idx=0;  tbl_idx<0xFF; tbl_idx++ )
					{
						if( pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl[ tbl_idx ] == NULL )		// If it has no routing destination,
							pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl[ tbl_idx ] = rt_entry;	// Mark it as setted entry
					}
				}
			}
		break;
		case 25 ... 32:		// If prefix length is larger than 24bits,
		{
			u8 tblng_base_offset = dst_network_base & 0xFF;						// Index for tblong - least 8 significant bits -  bcbase
			u8 tblng_max_offset = dst_network_max & 0xFF;						// Index for tblong - least 8 significant bits - largest
			
			if( tbl24[tbl24_base].tbl24 == 1 ){ 	// Need new memory allocation & initialization--> we should set all the other entry to default route entry ptr		
printk("<1>" "tbl24 ==1\n");
				// Save for rewriting rule
				struct pfinder_entry_ipv4* older_entry = (struct pfinder_entry_ipv4*)(pfinder_dir_pool[ tbl24[tbl24_base].pool ][ tbl24[tbl24_base].cell ].pfinder_entry);
printk("<1>" "old entry\n");
				
				// Memory allocation for extended table
				struct pfinder_entry_ipv4** pfinder_long_tbl = kcalloc( 0xFF, sizeof(struct pfinder_entry_ipv4*),  GFP_ATOMIC );
							
				tbl24[tbl24_base].tbl24 = 0;				// Mark that using extended table.
				tbl24[tbl24_base].pool = pool_cell >> 12;	// It's pointer is at this pool,
				tbl24[tbl24_base].cell = pool_cell & 0x0FFF;	// and this cell.
				
				// Loop every entry that overwrites a previous rule and a new rule
				for( i=0;  i<0xFF; i++ )
				{
					if( i >= tblng_base_offset && i <= tblng_max_offset )	// If a block is in the range of current network
						pfinder_long_tbl[i] = rt_entry;	// Mark it as setted entry
					else
						pfinder_long_tbl[i] = older_entry;	// In other case, we set it as allocated before.
				}

printk("<1>" "loop every entry");
				pfinder_dir_pool[ tbl24[tbl24_base].pool ][ tbl24[tbl24_base].cell ].pfinder_long_tbl = pfinder_long_tbl;
				
				//printk(KERN_INFO NOTIFY_NAME "TBLNG Updated!\n");
			}
			else {	// It was already allocated memories for each 255 entries.
printk("<1>" "already alloc\n");
				for( i=tblng_base_offset; i<=tblng_max_offset; i++ )							// Just override current table
					pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl[i] = rt_entry;		// Mark it as setted entry
					
				//printk(KERN_INFO NOTIFY_NAME "TBLNG Updated!\n");
			}
		}
		break;
	}

	return 1;
}

void pfinder_ipv4_delete( __be32 dst_addr, u8 prefix, struct pfinder_entry_ipv4* rt_entry )
{
	u32 dst_network_base = cpu_to_be32(dst_addr) & ~(0xFFFFFFFF >> prefix); 	// Netmask applied
	u32 dst_network_max = dst_network_base | (0xFFFFFFFF >> prefix); 		// Netmask applied maximum addr
	u32 tbl24_base = dst_network_base >> 8;		// Index for tbl24 - base
	u32 tbl24_max = dst_network_max >> 8;		// Index for tbl24 - largest
	u32 tblng_base_offset = dst_network_base & 0xFF;						// Index for tblong - least 8 significant bits -  bcbase
	u32 tblng_max_offset = dst_network_max & 0xFF;						// Index for tblong - least 8 significant bits - largest

	u32 i, j;
	

	for( i = tbl24_base; i <= tbl24_max; i++ ){ 	// make them initial state
		if( tbl24[i].tbl24 == 0 ){				// fragmented spaces for this entry
			u8 has_entry = 0;				// cummulate of table pointer
			for( j = tblng_base_offset; j <= tblng_max_offset; j++ ){ 
				if( pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl[j] == rt_entry ){
					// remove pointer only from it marked as routing to this entry
					pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl[j] = NULL;
				}
				has_entry |= (unsigned long)pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl[j];
			}

			if( has_entry == 0 ){ // cummulated result is 0 means, there's nothing in extended table
				// free of extended table
				kfree( pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl );
				
				pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl = NULL;  // entry re-initialize
				// (for futher improvement, in this point POOL, CELL will be useless)
				memset( &tbl24[i], 0xFF, sizeof( struct pfinder_dir_entry ));	     // tbl24 re-initialize
			}
		}
		else { // It's not fragmented table
			pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_entry = NULL; // entry re-initialize
			// (for futher improvement, in this point POOL, CELL will be useless)
			memset( &tbl24[i], 0xFF, sizeof( struct pfinder_dir_entry )); 	     // tbl24 re-initialize
		}
	}
}

void pfinder_ipv4_recovery( __be32 dst_addr, u8 prefix )
{
	struct pfinder_entry_ipv4* entry;
	__be32 dst_network = dst_addr & (0xFFFFFFFF >> (32-prefix)); 	// Netmask applied
	
	list_for_each_entry( entry, &ipv4_rtable, node ){
		__be32 comparison = entry->destination & dst_network;

		// if previous rt_entry overrides the rule to be deleted, then re-insert the entry
		if( comparison == entry->destination ){
			pfinder_ipv4_insert( entry->destination, entry->prefix, entry );
		}
	}
}

struct list_head* pfinder_ipv4_lookup( void* rt_info, struct list_head* rtable )
{
	u32	dst_addr = cpu_to_be32(*(__be32*)rt_info);
	u32 tbl24_idx = dst_addr >> 8;	// Index for tbl24 - base

	if ( *((u16*)&tbl24[tbl24_idx]) == 0xFFFF ){
		return NULL;
	} else if( tbl24[tbl24_idx].tbl24 ){
		return &(pfinder_dir_pool[ tbl24[tbl24_idx].pool ][ tbl24[tbl24_idx].cell ].pfinder_entry->node);
	} else if( pfinder_dir_pool[ tbl24[tbl24_idx].pool ][ tbl24[tbl24_idx].cell ].pfinder_long_tbl != NULL ){
		u8 tblng_idx = dst_addr & 0xFF;
		return &(pfinder_dir_pool[ tbl24[tbl24_idx].pool ][ tbl24[tbl24_idx].cell ].pfinder_long_tbl[tblng_idx]->node);
	}
	return NULL;
}

u8 pfinder_ipv4_update( u16 cmd, struct pfinder_entry_ipv4* rt_info )
{
	switch( cmd ){
		case SIOCADDRT: 
			printk( KERN_CRIT NOTIFY_NAME "Update Insertion Message!\n" );
			
			// DIR-24-8 table insertion			
			list_add_tail( &rt_info->node, &ipv4_rtable );
#ifdef GPU_USE
			savanna_ipv4_update(cmd,rt_info);
#else
			pfinder_ipv4_insert( rt_info->destination, rt_info->prefix,  rt_info );
#endif
		break;
		case SIOCDELRT:
		{
			struct pfinder_entry_ipv4* comp_entry;
			printk( KERN_CRIT NOTIFY_NAME "Update Deletion Message!\n" );

			// Search previous rt entry on raw ipv4_rtable
			list_for_each_entry( comp_entry, &ipv4_rtable, node ){
				if( comp_entry->destination == rt_info->destination && comp_entry->prefix == rt_info->prefix ) break;
			}
			
			if( comp_entry != NULL ){
				// DIR-24-8 table deletion
#ifndef GPU_USE
				pfinder_ipv4_delete( comp_entry->destination, comp_entry->prefix, comp_entry );
#endif
				// rt entry node removal
				list_del( &comp_entry->node );
				kfree( comp_entry );
			}
			
#ifdef GPU_USE
			savanna_ipv4_update(cmd,rt_info);
#else
			// DIR-24-8 table recovery (possibly, larger prefix may fill the blanks)
			pfinder_ipv4_recovery( rt_info->destination, rt_info->prefix );
#endif

			kfree( rt_info );
		}
		break;
	}
	
	return 0;
}

int get_next_hop_from_netdev(char *_dev_name)
{

	if(_dev_name == NULL)
		return -1;

//	printk("dev_name : %s \n", _dev_name);
	
	int nloop, ncur_index, netdev ;
	nloop = ncur_index = netdev = 0;

	while(!nloop)
	{
		switch(_dev_name[ncur_index])
		{
			case 'e':
				ncur_index ++;
				break;

			case 't':
				ncur_index ++;
				break;

			case 'h':
				ncur_index ++;
				nloop = 1;
				break;	
		}	
		
	}
	
	netdev = atoi(&_dev_name[ncur_index]);
	
		/*
		switch(_dev_name[nloop])
		{

			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':	
			case '8':
			case '9':
				ncur_index = nloop;
				break;

			default:
				ncur_index = 0;
				break;
		}	
		*/

	
	return netdev;
	
}


int build_next_hop(_client_rt_info *dest_info, struct pfinder_entry_ipv4* source_info)
{

	if(!source_info)
		return -1;
	
	char dev_name[MAX_NETDEV_NAME] = {0, };
	strcpy(dev_name, source_info->neigh->dev->name);
//	printk("dev_name : %s \n", dev_name);
	

	dest_info->nexthop = get_next_hop_from_netdev(dev_name);
	if(!dest_info->nexthop)
		return -1;

	return 1;
}


int make_sender_packet(_client_rt_info *dest_info, struct pfinder_entry_ipv4* source_info, unsigned int cmd )
{

	if(!source_info)
		return -1;

	if(!cmd)
		return -1;
	
	
	
	dest_info->cmd = cmd;
//	dest_info->pv_id = _pv_id;	
	dest_info->prefix = source_info->prefix;
	dest_info->dest_addr = source_info->destination;

	return 1;
}

void rt_copy_to_sm(gntpg_list_t* _target_gntpg, _client_rt_info *client_rt_info)
{

	if(!_target_gntpg)
		return ;

	if(!client_rt_info)
		return ;
	
	gntpg_t *_gnt_sm = NULL;
	_gnt_sm = _target_gntpg->gntpg_shareMem;

	//
	memcpy((_client_rt_info *)_gnt_sm[0].area->addr, (_client_rt_info *)client_rt_info, sizeof(_client_rt_info));
	//
//	memcpy(&global_info->sended_reqs[id], req, sizeof(req));
}

u8 savanna_ipv4_update(u16 cmd,struct pfinder_entry_ipv4 * rt_info){
		if(cmd == 0)
			return -1;
		if(!rt_info)
			return -1;
	
		_client_rt_info *client_rt_info = NULL;
		client_rt_info = kzalloc(sizeof(_client_rt_info), GFP_KERNEL);
		if(!client_rt_info)
			return -1;
	
		switch(cmd){
		case SIOCADDRT:
			make_sender_packet(client_rt_info,rt_info, INSERT_RT);
			client_rt_info->nexthop = rt_info->netdev->ifindex;
#if GY_DEBUG
			printk(KERN_INFO "nexthop input : %d\n",client_rt_info->nexthop);
#endif
			break;
		case SIOCDELRT:
			make_sender_packet(client_rt_info, rt_info, DELETE_RT);
			client_rt_info->nexthop = 0;
			break;
		}
	
		//보내는 것은 한 번에 보낸다. 
		// 공유 메모리에 write한다. 
	
	//	printk("done for entry\n");
	
		getnstimeofday(&upRTTest_S); // check point
		gntpg_list_t* _target_gntpg = NULL;
		_target_gntpg = savanna_getEnabledSM(); //gy add
		if(!_target_gntpg)
		{
			printk("SM is NULL\n");
			return -1;
		}
	//		  getnstimeofday(&(_target_gntpg->gpuLaunch[0])); // check point
		rt_copy_to_sm(_target_gntpg, client_rt_info);
		send_request(REQ_UPDATE_RT, _target_gntpg->listId, 0, 0); // GPU 구동 요청
	
		
		kfree(client_rt_info);
		
		
		
		return 0;
}

struct pfinder_entry_ipv4* pfinder_ipv4_entry_alloc( __be32 daddr, __be32 gateway, u8 prefix, const char* if_name )
{	
	struct pfinder_entry_ipv4* rt_info;

	rt_info = kzalloc( sizeof( struct pfinder_entry_ipv4 ), GFP_KERNEL );

	rt_info->destination = daddr & ~((u64)0x0FFFFFFFF << prefix);
	rt_info->gateway = gateway;
	rt_info->prefix = prefix;
	rt_info->is_local = false;

	if (if_name != NULL)	{
		rt_info->netdev = __dev_get_by_name(&init_net, if_name);

	}

	//initializing netdev_index
	//netdev_index is used for identifying each xmit buffer
	struct pfinder_dev_list* pf_dev = NULL;
	rt_info->netdev_index = -1;
	if( !list_empty( &pfinder_devices ) ){
		list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
			rt_info->netdev_index++;
			if( pf_dev->netdev == rt_info->netdev ) break;		
		}	
	}

	if(rt_info->netdev_index < 0) {
		printk("<1>" "pfinder_ipv4_entry_alloc function. <nedev_index error>\n");
	} else {
		printk("<1>" "pfinder_ipv4_entry_alloc function. <netdev_index : %d\n\n", rt_info->netdev_index);
	}

	
	printk("<1>" "dest : %d.%d.%d.%d\n", (rt_info->destination & 0xFF), (rt_info->destination & 0xFF00) >> 8, (rt_info->destination & 0xFF0000) >> 16, (rt_info->destination & 0xFF000000) >> 24);
	printk("<1>" "gateway : %d.%d.%d.%d\n", (rt_info->gateway & 0xFF), (rt_info->gateway & 0xFF00) >> 8, (rt_info->gateway & 0xFF0000) >> 16, (rt_info->gateway & 0xFF000000) >> 24);
	printk("<1>" "prefix : %d\n", rt_info->prefix);


	if( rt_info->destination == 0 ){
		printk("<1>" "dest = 0");
		kfree(rt_info);
		return NULL;
	}
	
	// getting neigh ptr
	rt_info->neigh = __neigh_lookup(&arp_tbl, &gateway, rt_info->netdev, true);
	printk("<1>" "%02X:%02x:%02x:%02x:%02x:%02x\n",rt_info->neigh->ha[0], rt_info->neigh->ha[1], rt_info->neigh->ha[2], rt_info->neigh->ha[3], rt_info->neigh->ha[4], rt_info->neigh->ha[5]);

	if( rt_info->neigh ){
		// arp lookup signal send
	//	neigh_event_send(rt_info->neigh, NULL);	
		atomic_dec(&rt_info->neigh->refcnt);
	}

	return rt_info;
}

struct sk_buff* pfinder_arp_handler( struct sk_buff* skb, struct pfinder_ptype_set* ptype_set )
{
	struct arphdr		*arph = arp_hdr(skb);
	struct list_head 	*rt_node;
	u8				*arp_ptr;
	__be32			sip, tip;
	u8				*sha;

	// ARP Information Extract
	arp_ptr = (unsigned char *)(arph + 1);
        sha     = arp_ptr;
        arp_ptr += skb->dev->addr_len;
        memcpy(&sip, arp_ptr, 4);
        arp_ptr += 4;
        arp_ptr += skb->dev->addr_len;
        memcpy(&tip, arp_ptr, 4);

	// We do proxy_arp for ethernet address should be directed here.
	switch( cpu_to_be16(arph->ar_op) ){
		case ARPOP_REQUEST:
		{
			// It is destined to local, we forward it to linux kernel stack otherwise,
			// it should be passed into fast routing part
			if( (rt_node = pfinder_ipv4_lookup(&tip, &ipv4_rtable)) ){
				struct pfinder_entry_ipv4* entry = list_entry( rt_node, struct pfinder_entry_ipv4, node );

				// ARP packet which is come from eth0 and destination is on eth0 then we do not proxy arp
				if( skb->dev == entry->neigh->dev && !strcmp( skb->dev->name, "eth0" ) )
					return skb;
				
				// if it has a routing entry make it for route next interface
				arp_send(ARPOP_REPLY, ETH_P_ARP,
						sip, skb->dev,
						tip,
						sha,
						skb->dev->dev_addr,
						sha );
			}
		}
		break;
		default:
		break;
	}

	// Kernel should have knowledge about arp received packet.
	return skb;
}

void pfinder_ipv4_pool_init( u8 pool ){
	pfinder_dir_pool[pool] = kcalloc( sizeof( struct pfinder_pool_entry ), DIR24_CELL_LENGTH, GFP_ATOMIC );
	active_pool = pool;
	active_cell = 0;
}

u16 pfinder_ipv4_pool_set( struct pfinder_entry_ipv4* entry ){
	if( active_cell >= DIR24_CELL_LENGTH ){
		pfinder_ipv4_pool_init( active_pool +1 );
	}

	pfinder_dir_pool[ active_pool ][ active_cell ].pfinder_entry= entry;
	
	return ((active_pool & 0x07) << 12) | (active_cell++ & 0x0FFF);
}


void pfinder_ipv4_register( void )
{
	struct pfinder_dev_list 	*pf_dev = NULL;
//	u8 idx;
	
	pfinder_ptype_register( &ipv4_ptype_set );
	pfinder_ptype_register( &arp_ptype_set );

	INIT_LIST_HEAD( &ipv4_rtable );

	/* Register into proc directory */
	ipv4_pe = proc_create("ipv4", 0600, pfinder_proc, &pfinder_ipv4_fops);
	if (ipv4_pe == NULL) {
		pr_err("ERROR: cannot create %s procfs xebra entry\n", "ipv4");
		return;
	}

	// tbl24 init
	memset( tbl24, 0xFF, 0x1000000 << 1 );

	// memory pool init
	pfinder_ipv4_pool_init(0);

	// Register local packets forwarding rule
	rtnl_trylock();
	list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
		struct net_device	*netdev = pf_dev->netdev;
		struct in_device 	*in_dev = __in_dev_get_rtnl(netdev);
		struct in_ifaddr 	*in_ifa;
		if( in_dev && in_dev->ifa_list ){
			in_ifa = in_dev->ifa_list;
			if( in_ifa->ifa_address != INADDR(127,0,0,1) ){
				struct pfinder_entry_ipv4* rt_info;
				u8 prefix = 32;
				while( ((cpu_to_be32(in_ifa->ifa_mask) >> (32-prefix)) & 1) == 0 ) prefix--;
				// Insert to routing table point to this interface on current device network.
				rt_info = pfinder_ipv4_entry_alloc( in_ifa->ifa_address, 0, prefix, netdev->name );
				if( rt_info ) {
					pfinder_ipv4_update( SIOCADDRT, rt_info );	
				}
				
				// Insert this interface address for heading local
				rt_info = pfinder_ipv4_entry_alloc( in_ifa->ifa_address, 0, 32, netdev->name );
				if( rt_info ){
					rt_info->gateway = rt_info->destination;
					rt_info->is_local = true;
					pfinder_ipv4_update( SIOCADDRT, rt_info );
				}
			}
		}
	}
	rtnl_unlock();
	printk(KERN_INFO NOTIFY_NAME "Default IPv4 Module Registered.\n");
}

void* pfinder_ipv4_flush(struct pfinder_ptype_set* ptype_set)
{
	u32 i;

	while( !list_empty(&ipv4_rtable) ){
		struct pfinder_entry_ipv4* rt_entry = list_entry(ipv4_rtable.next, struct pfinder_entry_ipv4, node);
		list_del(&rt_entry->node);
		kfree(rt_entry);
	}

	// DIR-24 Long prefix memory free
	for( i=0; i<0x1000000; i++){
		if( tbl24[i].tbl24 == 0 ) kfree( pfinder_dir_pool[ tbl24[i].pool ][ tbl24[i].cell ].pfinder_long_tbl );
	}

	// Memory Pool free
	for(i=0; i<=active_pool; i++){
		kfree( pfinder_dir_pool[i] );
	}

	// Proc Entry Deletion
	remove_proc_entry("ipv4", pfinder_proc);

	pfinder_ptype_unregister(ptype_set);
	pfinder_ptype_unregister( &arp_ptype_set );
	
	return 0;
}


/*
 * Function for sysfs
 */
static inline int pfinder_ipv4_sysfs_show(struct seq_file *seq, void *v)
{
	struct pfinder_entry_ipv4* entry;
//	int i;
	
	seq_puts(seq, "Debug Information\n");
/*
	seq_puts(seq, "  - Current Queue Status\n");
	for(i=0; i<IPV4_ROUTE_THREADS; i++){
		struct pfinder_queue* buf = rt_threads[i].rx_queue;
		seq_printf(seq, "rx queue [%d] @ len: %ld, drop: %ld\n", i, atomic64_read(&buf->qlen), atomic64_read(&buf->drop));
		buf = rt_threads[i].tx_queue;
		seq_printf(seq, "tx queue [%d] @ len: %ld, drop: %ld\n", i, atomic64_read(&buf->qlen), atomic64_read(&buf->drop));
	}
*/	
	
	seq_puts(seq, "\nIPv4 Routing Table\n");
	seq_puts(seq, "Destination \t Prefix \t Gateway \t Iface\n");
	
	list_for_each_entry(entry, &ipv4_rtable, node){
		seq_printf(seq, "%d.%d.%d.%d \t %4d \t\t %d.%d.%d.%d \t %s(%s)\n",
			entry->destination & 0x000000FF,
			(entry->destination & 0x0000FF00) >> 8,
			(entry->destination & 0x00FF0000) >> 16,
			(entry->destination & 0xFF000000) >> 24,
			entry->prefix,
			entry->gateway & 0x000000FF,
			(entry->gateway & 0x0000FF00) >> 8,
			(entry->gateway & 0x00FF0000) >> 16,
			(entry->gateway & 0xFF000000) >> 24,
			entry->neigh->dev->name,
			(entry->is_local)? "Slow" : "Fast"
		);
	}
	return 0;
}

inline int pfinder_ipv4_sysfs_open(struct inode *inode, struct file *file)
{
	return single_open(file, pfinder_ipv4_sysfs_show, PDE(inode)->data);
}

inline ssize_t pfinder_ipv4_sysfs_write(struct file *file, const char __user * user_buffer, size_t count, loff_t *ppos)
{
	#define CMD_ADD 1
	#define CMD_REMOVE 2
	#define CMD_FLUSH 3
	
	#define CMD_VARP_ADD 51
	#define CMD_VARP_REMOVE 53
	
	#define CMD_VLINK_FLUSH 54
		
	int cur_idx = 0;
	u8 command = 0;
	u8 if_name[IFNAMSIZ+5] = {0, };
	
	__be32 network = 0;
	__be32 gateway = 0;
	u8 prefix = 0;
	
	u8 dst_mac[ETH_ALEN] = {0, };
	u8 next_mac[ETH_ALEN] = {0, };
	__be32 dst_ip = 0;
	
	while( count > cur_idx ){
		char opt_name[40] = {0, };
		get_next_arg( user_buffer, opt_name, sizeof(opt_name), count, &cur_idx);
			
		if(!strcmp(opt_name, "add")){
			printk("<1>" "add command\n");
			command = CMD_ADD;
		}
		else if(!strcmp(opt_name, "remove")){
			command = CMD_REMOVE;
		}
		else if (!strcmp(opt_name, "dev")) {
			printk("<1>" "dev command\n");
			get_next_arg( user_buffer, if_name, sizeof(if_name), count, &cur_idx);

		} else if(!strcmp(opt_name, "VARP_ADD")) {
			command = CMD_VARP_ADD;

		} else if(!strcmp(opt_name, "VARP_REMOVE")) {
			command = CMD_VARP_REMOVE;

		} else if(!strcmp(opt_name, "vlinkFLUSH")){
			command = CMD_VLINK_FLUSH;

		} else if(!strcmp(opt_name, "dst_mac")){
			u8 hex_ethaddr[ETH_ALEN*3] = {0, };
			get_next_arg( user_buffer, hex_ethaddr, sizeof(hex_ethaddr), count, &cur_idx);
			hexmac_to_binmac( dst_mac, hex_ethaddr );

		} else if(!strcmp(opt_name, "next_mac")){
			u8 hex_ethaddr[ETH_ALEN*3] = {0, };
			get_next_arg( user_buffer, hex_ethaddr, sizeof(hex_ethaddr), count, &cur_idx);
			hexmac_to_binmac( next_mac, hex_ethaddr );

		} else if(!strcmp(opt_name, "dst_ip")){
			u8 str_ipaddr[4*4] = {0, };
			get_next_arg( user_buffer, str_ipaddr, sizeof(str_ipaddr), count, &cur_idx);
			dst_ip = in_aton(str_ipaddr);

		} else if(!strcmp(opt_name, "net")){
			printk("<1>" "net\n");
			u8 str_ipaddr[3*6+1] = {0, }; /*255.255.255.255/32*/
			u8 idx = 0;
			get_next_arg( user_buffer, str_ipaddr, sizeof(str_ipaddr), count, &cur_idx);
	
			while( str_ipaddr[idx] && (str_ipaddr[idx] != '/') ) idx++;
				
			if(idx > 0 && idx < sizeof(str_ipaddr)){ // If it was defined the prefix in the form of slash
				prefix = atoi(&str_ipaddr[idx+1]);
				str_ipaddr[idx] = 0;
			}
			if(prefix != 32) {
				network = in_aton(str_ipaddr) & (~htonl(0xFFFFFFFF >> prefix));
			} else {
				network = in_aton(str_ipaddr);
			}
			//printk("<1>" "%x", in_aton(str_ipaddr));
			printk("<1>" "network : %d.%d.%d.%d\n", (network & 0xFF), (network & 0xFF00) >> 8, (network & 0xFF0000) >> 16, (network & 0xFF000000) >> 24);

		} else if(!strcmp(opt_name, "prefix")){
			printk("<1>" "prefix");
			u8 str_prefix[3] = {0, };
			get_next_arg( user_buffer, str_prefix, sizeof(str_prefix), count, &cur_idx);
			prefix = atoi(str_prefix);

		} else if(!strcmp(opt_name, "gw")){
			u8 str_ipaddr[3*5+1] = {0, };
			get_next_arg( user_buffer, str_ipaddr, sizeof(str_ipaddr), count, &cur_idx);
			gateway = in_aton(str_ipaddr);

		} else if( strlen(opt_name) == 0 ){
			break;
		}
	}
	
	switch( command ){
		case CMD_VARP_ADD:	
			if( dst_ip && !is_zero_ether_addr(dst_mac) && !is_zero_ether_addr(next_mac) ){
				// varp entry insert
				virtlink_varp_add( dst_ip, next_mac, dst_mac );
			}
		break;
	
		case CMD_VARP_REMOVE:
			if( !is_zero_ether_addr(dst_mac) ){
				virtlink_varp_remove( dst_mac, NULL );
			}
		break;
	
		case CMD_VLINK_FLUSH:
			virtlink_varp_flush();
		break;
	
	
		case CMD_ADD:
			if(network) printk("<1>" "network true");
			if(strlen(if_name)) printk("<1>" "if_name true");
			if( network && strlen(if_name) ) {
				printk("<1>" "ipv4 update function called");	
				struct pfinder_entry_ipv4* rt_info;
				rt_info = pfinder_ipv4_entry_alloc(network, gateway, prefix, if_name);
				if(rt_info) {
					rt_info->gateway = rt_info->destination;
					pfinder_ipv4_update( SIOCADDRT, rt_info ); 
				}
			}
			
		break;
		
		case CMD_REMOVE:
		break;
	
		case CMD_FLUSH:
		break;
	}
	
	return count;
}

long pfinder_ipv4_sysfs_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct rtentry* rtentry = (struct rtentry*)arg;

		 __be32 network = ((struct sockaddr_in*)&rtentry->rt_dst)->sin_addr.s_addr;
		 __be32 mask = ((struct sockaddr_in*)&rtentry->rt_genmask)->sin_addr.s_addr;
	 	 u8 prefix = 32;
	
		 // prefix numbering
		while( ((cpu_to_be32(mask) >> (32-prefix)) & 1) == 0 && prefix > 0 ) prefix--;
	
	switch( cmd ){
		case SIOCADDRT:
		{
			 __be32 gateway = ((struct sockaddr_in*)&rtentry->rt_gateway)->sin_addr.s_addr;
		 	 char if_name[IFNAMSIZ];

			 u8 result = 0xFF;

			// device name copy
			result = copy_from_user( if_name, rtentry->rt_dev, IFNAMSIZ );
			
			if( result > 0 ){
				printk(KERN_CRIT NOTIFY_NAME "Routing device is not a valid one.");
				return 0;
			}

			printk(KERN_CRIT NOTIFY_NAME "network: [%08X], gateway: [%08X]/%d --> %s\n", network, gateway, prefix, if_name);
			rtentry = (struct rtentry*)pfinder_ipv4_entry_alloc( network, gateway, prefix, if_name );
			pfinder_ipv4_update( cmd, (void*)rtentry );
			break;
		}
		case SIOCDELRT:
		
			rtentry = (struct rtentry*)pfinder_ipv4_entry_alloc( network, 0, prefix, NULL );
			pfinder_ipv4_update( cmd, (void*)rtentry );
		break;
	}

	return 0;
}
