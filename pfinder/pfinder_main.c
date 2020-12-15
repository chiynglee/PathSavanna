#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/ktime.h>

#include "pfinder.h"
#include "pfinder_sysfs.h"
#include "pfinder_ipv4.h"

#define XBRT_PTYPE_HASH	4

//for schedule_timeout_interruptible of pfinder_clean_queue function
#include <linux/delay.h>
#include <linux/timer.h>

static const struct file_operations pfinder_fops = {
	.owner   = THIS_MODULE,
	.open    = pfinder_sysfs_open,
	.write   = pfinder_sysfs_write,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,

};

int pfinder_device_notifier( struct notifier_block *, unsigned long, void * );
void pfinder_device_unregister( struct net_device* , struct pfinder_dev_list*  );

struct list_head			ptype_handler;
struct pfinder_ptype_set		*ptype_hash[XBRT_PTYPE_HASH];

struct pfinder_queue*		fast_queue[PFINDER_MAX_QUEUES];
struct pfinder_queue		slowpath_queue;
struct proc_dir_entry		*pfinder_proc;
struct list_head			pfinder_devices;


static struct notifier_block device_nb = {
        .notifier_call = pfinder_device_notifier,
};

extern void pfinder_ipv4_register( void );
extern void* pfinder_direct_ipv4( void* data );
extern struct pfinder_entry_ipv4* pfinder_ipv4_entry_alloc( __be32 daddr, __be32 gateway, u8 prefix, const char* if_name );
extern u8 pfinder_ipv4_update( u16 cmd, struct pfinder_entry_ipv4* rt_info );


#ifdef GPU_USE
// cylee : add function declarations
void export_test(gntpg_list_t* target_gntpg, int req_dev_ifindex, int xmit_size); //export test : gy add
void pfinder_GPU_send_IP(gntpg_list_t* target_gntpg, void* data, u32 idx);
void pfinder_GPU_xmit(gntpg_list_t* target_gntpg, int req_dev_ifindex, int xmit_size);
#endif
	

#ifdef KU_OSL_VLINK
struct varp_table varp_tbl;

static __sum16 udp_checksum(struct encaphdr *encaph)
{
	struct udphdr *udph = &(encaph->udph);
	u16 ulen = ntohs(udph->len);
	__wsum pcsum;
	udph->check = 0;
	pcsum = csum_partial(udph, ulen,0);
	udph->check = csum_tcpudp_magic(encaph->iph.saddr, encaph->iph.daddr, ulen, encaph->iph.protocol, pcsum);
	return 0;
}


uint16_t ip_checksum(void* iph, size_t len)
{
	char* data = (char*)iph;
	uint32_t sum = 0xffff;
	size_t i = 0;
	for(; i+1 < len; i+=2) {

		uint16_t word;
		memcpy(&word, data+i, 2);
		sum += ntohs(word);
		if(sum > 0xffff) {
			sum-=0xffff;
		}
	}

	return htons(~sum); 
}

int ip_fragmentation(struct ixgbevf_tx_dma_buffer* tx_buffer, struct ixgbevf_tx_dma_buffer* tx_buffer2) {
	void* data = tx_buffer->data;
	struct ethhdr* eth = (struct ethhdr*)((u8*)data);
	struct iphdr* iph = (struct iphdr*)((u8*)data + ETH_HLEN);
	void* payload = ((u8*)data + ETH_HLEN + sizeof(struct iphdr));

	int header_len = ETH_HLEN + sizeof(struct iphdr);
	int packet_len = tx_buffer->len;
	int payload_len = packet_len - header_len;
	int fragment_size = packet_len - header_len - 1024;

	memcpy( tx_buffer2->data, tx_buffer->data, header_len );
	tx_buffer2->len = header_len;
	struct iphdr* buf2_iph = (struct iphdr*)(((u8*)tx_buffer2->data) + ETH_HLEN);

	memcpy( (tx_buffer2->data + header_len), (tx_buffer->data + packet_len - fragment_size), fragment_size );
	tx_buffer2->len += fragment_size;
	tx_buffer->len -= fragment_size;

	iph->tot_len = htons(ntohs(iph->tot_len) - fragment_size);
	buf2_iph->tot_len = htons(iph->ihl*4 + fragment_size);


	iph->frag_off = htons(0x2000);
	buf2_iph->frag_off = htons(1024 / 8);

	iph->check = 0;
	buf2_iph->check = 0;
	iph->check = ip_checksum((iph), iph->ihl*4);
	buf2_iph->check = ip_checksum((buf2_iph), iph->ihl*4);

	return 1;

}

int virtlink_encap_skb(struct sk_buff* skb, struct net_device* netdev) {

#ifdef PF_DEBUG
	printk("<1>" "encap skb func called \n");
#endif

	struct encaphdr *encaph;
	u8 pad_hlen = sizeof( struct encaphdr );
	bool is_broadcast = false;
		
	struct varp_entry* current_varp = NULL;
	struct pfinder_dev_list* pf_dev = NULL;

	// Addtion: if arp packet is skb->data[0-6] == 0xff ff ff ff ff ff
	// Therefore, we should skip searching them.
	if( unlikely(compare_ether_addr_64bits(skb->data, netdev->broadcast)) ){
		is_broadcast = true;
		current_varp = varp_tbl.head;
	} else {
		list_walk(current_varp, varp_tbl.head){
			if( compare_ether_addr_64bits(skb->data, current_varp->dst_ethaddr) )
				break;
		}
	}
	
	// find out original xmit function descriptor
	list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
		if( pf_dev->netdev == netdev ) break;
	}
		
	// there was completely matched on vARP entry, then capsule with it!
	if( likely(current_varp != NULL) ){
		// make new space for header padding
		if( unlikely(pskb_expand_head(skb, SKB_DATA_ALIGN(pad_hlen), 0, GFP_ATOMIC)) ) goto out;

		// make headroom to data space
		skb_push(skb, pad_hlen);
		// move layer header pointer to our encapsulated header
		skb_reset_mac_header(skb);
		skb_set_network_header(skb, sizeof(struct ethhdr));
		skb_set_transport_header(skb, sizeof(struct ethhdr) + sizeof(struct iphdr));
		
		// make clean encapulate headers (authoritative mac - ip - udp)
		memset( skb->data, 0, pad_hlen );
		// pointing encap header
		encaph = (struct encaphdr*) skb->data;
	
		// authoritative mac header settings
		memcpy(encaph->etherh.h_dest, current_varp->nexthop_ethaddr, ETH_ALEN);
		memcpy(encaph->etherh.h_source, netdev->dev_addr, ETH_ALEN);
		encaph->etherh.h_proto = cpu_to_be16(ETH_P_IP);
	
		// for checksum offloading
		//skb->ip_summed = CHECKSUM_PARTIAL;/*CHECKSUM_NONE*/; // checksum offload flag set
		skb->ip_summed = CHECKSUM_NONE;
		// protocol setting to IP
		skb->protocol = cpu_to_be16(ETH_P_IP);
			
		// authoritative ip header settings
		encaph->iph.ihl = 0x05;
		encaph->iph.version = IPVERSION;
		encaph->iph.ttl = IPDEFTTL;
		encaph->iph.tot_len = cpu_to_be16(skb->tail - skb->network_header);
		encaph->iph.protocol = IPPROTO_UDP;
		encaph->iph.daddr = current_varp->represent_dst_ipv4_addr;
		encaph->iph.saddr = pf_dev->represent_ipv4_addr;
		/*
		encaph->iph.tos = 0;
		encaph->iph.id = 0;
		encaph->iph.frag_off = 0;
		encaph->iph.check = 0; // checksum will be offloaded in function ixgbevf_tx_queue.
		*/
		encaph->iph.check = ip_checksum(&(encaph->iph), encaph->iph.ihl*4);	
		// udp header settings
		encaph->udph.dest = htons(ENCAP_BASE_UDP_PORT + pf_dev->vnet_id);
		encaph->udph.len = cpu_to_be16(skb->tail - skb->transport_header);
		//shlee add
		encaph->udph.source = htons(ENCAP_BASE_UDP_PORT);
		encaph->udph.check = 0;
		//udp_checksum(encaph);
	
	//	encaph->udph.check = (__sum16) __skb_checksum_complete_head(skb,(__u16) encaph->udph.len);
	//	encaph->udph.check = (int) csum_partial(skb_transport_header(skb), sizeof(struct udphdr), 0);
		/*
		encaph->udph.source = htons(ENCAP_BASE_UDP_PORT);
		encaph->udph.check = 0;
		*/
	} else {
		// Actually we should not send vARP failed packet
		// However, for test we do not drop them.
		goto out;
	}
	
	// xmit to original function of device
	if( likely(pf_dev != NULL) ){

		// broadcast processing : skb cloning
		if( is_broadcast ){

#ifdef PF_DEBUG
			printk("<1>" "encap skb broadcasting?\n");
#endif

			int retval = NETDEV_TX_OK;
			// traverse whole varp entries
			list_walk(current_varp, varp_tbl.head){
				struct sk_buff *cskb = skb_clone(skb, GFP_ATOMIC);
				if( cskb ){
					encaph = (struct encaphdr*) cskb->data;
					memcpy(encaph->etherh.h_dest, current_varp->nexthop_ethaddr, ETH_ALEN);
					encaph->iph.daddr = current_varp->represent_dst_ipv4_addr;
					retval |= pf_dev->prev_netdev_ops->ndo_start_xmit(cskb, netdev);
				}
			}
	
			// free original skb
			dev_kfree_skb_any(skb);
			return retval;
				
		} else {
			goto xmit;
		}
	}
	else goto out;
	
	
	xmit:
		return pf_dev->prev_netdev_ops->ndo_start_xmit(skb, netdev);
		
	out:
		printk(KERN_INFO "virtlink: encapsulated xmit failed.\n");
		skb->dev->_tx->tx_dropped++;
		dev_kfree_skb_any(skb);
		return NETDEV_TX_OK;

}


int virtlink_encap_dma_buffer(struct ixgbevf_tx_dma_buffer* tx_buffer, struct net_device* netdev)
{
#ifdef PF_DEBUG
	printk(KERN_INFO "encap dma buffer func called\n");
#endif

	void* data = tx_buffer->data;
	struct ethhdr* eth = (struct ethhdr*)((u8*)data);
	struct iphdr* iph = (struct iphdr*)((u8*)data + ETH_HLEN);
	struct udphdr* udph = (struct udphdr*)((u8*)data + ETH_HLEN + sizeof(struct iphdr));
	void* payload = ((u8*)data +  + ETH_HLEN + sizeof(struct iphdr) + sizeof(struct udphdr));

	int header_len = payload - data;
	int packet_len = tx_buffer->len;
	int payload_len = packet_len - header_len;

#ifdef PF_DEBUG
//        printk("<1>" "<eth heade>\n");
//        printk("<1>" "destination : %02X:%02x:%02x:%02x:%02x:%02x\n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
//        printk("<1>" "source : %02x:%02x:%02x:%02x:%02x:%02x\n", eth->h_source[0], eth->h_source[1], eth->h_source[2], eth->h_source[3], eth->h_source[4], eth->h_source[5]);

//        printk("<1>" "ip length %d\n",ntohs(iph->tot_len));
//        printk("<1>" "<ip header>\n");
//        printk("<1>" "soure ip addr : %d.%d.%d.%d\n", (iph->saddr & 0xFF), (iph->saddr & 0xFF00) >> 8, (iph->saddr & 0xFF0000) >> 16, (iph->saddr & 0xFF000000)>> 24);
//        printk("<1>" "dest ip addr : %d.%d.%d.%d\n", (iph->daddr & 0xFF), (iph->daddr & 0xFF00) >> 8, (iph->daddr & 0xFF0000) >> 16, (iph->daddr & 0xFF000000) >> 24);
#endif

	struct encaphdr* encaph;
	u8 pad_hlen = sizeof(struct encaphdr);
	struct pfinder_dev_list* pf_dev = NULL;

	//port????
	
	struct varp_entry* current_varp = NULL;

	list_walk(current_varp, varp_tbl.head){
#ifdef PF_DEBUG
//        printk("<1>" "routing : %02X:%02x:%02x:%02x:%02x:%02x\n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
//        printk("<1>" "varp : %02x:%02x:%02x:%02x:%02x:%02x\n", current_varp->dst_ethaddr[0], current_varp->dst_ethaddr[1], current_varp->dst_ethaddr[2], current_varp->dst_ethaddr[3], current_varp->dst_ethaddr[4], current_varp->dst_ethaddr[5]);
#endif
		if( !compare_ether_addr(eth->h_dest, current_varp->dst_ethaddr) ) {
			break;
		}
	}

	list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
		if( pf_dev->netdev == netdev ) break;
	}


	// there was completely matched on vARP entry, then capsule with it!
	if( likely(current_varp != NULL) ){
		//packet size ???????????
		tx_buffer->len += sizeof(struct encaphdr);
		memmove(data + sizeof(struct encaphdr), data, packet_len);
		memset(data, 0, sizeof(struct encaphdr));

		encaph = (struct encaphdr*)kmalloc(sizeof(struct encaphdr), GFP_KERNEL);
		if(!encaph) goto out;
			
		memcpy( encaph->etherh.h_dest, current_varp->nexthop_ethaddr, ETH_ALEN );
		memcpy( encaph->etherh.h_source, netdev->dev_addr, ETH_ALEN );
		encaph->etherh.h_proto = htons(ETH_P_IP);
              // for checksum offloading
             //  skb->ip_summed = CHECKSUM_PARTIAL;//CHECKSUM_NONE/; // checksum offload flag set
		// protocol setting to IP
		
		// authoritative ip header settings
		encaph->iph.ihl = 0x05;
		encaph->iph.version = IPVERSION;
		encaph->iph.tos = 0;
//		encaph->iph.ttl = IPDEFTTL;
		encaph->iph.tot_len = htons(tx_buffer->len - sizeof(struct ethhdr));
		encaph->iph.id = 0;
		encaph->iph.frag_off = 0;
		encaph->iph.ttl = 32;
		encaph->iph.protocol = IPPROTO_UDP;
		encaph->iph.daddr = current_varp->represent_dst_ipv4_addr;
		encaph->iph.saddr = pf_dev->represent_ipv4_addr;
//              encaph->iph.check = htons(0x80a6);
		encaph->iph.check = 0;
		 // udp header settings
		encaph->udph.dest = htons(ENCAP_BASE_UDP_PORT + pf_dev->vnet_id);
		encaph->udph.len = htons(tx_buffer->len - sizeof(struct ethhdr) - sizeof(struct iphdr));
		encaph->udph.source = htons(ENCAP_BASE_UDP_PORT);
		encaph->udph.check = 0;
//		udp_checksum(encaph);

		encaph->iph.check = ip_checksum(&(encaph->iph), encaph->iph.ihl*4);

		memcpy( data, encaph, sizeof(struct encaphdr) );

		
		kfree(encaph);
        } else {

#ifdef PF_DEBUG
		printk("<1>" "there isn't matched varp entry");
#endif

              goto out;
        }
//spin_unlock_irqrestore(&testlock, testflags);	
	return 0;	
out:
//spin_unlock_irqrestore(&testlock, testflags);

#ifdef PF_DEBUG
	printk("<1>" "virtlink: encapsulated xmit failed.\n");
#endif

	return 0;
}



struct sk_buff* virtlink_decap_skb(struct sk_buff *skb) {


	printk("<1>" "decap skb function called\n");


	struct encaphdr *encaph;
	u16 pkt_vnet_id;
	struct pfinder_dev_list* pf_dev = NULL;

	if( (skb = skb_share_check(skb, GFP_ATOMIC)) == NULL )	goto drop;
	if(!pskb_may_pull(skb, sizeof(struct encaphdr) - ETH_HLEN))	goto drop;

	// find out original device descriptor
	list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
		if( pf_dev->netdev == skb->dev ) break;
	}
	
	if( pf_dev == NULL ) goto drop;
	
	encaph = (struct encaphdr*) (skb->data - ETH_HLEN);
	pkt_vnet_id = cpu_to_be16(encaph->udph.dest) - ENCAP_BASE_UDP_PORT;

	// Decapsulation Step
	if ( encaph->iph.protocol == IPPROTO_UDP && pkt_vnet_id == pf_dev->vnet_id ){
		skb_pull(skb, sizeof(struct encaphdr) - ETH_HLEN);
		if(!pskb_may_pull(skb, ETH_HLEN))	goto drop;
		skb_reset_mac_header(skb);
		skb->pkt_type = PACKET_HOST;
		skb->protocol = eth_type_trans(skb, skb->dev); // eth_type_trans pull payload which of ethernet
		skb_reset_network_header(skb);		     // therefore, current head is start of network header

		// if it was not other host packet, we process it. --> else drop them.
		if( skb->pkt_type != PACKET_OTHERHOST )	goto out;
	}

drop:

#ifdef PF_DEBUG
	printk("pkt drp.\n");
#endif

	atomic_long_inc(&skb->dev->rx_dropped);
	kfree_skb(skb);
	skb = NULL;
	
out:
	return skb;
}

int virtlink_decap_dma_buffer(struct ixgbevf_tx_dma_buffer* dma_buffer, struct net_device* netdev) {

#ifdef PF_DEBUG
	printk("<1>" "decap dma buffer func called\n");
#endif

	void* data = dma_buffer->data;
	struct ethhdr* eth = (struct ethhdr*)data;
	struct iphdr* iph = (struct iphdr*)((u8*)data + ETH_HLEN);
	struct udphdr* udph = (struct udphdr*)((u8*)data + ETH_HLEN + sizeof(struct iphdr));
	void* payload = ((u8*)data +  + ETH_HLEN + sizeof(struct iphdr) + sizeof(struct udphdr));

        int header_len = payload - data;
        int packet_len = dma_buffer->len;
        int payload_len = packet_len - header_len;
	
#ifdef PF_DEBUG
//	printk("<1>" "decap\n dst : %02X:%02x:%02x:%02x:%02x:%02x\n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
//	printk("<1>" "src : %02x:%02x:%02x:%02x:%02x:%02x\n", eth->h_source[0], eth->h_source[1], eth->h_source[2], eth->h_source[3], eth->h_source[4], eth->h_source[5]);

//	printk("<1>" "ip length %d\n",ntohs(iph->tot_len));
//	printk("<1>" "src ip addr : %d.%d.%d.%d\n", (iph->saddr & 0xFF), (iph->saddr & 0xFF00) >> 8, (iph->saddr & 0xFF0000) >> 16, (iph->saddr & 0xFF000000)>> 24);
//	printk("<1>" "dst ip addr : %d.%d.%d.%d\n", (iph->daddr & 0xFF), (iph->daddr & 0xFF00) >> 8, (iph->daddr & 0xFF0000) >> 16, (iph->daddr & 0xFF000000) >> 24);
#endif

	struct encaphdr *encaph;
	u16 pkt_vnet_id;
	struct pfinder_dev_list* pf_dev = NULL;

	// find out original device descriptor
	list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
		if( pf_dev->netdev == netdev ) break;
	}	
	if( pf_dev == NULL )  goto drop;

	encaph = (struct encaphdr*) (dma_buffer->data);
	pkt_vnet_id = cpu_to_be16(encaph->udph.dest) - ENCAP_BASE_UDP_PORT;

	// Decapsulation Step
	if ( encaph->iph.protocol == IPPROTO_UDP && pkt_vnet_id == pf_dev->vnet_id ){
		dma_buffer->len -= sizeof(struct encaphdr);
		memmove(dma_buffer->data, dma_buffer->data + sizeof(struct encaphdr), dma_buffer->len); 
		goto out;
	}
drop:

#ifdef PF_DEBUG
	printk("pkt drp.\n");
#endif

	return -1;
out:
	return 0;

}


int virtlink_varp_add(__be32 dst_ipaddr, u8* nexthop_ethaddr, u8* dst_ethaddr)
{
//	printk("<1>" "varp add func called");

	struct varp_entry* new_entry = NULL;
	int node = cpu_to_node(smp_processor_id());
	
	// find if it is already added or not
	list_walk(new_entry, varp_tbl.head){
		if( new_entry == NULL ) break;
		if( !compare_ether_addr_64bits(dst_ethaddr, new_entry->dst_ethaddr)  ) break;
	}

	if(new_entry == NULL){	
		new_entry = kzalloc_node(sizeof(struct varp_entry), GFP_KERNEL, node);
		if( !new_entry)
		{
			printk(KERN_INFO "virtlink: Memory allocation failed.\n");
			goto out;
		}
		else
		{
			new_entry->prev = NULL;
			new_entry->next = NULL;
			if(varp_tbl.head == NULL){
				varp_tbl.head = new_entry;
				varp_tbl.tail = new_entry;
			} else {
				new_entry->prev = varp_tbl.tail;
				varp_tbl.tail->next = new_entry;
				varp_tbl.tail = new_entry;
			}
			
			varp_tbl.tbl_size++;
		}
	}
		
	new_entry->represent_dst_ipv4_addr = dst_ipaddr;
	memcpy(new_entry->nexthop_ethaddr, nexthop_ethaddr, ETH_ALEN);
	memcpy(new_entry->dst_ethaddr, dst_ethaddr, ETH_ALEN);

//	printk("<1>" "ip addr : %d", dst_ipaddr);
//	printk("<1>" "encap mac : %02X:%02x:%02x:%02x:%02x:%02x\n", nexthop_ethaddr[0], nexthop_ethaddr[1], nexthop_ethaddr[2], nexthop_ethaddr[3], nexthop_ethaddr[4], nexthop_ethaddr[5]);
//	printk("<1>" "dst mac : %02x:%02x:%02x:%02x:%02x:%02x\n", dst_ethaddr[0], dst_ethaddr[1], dst_ethaddr[2], dst_ethaddr[3], dst_ethaddr[4], dst_ethaddr[5]);


	printk(KERN_INFO "virtlink: vARP table updated.\n");
	
out:	
	return 0;
}

int virtlink_varp_remove(u8* dst_ethaddr, struct varp_entry* entry)
{
	if( entry == NULL && dst_ethaddr ){
		// find if it is in table or not
		list_walk(entry, varp_tbl.head){
			if( entry == NULL ) break;
			if( !compare_ether_addr_64bits(dst_ethaddr, entry->dst_ethaddr)  ) break;
		}
	}

	if(entry != NULL){
		// For maintaining linked list
		if( entry == varp_tbl.head ) { 
				varp_tbl.head = entry->next; 
				if( entry->next ) entry->next->prev = NULL;
			}
		else {
			entry->prev->next = entry->next;
		}

		if( entry == varp_tbl.tail ){
			varp_tbl.tail = entry->prev;
			if( entry->prev ) entry->prev->next = NULL;
		}
		else {
			entry->next->prev = entry->prev;
		}
		
		kfree(entry);
	}
	return 0;
}

int virtlink_varp_flush(void)
{
	struct varp_entry *cur_entry;	
	list_walk(cur_entry, varp_tbl.head){
		virtlink_varp_remove(NULL, cur_entry);
	}
	printk(KERN_INFO "virtlink: varp table flushed.\n");
	return 0;
}

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

#endif


int pfinder_ptype_register( struct pfinder_ptype_set* ptype_set )
{
	list_add_tail( &ptype_set->node, &ptype_handler );
	return 1;
}

int pfinder_ptype_unregister( struct pfinder_ptype_set* ptype_set )
{
	if( !list_empty(&ptype_set->node) )	list_del_init( &ptype_set->node );
	return 1;
}

int pfinder_ptype_flush( void )
{
	struct pfinder_ptype_set* ptype_set = NULL;
	
	while( !list_empty( &ptype_handler ) ){
		ptype_set = list_first_entry(&ptype_handler, struct pfinder_ptype_set, node);
		if( ptype_set->flush ) ptype_set->flush( ptype_set );
		pfinder_ptype_unregister( ptype_set );
	}
	
	return list_empty( &ptype_handler );
}

struct sk_buff* pfinder_skb_handler( struct sk_buff* skb )
{
	__be16 type = skb->protocol;
	struct pfinder_ptype_set* ptype_set = NULL;

	// If they are not incoming packet, just drop.
	if( unlikely(skb->pkt_type == PACKET_OTHERHOST) ) goto drop;

	// Hash Detection
 	if( likely(ptype_set == ptype_hash[ type%XBRT_PTYPE_HASH ])){
		if( likely(ptype_set->ptype == type) )
			goto out;
	}

	// Find if it's packet type is in our ptype sets
	list_for_each_entry(ptype_set, &ptype_handler, node){
		if( ptype_set->ptype == type ){
			ptype_hash[ type%XBRT_PTYPE_HASH ] = ptype_set;
			goto out;
		}
	}
	
	// if there is no packet type handler, let it do kernel.
	goto out;

drop:
	kfree_skb(skb);
	skb = NULL;
	
out:
	if( likely(ptype_set) )
		skb = ptype_set->handler( skb, ptype_set );
	
	return skb;
}

int pfinder_skb_direct(struct sk_buff* skb)
{
	if( !skb ) return 0;
	
	skb_reset_network_header(skb);
	//skb = pfinder_skb_handler(skb);
	if( unlikely(skb) ) netif_receive_skb(skb);
	return 0;
}

/*
struct net_device* pfinder_simple_forward(void* data, struct net_device* target_dev)
{
//	u8 dst_mac2[ETH_ALEN] = { 0x90, 0xE2, 0xBA, 0x0F, 0x9B, 0x30 };
//	memcpy( data, dst_mac2, ETH_ALEN );
//	memcpy( data+ETH_ALEN, target_dev->dev_addr, ETH_ALEN );
	target_dev = pfinder_direct_ipv4( data );
//	printk(KERN_INFO NOTIFY_NAME "target_dev : %s", target_dev->name);
	return target_dev;
}
*/

struct pfinder_entry_ipv4* pfinder_simple_forward(void* data)
{
	return pfinder_direct_ipv4( data );
}

#ifdef GPU_USE
void* pfinder_GPU_clean_queue(struct pfinder_queue* queue)
{
	struct ixgbevf_dma_buffer* dma_buffer;
	struct pfinder_entry_ipv4* rt_info;
	
	double elaps;
	u64 pps;
	gntpg_list_t* target_gntpg = NULL; //gy add

	
	//gy add
	if( target_gntpg == NULL){
		target_gntpg = savanna_getEnabledSM();
	}

	if( queue->dev->flag & PFINDER_DEV_ADV_BUFFER ){
		
		u32 count = 0;
		u32 bbuf_use = queue->bbuf_clean_use;

		struct ixgbevf_tx_dma_buffer** xmit_buffer = queue->batch_buffer[bbuf_use];
		
		while( !signal_pending(current) && !kthread_should_stop() ){
			struct ixgbevf_tx_dma_buffer* tx_buffer;
			struct net_device* target_dev;
			int target_index = 0;	
			
			// if there is some queues for processing in this thread,
			while( likely(queue->head != queue->tail) ){

				u32 find_limit = PFINDER_TX_LEN >> 4;

				// DMA buffer retrieve
				prefetch( queue->rx_buffers[ queue->head ].dma_buffer->data );
				dma_buffer = queue->rx_buffers[ queue->head ].dma_buffer;
			
				// Fetch next queue head
				queue->head = (queue->head + 1) % PFINDER_QUEUE_LEN;
				if( likely(queue->head != queue->tail) )
					prefetch( queue->rx_buffers[ queue->head ].dma_buffer->data );

				// Tx buffer retrieve
				tx_buffer = xmit_buffer[count];

				// Fetch next tx buffer
				memcpy(tx_buffer->data, dma_buffer->data, dma_buffer->len);
				tx_buffer->len = dma_buffer->len;
				dma_buffer->len = 0;
				if(queue->dev->vnet_id != 0) {
					if(virtlink_decap_dma_buffer(tx_buffer, queue->dev->netdev) == -1) { continue; }
				}

void* dataxx = tx_buffer->data;
struct ethhdr* ethxx = (struct ethhdr*) dataxx;
				//gy add
				pfinder_GPU_send_IP(target_gntpg, tx_buffer->data, count++);
#if GY_DEBUG
				printk(KERN_INFO "count : %d\n",count);
#endif
				if(count%PFINDER_CLEAN_BATCH_SIZE == 0){
#if GY_DEBUG
					printk(KERN_INFO "send request\n");
					printk(KERN_DEBUG "request options, 0 : %d, 1 : %d, 2 : %d\n", target_gntpg->listId, queue->dev->netdev->ifindex,count);
#endif
#if GY_TIME_ESTIMATE
					rs_rettime = ktime_get();
					rs_delta = ktime_sub(rs_rettime, rs_calltime);
					duration = (unsigned long long) ktime_to_ns(rs_delta) >> 10;
					printk(KERN_DEBUG "receive to send : %lld usecs\n", duration);
					sr_calltime = ktime_get();
#endif
					getnstimeofday(&(target_gntpg->gpuLaunch[0]));
					queue->bbuf_xmit_use = bbuf_use;
					send_request(REQ_GPULAUNCH, target_gntpg->listId,queue->dev->netdev->ifindex,count);

					sync_flag = 0;
					while(1){
						if(sync_flag == 1)	break;
						set_current_state(TASK_INTERRUPTIBLE);
			            msleep_interruptible(1);
                     	set_current_state(TASK_RUNNING);
#if GY_DEBUG
						printk(KERN_INFO "wait GPU complete\n");
#endif
					}

#if GY_TIME_ESTIMATE
					pf_sr_rettime = ktime_get();
					pf_sr_delta = ktime_sub(pf_sr_rettime, sr_calltime);
					pf_duration = (unsigned long long) ktime_to_ns(pf_sr_delta) >> 10;
					printk(KERN_DEBUG "send to pf recieve :	%lld usecs\n", pf_duration);
#endif
					target_gntpg = savanna_getEnabledSM();
					bbuf_use = (bbuf_use + 1) % PFINDER_CLEAN_BATCH_MAX;
					queue->bbuf_clean_use = bbuf_use;
					count = 0;
					xmit_buffer = queue->batch_buffer[bbuf_use];
				}	
			}
/*	
			if(count){
#if GY_DEBUG
//				printk(KERN_INFO "count %d\n",count);
//				printk(KERN_INFO "%d\n",target_gntpg);
//				print_data(target_gntpg, target_gntpg->listId );
				printk(KERN_INFO "send request\n");
				printk(KERN_DEBUG "request options, 0 : %d, 1 : %d, 2 : %d\n", target_gntpg->listId, queue->dev->netdev->ifindex,count);
#endif
#if GY_TIME_ESTIMATE
				rs_rettime = ktime_get();
				rs_delta = ktime_sub(rs_rettime, rs_calltime);
				duration = (unsigned long long) ktime_to_ns(rs_delta) >> 10;
				printk(KERN_DEBUG "receive to send : %lld usecs\n", duration);
				sr_calltime = ktime_get();
#endif
				getnstimeofday(&(target_gntpg->gpuLaunch[0]));
				queue->bbuf_xmit_use = bbuf_use;
				send_request(REQ_GPULAUNCH, target_gntpg->listId,queue->dev->netdev->ifindex,count);
				sync_flag = 0;
//				sync_flag++;
				while(1){
					if(sync_flag == 1)	break;
//					if(sync_flag <= NUMBER_SM) break;
					set_current_state(TASK_INTERRUPTIBLE);
		                      	msleep_interruptible(1);
              				set_current_state(TASK_RUNNING);
//					schedule_timeout_interruptible(0);//sleep function
//					ndelay(1000);
#if GY_DEBUG
					printk(KERN_INFO "wait GPU complete\n");
#endif
				}
				target_gntpg = savanna_getEnabledSM();
				bbuf_use = (bbuf_use + 1) % PFINDER_CLEAN_BATCH_MAX;
				queue->bbuf_clean_use = bbuf_use;
				count = 0;
				xmit_buffer = queue->batch_buffer[bbuf_use];
			}
*/				
			set_current_state(TASK_INTERRUPTIBLE);
			msleep_interruptible(1);
			set_current_state(TASK_RUNNING);
		}
	}
	else {
		while( !signal_pending(current) && !kthread_should_stop() ){
			// if there is some queues for processing in this thread,
			struct net_device* target_dev = NULL;
			while( likely(queue->head != queue->tail) ){			
				struct sk_buff* skb = queue->rx_buffers[ queue->head ].skb;
				int ret = 0;
				
				// Fetch next queue head
				queue->head = (queue->head + 1) % PFINDER_QUEUE_LEN;
				
//				target_dev = pfinder_simple_forward(skb->data, default_dev);
//				skb_set_dev(skb, target_dev);
				skb_push(skb, ETH_HLEN);
				skb->queue_mapping = 0;

				ret = target_dev->netdev_ops->ndo_start_xmit(skb, target_dev);
				if( unlikely(ret != NETDEV_TX_OK) ) dev_kfree_skb_any( skb );
			}
			set_current_state(TASK_INTERRUPTIBLE);
			msleep_interruptible(1);
			set_current_state(TASK_RUNNING);
		}	
	}
	
	elaps = ktime_to_ns(ktime_sub( queue->last_enqueue, queue->start_enqueue)) / 1000 / 1000 / 1000;
	pps = queue->total_packets / elaps;
	printk(KERN_INFO NOTIFY_NAME "%s enqueued packets in %llu pps. (Drops: %llu)", queue->dev->netdev->name, pps, queue->drops);

	return NULL;
}
#endif

#ifndef GPU_USE
void* pfinder_clean_queue(struct pfinder_queue* queue)
{
	struct ixgbevf_dma_buffer* dma_buffer;
	struct pfinder_entry_ipv4* rt_info;
	
	double elaps;
	u64 pps;

	if( queue->dev->flag & PFINDER_DEV_ADV_BUFFER ){
		int buffer_index = 0;
		u32 count[PFINDER_MAX_INTERFACES];
//		struct ixgbevf_tx_dma_buffer** xmit_buffer = kzalloc(sizeof(void*) * PFINDER_TX_LEN, GFP_ATOMIC);
		struct ixgbevf_tx_dma_buffer*** xmit_buffer = kzalloc(sizeof(struct ixgbevf_tx_dma_buffer**) * PFINDER_MAX_INTERFACES, GFP_ATOMIC);
		for( ; buffer_index < PFINDER_MAX_INTERFACES; buffer_index++) {
			xmit_buffer[buffer_index] = kzalloc(sizeof(void*) * PFINDER_TX_LEN, GFP_ATOMIC);
			count[buffer_index] = 0;
		}
//		u32 tx_buffer_taken = 0;






		while( !signal_pending(current) && !kthread_should_stop() ){
			struct ixgbevf_tx_dma_buffer* tx_buffer;
			struct net_device* target_dev;
			int target_index = 0;	
			
			// if there is some queues for processing in this thread,
			while( likely(queue->head != queue->tail) ){

				u32 find_limit = PFINDER_TX_LEN >> 4;

				// DMA buffer retrieve
				prefetch( queue->rx_buffers[ queue->head ].dma_buffer->data );
				dma_buffer = queue->rx_buffers[ queue->head ].dma_buffer;
			
				// Fetch next queue head
				queue->head = (queue->head + 1) % PFINDER_QUEUE_LEN;
				if( likely(queue->head != queue->tail) )
					prefetch( queue->rx_buffers[ queue->head ].dma_buffer->data );

				// Tx buffer retrieve
				tx_buffer = queue->tx_buffers[ queue->tx_head ];
//printk("PF Error %d %d\n", tx_buffer, queue->tx_head);
				/*while( unlikely(tx_buffer->status != NULL && ((*tx_buffer->status) & cpu_to_le32(0x01)) == 0 ) ){
					if( unlikely(find_limit-- == 0) ) {
						dma_buffer->len = 0;
						queue->drops++;
						break;
					}
					queue->tx_head = (queue->tx_head + 1) % PFINDER_TX_LEN;
					tx_buffer = queue->tx_buffers[ queue->tx_head ];
				}*/
//printk("PF Error %d\n", queue->tx_head);
				//if( unlikely(dma_buffer->len) == 0 ) continue;

				//tx_buffer->status = NULL;
//				tx_buffer->status = &tx_buffer_taken;

				// Fetch next tx buffer
				queue->tx_head = (queue->tx_head + 1) % PFINDER_TX_LEN;	
				memcpy(tx_buffer->data, dma_buffer->data, dma_buffer->len);
				tx_buffer->len = dma_buffer->len;
				dma_buffer->len = 0;
				if(queue->dev->vnet_id != 0) {
					if(virtlink_decap_dma_buffer(tx_buffer, queue->dev->netdev) == -1) {
continue; }
				}
	
//				target_dev = pfinder_simple_forward(tx_buffer->data, default_dev);
				rt_info = pfinder_simple_forward(tx_buffer->data);
				if(rt_info == NULL) {

#ifdef PF_DEBUG
					printk("<1>" "routeinfo == null\n");
#endif

					//broadcasting - To be implemented...
					//continue;
				} else {

					target_dev = rt_info->netdev;
					target_index = rt_info->netdev_index;

#ifdef PF_DEBUG
					printk("<1>" "dest : %d, netdev : %s\n", rt_info->destination, rt_info->netdev->name);
					printk("<1>" "target index : %d", target_index);
#endif
				}

				if (rt_info == NULL || rt_info->is_local) {

					//printk("<1>" "pf slowpath\n");
					queue->dev->slowpath_rx(queue->dev->netdev, tx_buffer->data, tx_buffer->len);
					continue;

				} else {

					//printk("<1>" "fast path\n");
					struct pfinder_dev_list* pf_dev = NULL;

					if( !list_empty( &pfinder_devices)) {
						list_for_each_entry(pf_dev, &pfinder_devices, dev_list) {
							if(pf_dev->netdev == target_dev) break;
						}

						if(pf_dev->netdev == target_dev) {
							if(pf_dev->vnet_id != 0) {
								if(tx_buffer->len >= 1400) {
									struct ixgbevf_tx_dma_buffer* tx_buffer2 = queue->tx_buffers[ queue->tx_head ];
									queue->tx_head = (queue->tx_head + 1)% PFINDER_TX_LEN;
									if(unlikely(!ip_fragmentation(tx_buffer, tx_buffer2))) {
										continue;
									} else {
										virtlink_encap_dma_buffer(tx_buffer2, target_dev);
										xmit_buffer[target_index][count[target_index]++] = tx_buffer2;
									}
								}
								
								virtlink_encap_dma_buffer(tx_buffer, target_dev);
							} else {
                                   			//no encap
                                    			//printk("<1>" "no encap\n");
							}
						}
					}
				}


//				xmit_buffer[count++] = tx_buffer;
				xmit_buffer[target_index][count[target_index]++] = tx_buffer;

void* dataxx = tx_buffer->data;
struct ethhdr* ethxx = (struct ethhdr*) dataxx;
//printk("<1>" "dest : %02X:%02x:%02x:%02x:%02x:%02x\n", ethxx->h_dest[0], ethxx->h_dest[1], ethxx->h_dest[2], ethxx->h_dest[3], ethxx->h_dest[4], ethxx->h_dest[5]);

			
//cj:batch size change
				if(count[target_index] % 102400 == 0 ){
					//printk(KERN_INFO NOTIFY_NAME "xmit target_dev : %s , count : %d", target_dev->name, count);
					queue->dev->xmit(target_dev, xmit_buffer[target_index], count[target_index]);
					count[target_index] = 0;
				}
			}
			
			if(count[target_index]){
				//printk(KERN_INFO NOTIFY_NAME "check2 xmit target_dev : %s , count : %d", target_dev->name, count);
				queue->dev->xmit(target_dev, xmit_buffer[target_index], count[target_index]);
				count[target_index] = 0;
			}
			
			set_current_state(TASK_INTERRUPTIBLE);
			msleep_interruptible(1);
			set_current_state(TASK_RUNNING);
		}
		
		for(buffer_index = 0; buffer_index < PFINDER_MAX_INTERFACES; buffer_index++) {
			kfree(xmit_buffer[buffer_index]);
		}
		kfree(xmit_buffer);
	}

	else {
		while( !signal_pending(current) && !kthread_should_stop() ){
			// if there is some queues for processing in this thread,
			struct net_device* target_dev;
			while( likely(queue->head != queue->tail) ){			
				struct sk_buff* skb = queue->rx_buffers[ queue->head ].skb;
				int ret = 0;
				
				// Fetch next queue head
				queue->head = (queue->head + 1) % PFINDER_QUEUE_LEN;
				
//				target_dev = pfinder_simple_forward(skb->data, default_dev);
				skb_set_dev(skb, target_dev);
				skb_push(skb, ETH_HLEN);
				skb->queue_mapping = 0;

				ret = target_dev->netdev_ops->ndo_start_xmit(skb, target_dev);
				if( unlikely(ret != NETDEV_TX_OK) ) dev_kfree_skb_any( skb );
			}
			set_current_state(TASK_INTERRUPTIBLE);
			msleep_interruptible(1);
			set_current_state(TASK_RUNNING);
		}	
	}
	
	elaps = ktime_to_ns(ktime_sub( queue->last_enqueue, queue->start_enqueue)) / 1000 / 1000 / 1000;
	pps = queue->total_packets / elaps;
	printk(KERN_INFO NOTIFY_NAME "%s enqueued packets in %llu pps. (Drops: %llu)", queue->dev->netdev->name, pps, queue->drops);

	return NULL;
}
#endif


u16 pfinder_dma_enqueue( u16 queue_idx, struct ixgbevf_dma_buffer* buffer_info )
{
	struct pfinder_queue* queue = fast_queue[ queue_idx ];
	u32 tail;

	// Queue check
	if( unlikely(!queue) ){
		buffer_info->len = 0;
		return 0;
	}

	tail = (queue->tail + 1) % PFINDER_QUEUE_LEN;
	
	// Buffer full check
	if( unlikely(queue->head == tail) ){
		buffer_info->len = 0;
		queue->drops++;
		return 0;
	}

	// Perform enqueue
	queue->rx_buffers[ queue->tail ].dma_buffer = buffer_info;

	// Wake up clean process
	if( tail % 16 == 0 )
		wake_up_process(queue->dev->clean_task);

	// Move to next buffer
	queue->tail = tail;

	// PPS Check Function
	if( ktime_to_ns(queue->start_enqueue) == 0 ) queue->start_enqueue = ktime_get();
	queue->last_enqueue = ktime_get();
	queue->total_packets++;

	return 1;
}

struct sk_buff* pfinder_enqueue(struct sk_buff* skb)
{
	struct pfinder_queue* queue = fast_queue[ skb->dev->ifindex ];
	u32 tail;
	
	// Queue check
	if( unlikely(!queue) ){
		return skb;
	}

	tail = (queue->tail + 1) % PFINDER_QUEUE_LEN;

	// Buffer full check
	if( unlikely(queue->head == tail) ){
		dev_kfree_skb_any( skb );
		queue->drops++;
		return NULL;
	}

	// Perform enqueue
	queue->rx_buffers[ queue->tail ].skb = skb;

	// Wake up clean process
	if( unlikely(tail % CLEANTASK_CALL_INTERVAL == 0) )
		wake_up_process(queue->dev->clean_task);

	// Move to next buffer
	queue->tail = tail;

	// PPS Check Function
	if( ktime_to_ns(queue->start_enqueue) == 0 ) queue->start_enqueue = ktime_get();
	queue->last_enqueue = ktime_get();
	queue->total_packets++;

	return NULL;
}

struct sk_buff* pfinder_forward(struct sk_buff* skb)
{
	struct net_device* other_dev;
	u32 ret;
	
	other_dev = list_entry( skb->dev->dev_list.next, struct net_device, dev_list );
	if( other_dev == NULL ){
		other_dev = list_entry( skb->dev->dev_list.prev, struct net_device, dev_list );
		if( other_dev == NULL ) dev_kfree_skb_irq(skb);
	}
	
//	other_dev = pfinder_simple_forward(skb->data, other_dev);
	skb_set_dev(skb, other_dev);
	skb_push(skb, ETH_HLEN);
	skb->queue_mapping = 0;

	ret = other_dev->netdev_ops->ndo_start_xmit(skb, other_dev);
	if( unlikely(ret != NETDEV_TX_OK) ) dev_kfree_skb_irq(skb);
	
	return NULL;
}
#ifndef GPU_USE
u16 pfinder_dma_forward( u16 queue_idx, struct ixgbevf_dma_buffer* dma_buffer )
{
	struct pfinder_queue* queue = fast_queue[ queue_idx ];
	struct ixgbevf_tx_dma_buffer* tx_buffer;
	u32 find_limit = PFINDER_TX_LEN >> 4;
	u32 tx_buffer_taken = 0;
	struct net_device* other_dev;
	other_dev = list_entry( queue->dev->netdev->dev_list.next, struct net_device, dev_list );
	if( other_dev == NULL ){
		dma_buffer->len = 0;
		return 0;
	}

	// DMA buffer retrieve
	prefetch( dma_buffer->data );

	// Tx buffer retrieve
	tx_buffer = queue->tx_buffers[ queue->tx_head ];
	while( unlikely(tx_buffer->status != NULL && ((*tx_buffer->status) & cpu_to_le32(0x01)) == 0 ) ){
		if( unlikely(find_limit-- == 0) ) {
			dma_buffer->len = 0;
			queue->drops++;
			break;
		}
		queue->tx_head = (queue->tx_head + 1) % PFINDER_TX_LEN;
		tx_buffer = queue->tx_buffers[ queue->tx_head ];
	}

	if( unlikely(dma_buffer->len) == 0 ) return 0;
	
	//tx_buffer->status = NULL;
	tx_buffer->status = &tx_buffer_taken;

	// Fetch next tx buffer
	queue->tx_head = (queue->tx_head + 1) % PFINDER_TX_LEN;				

	memcpy(tx_buffer->data, dma_buffer->data, dma_buffer->len);
	tx_buffer->len = dma_buffer->len;
	dma_buffer->len = 0;
	
//	other_dev = pfinder_simple_forward(tx_buffer->data, other_dev);
	
	queue->dev->xmit(other_dev, &tx_buffer, 1);
	
	return 0;
}
#endif

int pfinder_device_poll( struct pfinder_dev_list* pf_dev )
{
	struct net_device* netdev = pf_dev->netdev;
	while( !signal_pending(current) && !kthread_should_stop() ){
		netdev->netdev_ops->ndo_do_ioctl( netdev, NULL, PFINDER_IOCTL_POLL );	

		//set_current_state(TASK_INTERRUPTIBLE);
		//msleep_interruptible(1);
		usleep_range(1,1);
		//set_current_state(TASK_RUNNING);
	}
	return 0;
}

void pfinder_device_register( struct net_device* netdev, __be32 represent_ipv4, u8 vnet_id )
{
	int ret = -1;
	struct pfinder_dev_list* pf_dev = NULL;
	struct ifreq ifr;

	//kllaf
	int node = cpu_to_node(smp_processor_id());

	if( !netdev->dev.parent ) return ;
	if( !netdev->dev.parent->driver ) return ;

	// Test if it was registered,
	if( !list_empty( &pfinder_devices ) ){
		list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
			if( pf_dev->netdev == netdev ) break;
		}
		
		if( pf_dev->netdev == netdev ) return ;
	}

	// Pathfinder device registration
	pf_dev = kzalloc( sizeof(struct pfinder_dev_list), GFP_KERNEL );
	if( unlikely(!pf_dev) ) goto error;
	
	// Pathfinder device initialization
	INIT_LIST_HEAD( &pf_dev->dev_list );
	pf_dev->netdev = netdev;
	list_add_tail( &pf_dev->dev_list, &pfinder_devices );



	// kllaf
	pf_dev->represent_ipv4_addr = represent_ipv4;
	pf_dev->vnet_id = vnet_id;
/*
	// Change netdev_ops into virtlink xmit included.
	if( netdev->netdev_ops ) {
		pf_dev->new_netdev_ops = kzalloc_node(sizeof(struct net_device_ops), GFP_KERNEL, node);
		pf_dev->prev_netdev_ops = netdev->netdev_ops;
		memcpy( pf_dev->new_netdev_ops, netdev->netdev_ops, sizeof(struct net_device_ops));
		pf_dev->new_netdev_ops->ndo_start_xmit = &virtlink_encap_skb;
		netdev->netdev_ops = pf_dev->new_netdev_ops;
	}
*/
	// Request encapsulation to driver
	if ( netdev->netdev_ops->ndo_do_ioctl && vnet_id != 0) {
		struct ifreq ifr;
		struct varp_proto_t* varp_proto = kzalloc(sizeof(struct varp_proto_t), GFP_KERNEL);
		varp_proto->other_info.vnet_id = ENCAP_BASE_UDP_PORT + pf_dev->vnet_id;
		ifr.ifr_ifru.ifru_data = varp_proto;
		netdev->netdev_ops->ndo_do_ioctl(netdev, &ifr, SIOCDEVPRIVATE+1);

		// Change netdev_ops into virtlink xmit included.
		if( netdev->netdev_ops ) {
			pf_dev->new_netdev_ops = kzalloc_node(sizeof(struct net_device_ops), GFP_KERNEL, node);
			pf_dev->prev_netdev_ops = netdev->netdev_ops;
			memcpy( pf_dev->new_netdev_ops, netdev->netdev_ops, sizeof(struct net_device_ops));
			pf_dev->new_netdev_ops->ndo_start_xmit = &virtlink_encap_skb;
			netdev->netdev_ops = pf_dev->new_netdev_ops;
		}

		// Change MTU Size
		if (netdev->netdev_ops->ndo_change_mtu ) {
			// Device is knowing as device MTU is normal DATA_LEN(1500)
			// Prevents hardware filters more than its MTU.
			// We should receive 1500 bytes in hardware
			if ( netdev->mtu != ETH_DATA_LEN )
				netdev->netdev_ops->ndo_change_mtu(netdev, ETH_DATA_LEN);
				// Only kernel cares its MTU as 1458
				netdev->mtu = ETH_DATA_LEN - sizeof( struct encaphdr );
		}

	}





	// Make path finder queue
	pf_dev->queue = kzalloc( sizeof(struct pfinder_queue), GFP_KERNEL );
	if( !pf_dev->queue ) goto error;
	
	pf_dev->queue->dev = pf_dev;
	pf_dev->queue->head = 0;
	pf_dev->queue->tail = 0;
#ifdef GPU_USE
	pf_dev->queue->bbuf_clean_use = 0;
	pf_dev->queue->bbuf_xmit_use = 1;
#endif
	fast_queue[ netdev->ifindex ] = pf_dev->queue;
	pf_dev->flag = 0;
	
	// Probe whether this NIC is under controlled by Xebra or not
	if( netdev->netdev_ops->ndo_do_ioctl ){
		// Try polling method
		ret = netdev->netdev_ops->ndo_do_ioctl(netdev, NULL, PFINDER_IOCTL_INTERRUPT);
		if( ret == EBADMSG ){
			//pf_dev->flag = PFINDER_DEV_ADV_BUFFER | PFINDER_DEV_DD_FWD;
			pf_dev->flag = PFINDER_DEV_RT_QUEUE | PFINDER_DEV_DD_FWD | PFINDER_DEV_ADV_BUFFER;
		}
	}

	// Decide whether this device mode is communicate with device driver or not
	if( pf_dev->flag & PFINDER_DEV_ADV_BUFFER ){
		ifr.ifr_name[0] = 1; // Notify this function is for ADV Buffer
	} else if( pf_dev->flag & PFINDER_DEV_DD_FWD ){
		ifr.ifr_name[0] = 0; // Notify this function is for DSP Mode
	}

	// Decide interrupt mode
	ifr.ifr_name[1] = 1;	// Default, RX Interrupt turned on
	ifr.ifr_name[2] = 1;   // Default, TX Interrupt turned on
	if( pf_dev->flag & PFINDER_DEV_ADV_BUFFER ){
	//	ifr.ifr_name[2] = 0; // TX Interrupts are tuned off
	}
	if( pf_dev->flag & PFINDER_DEV_RX_POLL ){
		ifr.ifr_name[1] = 0; // RX Interrupts are tuned on
	}

	// Decide forwarding function
	ifr.ifr_data = NULL;
	if( pf_dev->flag & PFINDER_DEV_ADV_BUFFER ){
		if( pf_dev->flag & PFINDER_DEV_RT_QUEUE )
			ifr.ifr_data = &pfinder_dma_enqueue;
#ifndef GPU_USE
		else
			ifr.ifr_data = &pfinder_dma_forward;
#endif
	} else if( pf_dev->flag & PFINDER_DEV_RT_QUEUE ){
		ifr.ifr_data = &pfinder_enqueue;	
	} else {
		ifr.ifr_data = &pfinder_forward;
	}

	// Do each mode specific action
	if( pf_dev->flag & PFINDER_DEV_ADV_BUFFER ){
		u32 i;
#ifdef GPU_USE
		u32 j;
		for(i=0; i<PFINDER_CLEAN_BATCH_MAX; i++){
			for(j=0; j<PFINDER_CLEAN_BATCH_SIZE; j++){
				// Make tx pre-defined memory pool
				struct ixgbevf_tx_dma_buffer* buf;
				buf = kzalloc( sizeof(struct ixgbevf_tx_dma_buffer), GFP_ATOMIC );
				buf->data = dma_alloc_coherent(netdev->dev.parent, 1518, &buf->dma, GFP_DMA);
				if( buf->data == NULL ){
					printk(KERN_INFO NOTIFY_NAME "Memory allocation failed from %s\n", netdev->name);
					goto error;
				}
			
				buf->dma = cpu_to_le64(buf->dma);
				pf_dev->queue->batch_buffer[i][j] = buf;
			}
		}
#endif	
#ifndef GPU_USE
		for(i=0; i<PFINDER_TX_LEN; i++){
			// Make tx pre-defined memory pool
			struct ixgbevf_tx_dma_buffer* buf;
			buf = kzalloc( sizeof(struct ixgbevf_tx_dma_buffer), GFP_ATOMIC );
			buf->data = dma_alloc_coherent(netdev->dev.parent, 1518, &buf->dma, GFP_DMA);
			if( buf->data == NULL ){
				printk(KERN_INFO NOTIFY_NAME "Memory allocation failed from %s\n", netdev->name);
				goto error;
			}
			
			buf->dma = cpu_to_le64(buf->dma);
			pf_dev->queue->tx_buffers[i] = buf;
		}
#endif
		printk(KERN_INFO NOTIFY_NAME "[%s] Advanced buffer activated.", netdev->name);
	} 

	// Pathfinder offers routing queue for this device
	if( pf_dev->flag & PFINDER_DEV_RT_QUEUE ){
#ifdef GPU_USE
		// Create kernel thread for cleaning socket buffer
		pf_dev->clean_task = kthread_create((void*)pfinder_GPU_clean_queue, pf_dev->queue, "xebra:qclean");
#else
		// Create kernel thread for cleaning socket buffer
		pf_dev->clean_task = kthread_create((void*)pfinder_clean_queue, pf_dev->queue, "xebra:qclean");
#endif

#if 0   // JYJ : fix clean_task affinity to CPU 1

		if( !strcmp(pf_dev->netdev->name, "eth1") )
			kthread_bind(pf_dev->clean_task, 1);
		else if( !strcmp(pf_dev->netdev->name, "eth2") )
			kthread_bind(pf_dev->clean_task, 3);
		else
			kthread_bind(pf_dev->clean_task, 3);
#else
		kthread_bind(pf_dev->clean_task, 1);
#endif
	
		wake_up_process(pf_dev->clean_task);
		printk(KERN_INFO NOTIFY_NAME "[%s] Receive queue ready.", netdev->name);
	}

	// Pathfinder do the polling receive queue of this device
	if( pf_dev->flag & PFINDER_DEV_RX_POLL ){
		// Create kernel thread for receive queue polling
		pf_dev->poll_task = kthread_create((void*)pfinder_device_poll, pf_dev, "xebra:devpoll");
		kthread_bind(pf_dev->poll_task, 0);
		wake_up_process(pf_dev->poll_task);
		printk(KERN_INFO NOTIFY_NAME "[%s] Polling NIC receive buffer.", netdev->name);
	}

	if( pf_dev->flag & (PFINDER_DEV_RX_POLL | PFINDER_DEV_DD_FWD | PFINDER_DEV_ADV_BUFFER) ) {	
		// Let device know the forwarding function and interruptible status
		ret = netdev->netdev_ops->ndo_do_ioctl(netdev, &ifr, PFINDER_IOCTL_INTERRUPT);
		
		if( pf_dev->flag & PFINDER_DEV_ADV_BUFFER ){
			// Device will gives us a xmit function for transmit a packet
			pf_dev->xmit = ifr.ifr_data;
			printk(KERN_INFO NOTIFY_NAME "[%s] Advanced buffer transmit function registered.", netdev->name);
		} else if( pf_dev->flag & PFINDER_DEV_DD_FWD ){
			printk(KERN_INFO NOTIFY_NAME "[%s] NIC will help to handle data from pfinder.", netdev->name);			
		}
	}

	// Device driver helps to receive data
	if( (pf_dev->flag & PFINDER_DEV_DD_FWD) == 0 ){
		// Netdevice Handler Registration
		printk(KERN_INFO NOTIFY_NAME "[%s] netdev_rx_handler registered.", netdev->name);
		netdev_rx_handler_register( netdev, ifr.ifr_data, NULL );
	}

	//kllaf slowpath rx
	netdev->netdev_ops->ndo_do_ioctl(netdev, &ifr, PFINDER_IOCTL_SLOWPATH);
	pf_dev->slowpath_rx = ifr.ifr_data;


	//kllaf
//	netdev_rx_handler_register(netdev, virtlink_decap_skb, NULL);

	dev_hold(netdev);
	return;
	
error:
	if( pf_dev ){
		if( pf_dev->queue ) kfree( pf_dev->queue );
		list_del(&pf_dev->dev_list);
		kfree(pf_dev);
	}
	return;
	
}

void pfinder_device_unregister( struct net_device* netdev, struct pfinder_dev_list* pf_dev )
{
	if( pf_dev == NULL ){
		list_for_each_entry( pf_dev, &pfinder_devices, dev_list ){
			if( pf_dev->netdev == netdev )	break;
		}
	} else {
		netdev = pf_dev->netdev;
	}

	if( pf_dev && netdev ){
		if(pf_dev->vnet_id != 0) {

			// unregister protocol handler
			//netdev_rx_handler_unregister(netdev);

			// unregister netdev operations
			if(netdev->netdev_ops && pf_dev->prev_netdev_ops)
				netdev->netdev_ops = pf_dev->prev_netdev_ops;

			// Request remove encapsulation to driver
			if ( netdev->netdev_ops->ndo_do_ioctl ) {
				struct ifreq ifr;
				ifr.ifr_ifru.ifru_ivalue = 0;
				netdev->netdev_ops->ndo_do_ioctl(netdev, &ifr, SIOCDEVPRIVATE+1);

			}

			// Rewind mtu size
			netdev->mtu += sizeof( struct encaphdr );

			// memory free
			kfree( pf_dev->new_netdev_ops );

		}
	

		if ( (pf_dev->flag & PFINDER_DEV_DD_FWD) == 0 ) {
			// Otherwise, NIC is not controlled on Xebra, so we just clear handler mode
			netdev_rx_handler_unregister(netdev);
		} else {
			struct ifreq ifr;
			ifr.ifr_name[0] = (pf_dev->flag & PFINDER_DEV_ADV_BUFFER)? 1 : 0;
			ifr.ifr_name[1] = 1;
			ifr.ifr_name[2] = 1;
			ifr.ifr_data = NULL;
			netdev->netdev_ops->ndo_do_ioctl(netdev, &ifr, PFINDER_IOCTL_INTERRUPT);
		}
		
		if ( pf_dev->poll_task ){
			if( !task_is_dead(pf_dev->poll_task) || !task_is_stopped(pf_dev->poll_task))
				kthread_stop(pf_dev->poll_task);
		}
		
		if ( pf_dev->clean_task ){
			if( !task_is_dead(pf_dev->clean_task) || !task_is_stopped(pf_dev->clean_task))
				kthread_stop(pf_dev->clean_task);
		}

		if( pf_dev->flag & PFINDER_DEV_ADV_BUFFER ){
			u32 i;
#ifdef GPU_USE
			u32 j;
			for(i=0; i<PFINDER_CLEAN_BATCH_MAX; i++){
				for(j=0; j<PFINDER_CLEAN_BATCH_SIZE; j++){
					// Make tx pre-defined memory pool
					struct ixgbevf_tx_dma_buffer* buf = pf_dev->queue->batch_buffer[i][j];
					if( buf ){
						dma_free_coherent(netdev->dev.parent, 1518, buf->data, cpu_to_le64(buf->dma));
						kfree(buf);
						buf = NULL;
					}
					pf_dev->queue->batch_buffer[i][j] = NULL;
				}
			}
#endif	
#ifndef GPU_USE
			for(i=0; i<PFINDER_TX_LEN; i++){
				// Make tx pre-defined memory pool
				struct ixgbevf_tx_dma_buffer* buf = pf_dev->queue->tx_buffers[i];
			
				if( buf ){
					dma_free_coherent(netdev->dev.parent, 1518, buf->data, cpu_to_le64(buf->dma));
					kfree(buf);
					buf = NULL;
				}
				pf_dev->queue->tx_buffers[i] = NULL;
			}
#endif
		}
		
		if( pf_dev->queue ){
			fast_queue[ pf_dev->netdev->ifindex ] = NULL;
			kfree( pf_dev->queue );
			pf_dev->queue = NULL;
		}
		
		list_del(&pf_dev->dev_list);
		kfree(pf_dev);
		
		dev_put(netdev);
		printk(KERN_INFO NOTIFY_NAME "[%s] unregistered from Xebra device\n", netdev->name);
	}
}

int pfinder_device_notifier( struct notifier_block *unused, unsigned long event, void *ptr ){
	struct net_device *netdev = ptr;
	
	if (!net_eq(dev_net(netdev), &init_net))	return NOTIFY_DONE;

	// eth* will be accepted, eth0 will be declined
	if( strncmp(netdev->name, "eth", 3) || !strcmp(netdev->name, "eth0") ){
		return NOTIFY_DONE;
	}
	
	switch( event ){
		
		case NETDEV_UP:
		//	pfinder_device_register(netdev);
		case NETDEV_REGISTER:
		break;

		case NETDEV_GOING_DOWN:
			pfinder_device_unregister(netdev, NULL);
		case NETDEV_DOWN:
		case NETDEV_REBOOT:
		case NETDEV_UNREGISTER:
		break;
	}
		
	return NOTIFY_DONE;
}

int pfinder_init(void)
{

	printk(KERN_INFO NOTIFY_NAME "Xebra Packet Route Module - Version " VERSION "\n");
	
	pfinder_proc = proc_mkdir("xebra", init_net.proc_net);
	if (!pfinder_proc)	return -ENODEV;

	struct proc_dir_entry *pe;
	pe = proc_create("linkctrl", 0600, pfinder_proc, &pfinder_fops);
	if (pe == NULL) {
		pr_err("ERROR : cannot create linkctrl procfs entry\n");
		proc_net_remove(&init_net, "xebra");
		return -EINVAL;
	}


	INIT_LIST_HEAD( &ptype_handler );
	INIT_LIST_HEAD( &pfinder_devices );

	varp_tbl.head = NULL;
	varp_tbl.tail = NULL;
	varp_tbl.tbl_size = 0;

	register_netdevice_notifier(&device_nb);

    return 0;
}


// initialization sequence : 
// GPU module 
// -> shared memory setting using connect application 
// -> pfinder module
int init_module()
{
#ifdef GPU_USE
// cylee: module integration
	init_GPGPU_module();
	return 0;

#else
	printk(KERN_INFO NOTIFY_NAME "Xebra Packet Route Module - Version " VERSION "\n");
	
	pfinder_proc = proc_mkdir("xebra", init_net.proc_net);
	if (!pfinder_proc)	return -ENODEV;

	struct proc_dir_entry *pe;
	pe = proc_create("linkctrl", 0600, pfinder_proc, &pfinder_fops);
	if (pe == NULL) {
		pr_err("ERROR : cannot create linkctrl procfs entry\n");
		proc_net_remove(&init_net, "xebra");
		return -EINVAL;
	}

	INIT_LIST_HEAD( &ptype_handler );
	INIT_LIST_HEAD( &pfinder_devices );

	varp_tbl.head = NULL;
	varp_tbl.tail = NULL;
	varp_tbl.tbl_size = 0;

	register_netdevice_notifier(&device_nb);
        return 0;
#endif
}

void cleanup_module()
{
	struct pfinder_dev_list* pf_dev;

	#ifdef KU_OSL_VLINK
	virtlink_varp_flush();
	#endif

	pfinder_ptype_flush();
	rtnl_trylock();
	
	while( !list_empty( &pfinder_devices ) ){
		pf_dev = list_first_entry( &pfinder_devices, struct pfinder_dev_list, dev_list );
		pfinder_device_unregister( NULL, pf_dev );
	}

	rtnl_unlock();

	unregister_netdevice_notifier(&device_nb);
	proc_net_remove(&init_net, "xebra");

#ifdef GPU_USE
	// cylee: module integration
	cleanup_GPGPU_module();
#endif

	printk(KERN_INFO NOTIFY_NAME "Cleanup module\n");
}

#ifdef GPU_USE
void export_test(gntpg_list_t* target_gntpg, int req_dev_ifindex, int xmit_size){
	
#if GY_DEBUG
        printk(KERN_INFO "export test : pfinder to savanna\n");
#endif
	pfinder_GPU_xmit(target_gntpg, req_dev_ifindex, xmit_size);

	return;
}

//gy add
void pfinder_GPU_send_IP(gntpg_list_t* target_gntpg, void* data, u32 idx){
	struct iphdr* iph = (struct iphdr*)((u8*)data + ETH_HLEN);
	__be32  daddr = iph->daddr;
	gntpg_t *gnt_sm;
	unsigned int *ip;
	// cylee : error - invalid use of void expression
	gnt_sm = target_gntpg->gntpg_shareMem;
	ip = (unsigned int *)gnt_sm[idx/1024].area->addr;
	ip[idx%1024] = ntohl(daddr);
	//dev_get_by_name("eth%d",1);
	return;
}

//gy add
void pfinder_GPU_xmit(gntpg_list_t* target_gntpg, int req_dev_ifindex, int xmit_size){

	u32 idx = 0;
	int hop_info;
//	int sm_num = 100;

	gntpg_t *gnt_sm;
	unsigned int *dev_if;

	struct net_device* target_dev;
	
	struct neighbour* neigh = NULL;

	struct ethhdr* eth;
	struct iphdr* iph;
	__be32  daddr;

	__be32 be_prefix;

	struct pfinder_entry_ipv4* rt_info = NULL;
	struct pfinder_entry_ipv4* cmp_info = NULL;

	struct pfinder_dev_list* pf_dev;
	struct pfinder_queue* queue;

	struct pfinder_queue* req_queue = fast_queue[req_dev_ifindex];
	int xmit_bbuf_idx = req_queue->bbuf_xmit_use;
	struct ixgbevf_tx_dma_buffer** batch_buffer = req_queue->batch_buffer[xmit_bbuf_idx];
	struct ixgbevf_tx_dma_buffer* temp_buffer;

	struct ixgbevf_tx_dma_buffer*** xmit_buffer = kzalloc(sizeof(struct ixgbevf_tx_dma_buffer**) * PFINDER_MAX_INTERFACES, GFP_ATOMIC);
        u32 count[PFINDER_MAX_INTERFACES];
        int buffer_index = 0;
        int target_index = 0;

        for( ; buffer_index < PFINDER_MAX_INTERFACES; buffer_index++) {
                xmit_buffer[buffer_index] = kzalloc(sizeof(void*) * xmit_size, GFP_ATOMIC);
                count[buffer_index] = 0;
        }
		
	gnt_sm = target_gntpg->gntpg_shareMem;
	for(idx=0;idx<xmit_size;idx++){
		dev_if = (unsigned int *)gnt_sm[target_gntpg->shareMem_num/2+idx/1024].area->addr;
		hop_info = dev_if[idx%1024];
#if GY_DEBUG
		printk(KERN_INFO "xmit func : tx_buffer idx %d\n",idx);
		printk(KERN_INFO "xmit func : hop_info %d\n",hop_info);
#endif
	
		queue = fast_queue[hop_info];
		pf_dev = queue->dev;
		target_dev = pf_dev->netdev;
		temp_buffer = batch_buffer[idx];
	
		eth = (struct ethhdr*)(temp_buffer->data);
		iph = (struct iphdr*)((u8*)eth + ETH_HLEN);
		daddr = iph->daddr;
		neigh = __neigh_lookup(&arp_tbl, &daddr, target_dev, true);

		if( !neigh->hh ){
				printk("<1>" "no hh cache\n");
				pfinder_hh_init( neigh, eth->h_proto );
		} else if( neigh->hh->hh_data ){
#if GY_DEBUG
			printk(KERN_INFO "using cache\n");
#endif
			int hh_len, hh_alen;
			hh_len = neigh->hh->hh_len;
			hh_alen = HH_DATA_ALIGN(hh_len);
			memcpy(temp_buffer->data,((u8 *) neigh->hh->hh_data + 2),	hh_len);
#if GY_DEBUG
			printk(KERN_INFO "destination addr after neghbor look up : %02X:%02X:%02X:%02x:%02x:%02x\n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
#endif
		}else{
			memcpy(eth->h_source, neigh->dev->dev_addr, ETH_ALEN);
			memcpy(eth->h_dest, neigh->ha, ETH_ALEN);
			neigh->dev->header_ops->cache(neigh, neigh->hh);
#if GY_DEBUG
			printk(KERN_INFO "destination addr after neghbor look up : %02X:%02X:%02X:%02x:%02x:%02x\n", eth->h_dest[0], eth->h_dest[1], eth->h_dest[2], eth->h_dest[3], eth->h_dest[4], eth->h_dest[5]);
#endif
		}

		// Search previous rt entry on raw ipv4_rtable
#if GY_DEBUG
		printk(KERN_DEBUG "Search previous rt entry on raw ipv4_rtable\n");
#endif
		list_for_each_entry( cmp_info, pfinder_get_ipv4_rtable(), node ){
			if( 0 < cmp_info->prefix && cmp_info->prefix<=8) be_prefix = 255 >> (8 - cmp_info->prefix);
			else if( 8< cmp_info->prefix && cmp_info->prefix<=16) be_prefix = 65535 >> (16 - cmp_info->prefix);
			else if( 16< cmp_info->prefix && cmp_info->prefix<=24) be_prefix = 16777215 >> (24 - cmp_info->prefix);
			else if( 24< cmp_info->prefix && cmp_info->prefix<=32) be_prefix = 4294967295 >> (32 - cmp_info->prefix);
#if GY_DEBUG
			printk(KERN_DEBUG "cmp_info netdev ifindex : %d\n", cmp_info->netdev->ifindex);
			printk(KERN_DEBUG "target_dev ifindex : %d\n", target_dev->ifindex);
			printk(KERN_DEBUG "cmp_info destination : %d\n", cmp_info->destination);
			printk(KERN_DEBUG "iph daddr : %d\n", (iph->daddr & be_prefix));
#endif
			if( cmp_info->netdev->ifindex == target_dev->ifindex && cmp_info->destination == (iph->daddr & be_prefix) ){
				rt_info = cmp_info;
				break;
			}
		}
#if GY_DEBUG
		printk(KERN_DEBUG "Search previous rt entry on raw ipv4_rtable - end\n");
#endif

		target_index = rt_info->netdev_index;

		if(rt_info == NULL) {
				
#ifdef PF_DEBUG
			printk(KERN_DEBUG "routeinfo == null\n");
#endif
			//broadcasting - To be implemented...
			//continue;
		}
				
#ifdef PF_DEBUG
		printk("<1>" "dest : %d, netdev : %s\n", rt_info->destination, rt_info->netdev->name);
		printk("<1>" "target index : %d", target_dev->ifindex);
#endif

		if (rt_info == NULL || rt_info->is_local) {
		
			//printk("<1>" "pf slowpath\n");
			queue->dev->slowpath_rx(queue->dev->netdev, temp_buffer->data, temp_buffer->len);
			continue;

		} else {
			//printk("<1>" "fast path\n");

			if(pf_dev->netdev == target_dev) {
				if(pf_dev->vnet_id != 0) {
					struct ixgbevf_tx_dma_buffer* temp_buffer2 = xmit_buffer[ ++idx % PFINDER_CLEAN_BATCH_SIZE ];
					if(temp_buffer->len >= 1400) {
						if(unlikely(!ip_fragmentation(temp_buffer, temp_buffer2))) {
							continue;
						} else {
							virtlink_encap_dma_buffer(temp_buffer2, target_dev);
							xmit_buffer[target_index][count[target_index]] = temp_buffer2;
							continue;
						}
					}
					
					virtlink_encap_dma_buffer(temp_buffer, target_dev);
				} else {
					//no encap
					//printk("<1>" "no encap\n");
				}
			}
		}

		xmit_buffer[target_index][count[target_index]++] = temp_buffer;
	}

#if GY_DEBUG
	printk(KERN_INFO "do xmit\n");
#endif
	list_for_each_entry( rt_info, pfinder_get_ipv4_rtable(), node ){
		target_index = rt_info->netdev_index;
		if(count[target_index] != 0 && rt_info->is_local != true){
	        	target_dev = rt_info->netdev;
        		fast_queue[target_dev->ifindex]->dev->xmit(target_dev,xmit_buffer[target_index],count[target_index]);
	        }
	}

	for(buffer_index = 0; buffer_index < PFINDER_MAX_INTERFACES; buffer_index++) {
		kfree(xmit_buffer[buffer_index]);
	}
	kfree(xmit_buffer);

}
#endif

/*
 * 
 */
static inline int pfinder_sysfs_show(struct seq_file *seq, void *v){
	struct pfinder_dev_list* pf_dev;
	struct varp_entry* varp;
	
	seq_puts(seq, "vARP device status\n");
	seq_puts(seq, "Interface\tRepresentative IPv4\tVirtual Network ID\n");

	while (!list_empty(&pfinder_devices)) {
		pf_dev = list_first_entry(&pfinder_devices, struct pfinder_dev_list, dev_list);

		seq_printf(seq, "%s \t\t", pf_dev->netdev->name);
		seq_printf(seq, "%d.%d.%d.%d \t\t",
			(pf_dev->represent_ipv4_addr & 0xFF),
			(pf_dev->represent_ipv4_addr & 0xFF00) >> 8,
			(pf_dev->represent_ipv4_addr & 0xFF0000) >> 16,
			(pf_dev->represent_ipv4_addr & 0xFF000000) >> 24
		);
		seq_printf(seq, "%d\n", pf_dev->vnet_id);
	}


	seq_puts(seq, "\nvARP table\n");
	seq_puts(seq, "Destination\t\tDestination IPv4\tNexthop\n");

	list_walk(varp, varp_tbl.head){
		if( varp == NULL ) break;
		seq_printf(seq, "%02X-%02X-%02X-%02X-%02X-%02X \t",
			varp->dst_ethaddr[0],
			varp->dst_ethaddr[1],
			varp->dst_ethaddr[2],
			varp->dst_ethaddr[3],
			varp->dst_ethaddr[4],
			varp->dst_ethaddr[5]
		);
		
		seq_printf(seq, "%d.%d.%d.%d \t\t",
			(varp->represent_dst_ipv4_addr & 0xFF),
			(varp->represent_dst_ipv4_addr & 0xFF00) >> 8,
			(varp->represent_dst_ipv4_addr & 0xFF0000) >> 16,
			(varp->represent_dst_ipv4_addr & 0xFF000000) >> 24
		);
		
		seq_printf(seq, "%02X-%02X-%02X-%02X-%02X-%02X \n",
			varp->nexthop_ethaddr[0],
			varp->nexthop_ethaddr[1],
			varp->nexthop_ethaddr[2],
			varp->nexthop_ethaddr[3],
			varp->nexthop_ethaddr[4],
			varp->nexthop_ethaddr[5]
		);
	}


	return 0;
}

inline int pfinder_sysfs_open(struct inode *inode, struct file *file)
{
	return single_open(file, pfinder_sysfs_show, PDE(inode)->data);
}

inline ssize_t pfinder_sysfs_write(struct file *file, const char __user * user_buffer, size_t count, loff_t *ppos)
{
	#define CMD_REGISTER			1
	#define CMD_UNREGISTER			2
	#define CMD_REGISTER_IPv4		3
	#define CMD_ROUTING_UPDATE_TEST 4

	int cur_idx = 0;
	u8 command = 0;
	u8 if_name[IFNAMSIZ+5] = {0, };

	__be32 represent_ip = 0;
	u8   vnet_id = 0;

	__be32 daddr = 0;
	__be32 gateway = 0;
	u8 prefix = 0;
	u16 rt_cmd = 0;
	

	while( count > cur_idx )
	{
		char opt_name[40] = {0, };
		get_next_arg( user_buffer, opt_name, sizeof(opt_name), count, &cur_idx);	
		if(!strcmp(opt_name, "REGISTER")) {
			command = CMD_REGISTER;
		} else if(!strcmp(opt_name, "UNREGISTER")) {
			command = CMD_UNREGISTER;
		} else if(!strcmp(opt_name, "REGISTER_IPv4")) {
			command = CMD_REGISTER_IPv4;
		} else if(!strcmp(opt_name, "ROUTING_UPDATE_TEST"))
		{
			command = CMD_ROUTING_UPDATE_TEST;
		} else if (!strcmp(opt_name, "add")){
			rt_cmd = SIOCADDRT;
		} else if (!strcmp(opt_name, "del")) {
			rt_cmd = SIOCDELRT;
		} else if (!strcmp(opt_name, "dev")) {
			get_next_arg( user_buffer, if_name, sizeof(if_name), count, &cur_idx);
		} else if(!strcmp(opt_name, "represent_ip")){
			u8 str_ipaddr[4*4] = {0, };
			get_next_arg( user_buffer, str_ipaddr, sizeof(str_ipaddr), count, &cur_idx);
			represent_ip = in_aton(str_ipaddr);
		} else if(!strcmp(opt_name, "vnet_id")){
			u8 str_vnet_id[4] = {0, };
			get_next_arg( user_buffer, str_vnet_id, sizeof(str_vnet_id), count, &cur_idx);
			vnet_id = atoi(str_vnet_id);
		} else if(!strcmp(opt_name, "dest_addr")){
			u8 str_daddr[4*4] = {0, };
			get_next_arg( user_buffer, str_daddr, sizeof(str_daddr), count, &cur_idx);
			daddr = in_aton(str_daddr);
		} else if(!strcmp(opt_name, "gateway")){
			u8 str_gateway[4] = {0, };
			get_next_arg( user_buffer, str_gateway, sizeof(str_gateway), count, &cur_idx);
			gateway = atoi(str_gateway);
		} else if(!strcmp(opt_name, "prefix")){
			u8 str_prefix[4] = {0, };
			get_next_arg( user_buffer, str_prefix, sizeof(str_prefix), count, &cur_idx);
			prefix = atoi(str_prefix);
		}
		else if( strlen(opt_name) == 0 ){
			break;
		}
	}

	switch( command ){
		case CMD_REGISTER:
			//VLINK setting command from user echo
			if( strlen(if_name) > 0 && represent_ip > 0 && vnet_id >= 0 ){
				//register it as virtlink module attached.
			
				rtnl_lock();
				pfinder_device_register(dev_get_by_name(&init_net, if_name), represent_ip, vnet_id);
				rtnl_unlock();
			}
		break;
		case CMD_UNREGISTER:
			if(strlen(if_name) > 0){
				rtnl_lock();
				//unregister_encapsulation(dev_get_by_name(&init_net, if_name));
				rtnl_unlock();
			}
		break;
		case CMD_REGISTER_IPv4:
			pfinder_ipv4_register();
		break;
		case CMD_ROUTING_UPDATE_TEST:
			printk("CMD_ROUTING_UPDATE_TEST\n");
			printk("rt_cmd : %d\n",rt_cmd);
			
			if(rt_cmd > 0)
			{
				pfinder_ipv4_update( rt_cmd, pfinder_ipv4_entry_alloc(daddr, gateway, prefix, if_name) );
			}
		break;
	}
	return count;
}

MODULE_AUTHOR("Korea University Xebra Team <xebra@os.korea.ac.kr>");
MODULE_DESCRIPTION("Xebra Packet Route Module: pfinder");
MODULE_LICENSE("GPL");
MODULE_VERSION(VERSION);
