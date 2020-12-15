#ifndef LINUX_VERSION_CODE
	#include <linux/version.h>
#else
	#define KERNEL_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#endif

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,36))
	#include <linux/if_bridge.h>
	struct net_bridge_port;
	
	struct sk_buff* (*previous_br_handle)(struct net_bridge_port* p, struct sk_buff* skb); //  previous bridge handle function pointer place
	struct sk_buff* (*br_handler)(struct sk_buff* skb);
	
	struct sk_buff* xbroute_br_hook( struct net_bridge_port* p, struct sk_buff* skb )
	{
		return previous_br_handle(p, br_handler(skb));
	}

	int xbroute_br_hook_register( struct net_device* netdev, struct sk_buff* (*func)(struct sk_buff*) )
	{
		rtnl_unlock();
		
		// Previous hook function will be saved in the place.
		if( br_handler == NULL ){
			previous_br_handle = br_handle_frame_hook;
			br_handler = func;
			rcu_assign_pointer(br_handle_frame_hook, xbroute_br_hook);
		}
		
		return 1;
	}

	int xbroute_br_hook_unregister( struct net_device* netdev )
	{
		rtnl_unlock();

		if( br_handler != NULL ){
			rcu_assign_pointer(br_handle_frame_hook, previous_br_handle);
			previous_br_handle = NULL;
			br_handler = NULL;
		}

		return 1;
	}
#endif

#if (LINUX_VERSION_CODE < KERNEL_VERSION(2,6,36))
	#define netdev_rx_handler_unregister(netdev) xbroute_br_hook_unregister(netdev)
	#define netdev_rx_handler_register(netdev, func, unused) xbroute_br_hook_register(netdev, func)
#endif

