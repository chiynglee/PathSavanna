#include <linux/ctype.h>
#include "xbroute.h"

extern u8 xbroute_ipv4_update( void* update_info );
extern void* xbroute_ipv4_entry_alloc( __be32 daddr, __be32 gateway, u8 prefix, const char* if_name );
extern void xbroute_ipv4_show(struct seq_file *seq);

int hex_to_bin(char ch)
{
        if ((ch >= '0') && (ch <= '9'))
                return ch - '0';
        ch = tolower(ch);
        if ((ch >= 'a') && (ch <= 'f'))
                return ch - 'a' + 10;
        return -1;
}

static inline int count_trail_chars(const char __user * user_buffer,
			     unsigned int maxlen)
{
	int i;

	for (i = 0; i < maxlen; i++) {
		char c;
		if (get_user(c, &user_buffer[i]))
			return -EFAULT;
		switch (c) {
		case '\"':
		case '\n':
		case '\r':
		case '\t':
		case ' ':
		case '=':
			break;
		default:
			goto done;
		}
	}
done:
	return i;
}

static inline int strn_len(const char __user * user_buffer, unsigned int maxlen)
{
	int i;

	for (i = 0; i < maxlen; i++) {
		char c;
		if (get_user(c, &user_buffer[i]))
			return -EFAULT;
		switch (c) {
		case '\"':
		case '\n':
		case '\r':
		case '\t':
		case ' ':
		case '=':
			goto done_str;
			break;
		default:
			break;
		}
	}
done_str:
	return i;
}


static inline int get_next_arg( const char __user * user_buffer, char* opt, size_t opt_size, size_t count, int *cur_idx)
{
	int len;
	if( count < 1 ) goto out;

	// skip trail characters
	len = count_trail_chars(&user_buffer[*cur_idx], count-*cur_idx);
	if( len < 0 ) goto out;
	*cur_idx += len;

	// Check option name length
	len = strn_len(&user_buffer[*cur_idx], opt_size - 1);
	if (len < 0) goto out;

	// copy to buffer
	if (copy_from_user(opt, &user_buffer[*cur_idx], len))	goto out;
	*cur_idx += len;

out:
	return 0;
}

static inline int atoi(const char *name)
{
    int val = 0;
    for (;; name++) {
        switch (*name) {
            case '0'...'9':
                val = 10 * val + (*name-'0');
                break;
            default:
                return val;
        }
    }
    return val;
}


/*
 * Function that show Receiver statistics
 */
static inline int xbroute_sysfs_show(struct seq_file *seq, void *v)
{
	seq_puts(seq, "Xebra Routing Table\n\n");
	xbroute_ipv4_show(seq);
	return 0;
}

inline int xbroute_sysfs_open(struct inode *inode, struct file *file)
{
	return single_open(file, xbroute_sysfs_show, PDE(inode)->data);
}

inline ssize_t xbroute_sysfs_write(struct file *file, const char __user * user_buffer,
				size_t count, loff_t *ppos)
{
	#define CMD_ADD 1
	#define CMD_REMOVE 2
	#define CMD_FLUSH 3
	
	int cur_idx = 0;
	u8 command = 0;
	u8 if_name[IFNAMSIZ+5] = {0, };

	__be32 network = 0;
	__be32 gateway = 0;
	u8 prefix = 0;
	

	while( count > cur_idx ){
		char opt_name[40] = {0, };
		get_next_arg( user_buffer, opt_name, sizeof(opt_name), count, &cur_idx);
		
		if(!strcmp(opt_name, "add")){
			command = CMD_ADD;
		}
		else if(!strcmp(opt_name, "remove")){
			command = CMD_REMOVE;
		}
		else if (!strcmp(opt_name, "dev")) {
			get_next_arg( user_buffer, if_name, sizeof(if_name), count, &cur_idx);
		}
		else if(!strcmp(opt_name, "net")){
			u8 str_ipaddr[3*6+1] = {0, }; /*255.255.255.255/32*/
			u8 idx = 0;
			get_next_arg( user_buffer, str_ipaddr, sizeof(str_ipaddr), count, &cur_idx);
			while( str_ipaddr[idx] && (str_ipaddr[idx] != '/') ) idx++;
			
			if(idx > 0 && idx < sizeof(str_ipaddr)){ // If it was defined the prefix in the form of slash
				prefix = atoi(&str_ipaddr[idx+1]);
				str_ipaddr[idx] = 0;
			}
			network = in_aton(str_ipaddr) & ~(0xFFFFFFFF >> prefix);
		}
		else if(!strcmp(opt_name, "prefix")){
			u8 str_prefix[3] = {0, };
			get_next_arg( user_buffer, str_prefix, sizeof(str_prefix), count, &cur_idx);
			prefix = atoi(str_prefix);
		}
		else if(!strcmp(opt_name, "gw")){
			u8 str_ipaddr[3*5+1] = {0, };
			get_next_arg( user_buffer, str_ipaddr, sizeof(str_ipaddr), count, &cur_idx);
			gateway = in_aton(str_ipaddr);
		}
		else if( strlen(opt_name) == 0 ){
			break;
		}
	}

	switch( command ){
		case CMD_ADD:
		{
			if( network && strlen(if_name) )
				xbroute_ipv4_update( xbroute_ipv4_entry_alloc(network, gateway, prefix, if_name) );
		}
		break;

		case CMD_REMOVE:
		break;

		case CMD_FLUSH:
		break;
	}

	return count;
}
