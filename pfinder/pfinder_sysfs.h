#include <linux/ctype.h>  /* for tolower , toupper when we convert hexa into bin */
/*
int hex_to_bin(char ch)
{
        if ((ch >= '0') && (ch <= '9'))
                return ch - '0';
        ch = tolower(ch);
        if ((ch >= 'a') && (ch <= 'f'))
                return ch - 'a' + 10;
        return -1;
}
*/
static inline int count_trail_chars(const char __user * user_buffer, unsigned int maxlen)
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

#ifdef KU_OSL_VLINK
static void hexmac_to_binmac( u8* bin_mac, u8* str_mac ){
	int i;
	for(i=0; i<ETH_ALEN; i++){
		bin_mac[i] = (hex_to_bin( str_mac[i*3] ) << 4) | (hex_to_bin( str_mac[i*3+1] ));
	}
}
/*
int atoi(const char *name)
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
}*/

#endif

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
