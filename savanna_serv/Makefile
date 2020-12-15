obj-m := savanna_serv.o

savanna_serv-y := savanna_serv_mod.o 

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

default:
	make -C $(KDIR) M=$(PWD) modules
clean:
	make -C $(KDIR) M=$(PWD) clean
