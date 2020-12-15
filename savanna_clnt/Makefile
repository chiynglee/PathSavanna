# modified for module integration between pfinder and savanna_clnt

#obj-m := savanna_clnt_mod.o

#KDIR := /lib/modules/$(shell uname -r)/build
#PWD := $(shell pwd)

#default:
#	make -C $(KDIR) M=$(PWD) modules
#clean:
#	make -C $(KDIR) M=$(PWD) clean
##############################


# GPGPU client application compile

OBJ := gpulaunch
CFILES := savanna_clnt_gpulaunch.c

default:
	gcc -o $(OBJ) $(CFILES)
clean:
	rm -f $(OBJ)
