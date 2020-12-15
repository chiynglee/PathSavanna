modprobe oprofile timer=1
opcontrol --reset
opcontrol --start --xen=/xen-syms --vmlinux=/lib/modules/2.6.37.1/source/vmlinux

#opcontrol --start --xen=/xen-syms --no-vmlinuxa
