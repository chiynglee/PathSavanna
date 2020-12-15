#!/bin/bash
rmmod savanna_serv.ko
make clean
make
insmod savanna_serv.ko
dmesg -c
#clear
#ls
exit 0
