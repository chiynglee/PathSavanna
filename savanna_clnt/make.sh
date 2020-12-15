#!/bin/bash
gcc -o gpulaunch savanna_clnt_gpulaunch.c
rmmod savanna_clnt_mod.ko
make clean
make
insmod savanna_clnt_mod.ko
dmesg -c
#clear
#ls
exit 0
