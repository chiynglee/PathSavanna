opcontrol --stop
opreport --symbols --image-path=/root/PathSavanna2.2_mxq/pfinder > ./profile/pf_$1.txt
opreport --symbols --image-path=/root/PathSavanna2.2_mxq/savanna_clnt > ./profile/sa_$1.txt
opreport --symbols --image-path=/root/ixgbevf-2.0.0-xebra-fromPV1/src/ > ./profile/vf_$1.txt
opcontrol --shutdown
