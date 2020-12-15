DIR_PF := pfinder
DIR_GPU_CLNT := savanna_clnt

OBJ_MOD := pfinder.ko
OBJ_APP := gpulaunch

default:
	cd $(DIR_PF); make; cp $(OBJ_MOD) ../$(OBJ_MOD); cd ..
	cd $(DIR_GPU_CLNT); make; cp $(OBJ_APP) ../$(OBJ_APP); cd ..

clean:
	rm -f $(OBJ_MOD) $(OBJ_APP)
	cd $(DIR_PF); make clean; cd ..
	cd $(DIR_GPU_CLNT); make clean; cd ..
