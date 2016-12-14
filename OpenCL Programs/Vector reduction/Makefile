CC = gcc -Wall
INC_GPU = -I. -I$CUDADIR/include
LD_GPU  = -L$CUDADIR/lib64 -lOpenCL

all: sumReductionGPU 

sumReductionGPU : sumReductionGPU.c 
	$(CC) $(INC_GPU) $(LD_GPU) -o $@ sumReductionGPU.c

clean : 
	/bin/rm -f *.o sumReductionGPU 
