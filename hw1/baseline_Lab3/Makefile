.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib -lcudart
CC	:= nvcc
DEP	:=  
ifeq ($(def), )
DEF := 
else
DEF := -D$(def)
endif

NVCCFLAGS	:= -lineinfo -arch=sm_53 --ptxas-options=-v -g 

all:	const shared global

const:	cmpe297_matMul_const.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o matrixMul_const cmpe297_matMul_const.o $(LIB)

shared:	cmpe297_matMul_shared.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o matrixMul_shared cmpe297_matMul_shared.o $(LIB)

global:	cmpe297_matMul_global.o $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o matrixMul_global cmpe297_matMul_global.o $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) $(DEF) -c $< -o $@
	

clean:
	rm -f *.o matrixMul_*


