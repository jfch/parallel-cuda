.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib 
CC	:= nvcc
OBJS	:= cmpe297_hw3_sudoku.o 
DEP	:=  

NVCCFLAGS	:= -lineinfo -arch=sm_53 -g

all:	sudoku	

sudoku:	$(OBJS) $(DEP)
	$(CC) $(INC) $(NVCCFLAGS) -o sudoku $(OBJS) $(LIB)

.cpp.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@ 

.cu.o:
	$(CC) $(INC) $(NVCCFLAGS) -c $< -o $@
	

clean:
	rm -f *.o sudoku


