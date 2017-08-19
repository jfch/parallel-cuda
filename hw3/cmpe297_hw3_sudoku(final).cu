// Simple SUDOKU probram in CUDA
// cmpe297_hw3_easysudoku.cu


/* *
 HW3 by Jiongfeng Chen

Parallelized GPU version of Sudoku Puzzle Algorithm and Its OPTIMIZATION

OPTIMIZATION Thinking:		
	ADD possible value array to save sesults from the last run to reduce iteration times 
   	So it only compare the possible values found from last run, and largely reduce the i
   	teration times of the loop in the following kernel code. It is extremely useful when 
   	the IS_OPTION is complex and require much excution time.
   	    temp = IS_OPTION(row, col, k); //old
   		temp = IS_OPTION(row, col, value_avail[mIter]); //optimized
					
Test Case: 
 	input matrix:
 	const int input_sdk[9][9] 
 		   =  {{0, 7, 0, 0, 6, 5, 0, 8, 0},
			  {6, 0, 0, 0, 3, 0, 4, 0, 0},
			  {0, 2, 0, 0, 4, 0, 7, 0, 0},
			  {8, 6, 0, 0, 0, 2, 5, 7, 0},
			  {0, 0, 7, 4, 0, 6, 1, 0, 0},
			  {0, 5, 2, 3, 0, 0, 0, 6, 4},
			  {0, 0, 8, 0, 2, 0, 0, 3, 0},
			  {0, 0, 5, 0, 8, 0, 0, 0, 1},
			  {0, 4, 0, 7, 1, 0, 0, 5, 0}};
Output:
	  0    *7*    0   |   0    *6*   *5*  |   0    *8*    0   
	 *6*    0     0   |   0    *3*    0   |  *4*    0     0   
	  0    *2*    0   |   0    *4*    0   |  *7*    0     0   
	---------------------------------------------------------------------
	 *8*   *6*    0   |   0     0    *2*  |  *5*   *7*    0   
	  0     0    *7*  |  *4*    0    *6*  |  *1*    0     0   
	  0    *5*   *2*  |  *3*    0     0   |   0    *6*   *4*  
	---------------------------------------------------------------------
	  0     0    *8*  |   0    *2*    0   |   0    *3*    0   
	  0     0    *5*  |   0    *8*    0   |   0     0    *1*  
	  0    *4*    0   |  *7*   *1*    0   |   0    *5*    0   

	 *4*   *7*   *1*  |  *9*   *6*   *5*  |  *3*   *8*   *2*  
	 *6*   *8*   *9*  |  *2*   *3*   *7*  |  *4*   *1*   *5*  
	 *5*   *2*   *3*  |  *8*   *4*   *1*  |  *7*   *9*   *6*  
	---------------------------------------------------------------------
	 *8*   *6*   *4*  |  *1*   *9*   *2*  |  *5*   *7*   *3*  
	 *3*   *9*   *7*  |  *4*   *5*   *6*  |  *1*   *2*   *8*  
	 *1*   *5*   *2*  |  *3*   *7*   *8*  |  *9*   *6*   *4*  
	---------------------------------------------------------------------
	 *9*   *1*   *8*  |  *5*   *2*   *4*  |  *6*   *3*   *7*  
	 *7*   *3*   *5*  |  *6*   *8*   *9*  |  *2*   *4*   *1*  
	 *2*   *4*   *6*  |  *7*   *1*   *3*  |  *8*   *5*   *9*  

	Kernel Execution Time: 37960 cycles
	Total cycles: 37960 

	
run:nvcc -I/usr/local/cuda/include -I. -lineinfo -arch=sm_53 -g -c cmpe297_hw3_sudoku.cu -o cmpe297_hw3_sudoku.o

*/


#include<stdio.h>
#include<string.h>
#include <cuda_runtime.h>

const int big_2x[9][9] = {{1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1}};

// input 9x9 sudoku : 
// - 1~9 : valid values 
// - 0 : no value is decided
const int input_sdk[9][9] =  {{0, 7, 0, 0, 6, 5, 0, 8, 0},
			  {6, 0, 0, 0, 3, 0, 4, 0, 0},
			  {0, 2, 0, 0, 4, 0, 7, 0, 0},
			  {8, 6, 0, 0, 0, 2, 5, 7, 0},
			  {0, 0, 7, 4, 0, 6, 1, 0, 0},
			  {0, 5, 2, 3, 0, 0, 0, 6, 4},
			  {0, 0, 8, 0, 2, 0, 0, 3, 0},
			  {0, 0, 5, 0, 8, 0, 0, 0, 1},
			  {0, 4, 0, 7, 1, 0, 0, 5, 0}};
typedef struct {
	int val[9][9]; // values that each entry can get
	int num_options[9][9]; // number of values that each entry can get
	int not_in_cell[9][9];	// values not in each 3x3 cell
	int not_in_row[9][9];	// values not in each row
	int not_in_col[9][9];	// values not in each column
} stContext;
stContext context;

void initialize_all();
void print_all();

#define WIDTH	9

#define IS_OPTION(row, col, k) \
			((shared_not_in_row[row][k] == 1) && \
			(shared_not_in_col[col][k] == 1) && \
			(shared_not_in_cell[row/3+(col/3)*3][k] == 1))? 1 : 0;

__device__ int
memcmp(stContext *input, int width)
{
	for(int i = 0; i< width; i++)
		for(int j = 0; j< width; j++)
			if(input->num_options[i][j] != 1)
				return 1;
	return 0;
}

#define FINISHED()	(memcmp(context, WIDTH) == 0? 1: 0)

// rule: numbers should be unique in each sub-array, each row, and each column
__global__ void k_Sudoku(stContext *context, int *runtime)
{
    const unsigned int col = threadIdx.x;
    const unsigned int row = threadIdx.y;
    
    //share memory to improve performance
		__shared__ int shared_val[WIDTH][WIDTH];
		__shared__ int shared_num_options[WIDTH][WIDTH];
		__shared__ int shared_not_in_cell[WIDTH][WIDTH];
		__shared__ int shared_not_in_row[WIDTH][WIDTH];
		__shared__ int shared_not_in_col[WIDTH][WIDTH];
		
		
		shared_val[row][col] = context->val[row][col];
		shared_num_options[row][col] = context->num_options[row][col];
		shared_not_in_cell[row][col] = context->not_in_cell[row][col];
		shared_not_in_row[row][col] = context->not_in_row[row][col];
		shared_not_in_col[row][col] = context->not_in_col[row][col];
		__syncthreads();
		
	    		
	//CLOCK: Measure the performance AFTER OPTIMIZATION
	int start_time = clock64();
    
    //printf("col %d row %d threads\n", col, row);
	while(!FINISHED())
	{
		//printf("again col %d row %d threads\n", col, row);		

		if(shared_num_options[row][col] > 1)
		{
					// Find values that are not in the row, col, and the 
					// 3x3 cell that (row, col) is belonged to.			
					
					
					int value = 0, temp;
					shared_num_options[row][col] = 0;
					
					//OPTIMIZATION, ADD possible value to save 
					//results from the last run to reduce iteration times 
					int value_avail[9];
					for(int kIter = 0; kIter < 9; kIter++)
						 value_avail[kIter] = 0;
						 
					//BEFORE OPTIMIZATION
					/*for(int k = 0; k < 9; k++)
					{
						temp = IS_OPTION(row, col, k);
						if(temp == 1)
						{
							shared_num_options[row][col]++;
							value = k;

						}
					}
					*/					
					//AFTER OPTIMIZATION, ADD possible value to save 
					//results from the last run to reduce iteration times
					int firstRun=1;
					int nIter =0;
					if(firstRun==1){	
						for(int k = 0; k < 9; k++)
						{
							temp = IS_OPTION(row, col, k);
							if(temp == 1)
							{
								shared_num_options[row][col]++;
								value = k;
								value_avail[nIter++] =value;
							}
						}
						firstRun=0;
					}
					else{
						int mIter=0;
						while(value_avail[mIter]!=0)
						{	//OPTIMIZATION
							//only compare the possible values found from last run
							temp = IS_OPTION(row, col, value_avail[mIter]);
							if(temp == 1)
							{
								shared_num_options[row][col]++;
								value = value_avail[mIter];
							}
							mIter++;
						}	
						value_avail[mIter] = 0;					
					}
					// If the above loop found only one value, 
					// set the value to (row, col)
					if(shared_num_options[row][col] == 1)
					{
						shared_not_in_row[row][value] = 0;
						shared_not_in_col[col][value] = 0;
						shared_not_in_cell[(row)/3+((col)/3)*3][value] = 0;
						shared_val[row][col] = value+1;
					}
		}
		context->num_options[row][col] = shared_num_options[row][col];
		__syncthreads();
	}//end while, find all grids
   
   	context->val[row][col] = shared_val[row][col];
	context->num_options[row][col] = shared_num_options[row][col];
	context->not_in_cell[row][col] = shared_not_in_cell[row][col];
	context->not_in_row[row][col] = shared_not_in_row[row][col];
	context->not_in_col[row][col] = shared_not_in_col[row][col];
	__syncthreads();

	//CLOCK: Measure the performance AFTER OPTIMIZATION
	int stop_time = clock64();
	runtime[row*WIDTH+col] = (int)(stop_time - start_time);
		

}

int main(int argc, char **argv)
{
    cudaError_t err;

    initialize_all();
    print_all();

    stContext *k_context; //device matrix
    
	//CLOCK: Measure the performance AFTER OPTIMIZATION
	int* runtime; 				/*Exection cycles*/
	int* d_runtime;
	int runtime_size = WIDTH*WIDTH*sizeof(int); 
    cudaMalloc((void**)&d_runtime, runtime_size);	
	runtime = (int *) malloc(runtime_size);
	memset(runtime, 0, runtime_size);
	
	// TODO: Allocate matrix in GPU device memory 
	// Print the matrix size to be used, and compute its size
    int size = 5*WIDTH*WIDTH*sizeof(int);
    //printf("[MatrixMul of %d x %d elements]\n", WIDTH, WIDTH);
    err = cudaMalloc((void**)&k_context, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // TODO: Copy the input matrix to GPU
    err = cudaMemcpy(k_context, &context, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Assign as many threads as the matrix size so that
    // each thread can deal with one entry of the matrix
    dim3 dimBlock(WIDTH, WIDTH, 1);
    dim3 dimGrid(1, 1, 1);

	
    // TODO: Call the kernel function
    k_Sudoku<<<dimGrid,dimBlock>>>(k_context, d_runtime);
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();
	
	// TODO: Copy the result matrix from the GPU device memory
	err = cudaMemcpy(&context, k_context, size, cudaMemcpyDeviceToHost); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();
	
	//CLOCK: Measure the performance AFTER OPTIMIZATION
	//Copy the execution times from the GPU memory to HOST Code
	cudaMemcpy(runtime, d_runtime, runtime_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize(); 
    //unsigned long long  elapsed_time = 0;
    int elapsed_time = 0;
    for(int i = 0; i < WIDTH*WIDTH; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];
	
    // Print the result
    print_all();

	//printf("Kernel Execution Time: %llu cycles\n", elapsed_time);
	printf("Kernel Execution Time: %d cycles\n", elapsed_time);
	printf("Total cycles: %d \n", elapsed_time);

	// Free host memory
	free(runtime);

    // Free the device memory
    err = cudaFree(k_context);
    err = cudaFree(d_runtime);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free gpu data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    getchar();
	
    return 0;
}

void initialize_all()
{
    int i, j;

    memcpy(context.not_in_cell,big_2x, sizeof(big_2x));	
    memcpy(context.not_in_row,big_2x, sizeof(big_2x));	
    memcpy(context.not_in_col,big_2x, sizeof(big_2x));	
		
    for(i=0; i<9; i++){
    	for(j=0; j<9; j++){
            if(input_sdk[i][j] == 0)
            {
                context.val[i][j] = 0;
                context.num_options[i][j]=9;
            }
            else
            {
                context.val[i][j] = input_sdk[i][j];
                context.num_options[i][j] = 1;
                context.not_in_cell[i/3+(j/3)*3][input_sdk[i][j]-1] = 0;
                context.not_in_col[j][input_sdk[i][j]-1] = 0;
                context.not_in_row[i][input_sdk[i][j]-1] = 0;
            }
        }
    }
}


void print_all()
{
    int i, j, k;

    for(i=0; i<9; i++){
        for(j=0; j<9; j++){
            if(context.val[i][j] == 0)
                fprintf(stdout, "  %1d   ", context.val[i][j]);  
            else
                fprintf(stdout, " *%1d*  ", context.val[i][j]);  
            if((j==2)||(j==5)){
                fprintf(stdout, "| ");	
            }
        }
        fprintf(stdout, "\n");	
        if((i==2)||(i==5)){
            for(k=0; k<69; k++){
                fprintf(stdout, "-");	
            }
            fprintf(stdout, "\n");	
        }
    }
    fprintf(stdout, "\n");
}

