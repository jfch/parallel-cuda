// CMPE297-6 HW2
// CUDA version Rabin-Karp
/* *
 HW2 by Jiongfeng Chen

Parallelize the sequential version of Rabin-Karp String matching algorithm
on GPU: Searching for single pattern in the input

Test Case: 
 	input string:"HEABAL"
 	pattern :"AB";
Output:
	Kernel Execution Time: 1546 cycles
	Total cycles: 1546 
	Kernel Execution Time: 1546 cycles
	Searching for a single pattern in a single string
	Print at which position the pattern was found
	Input string: HEABAL
	Pattern: AB
	Pos: 0 Result: 0
	Pos: 1 Result: 0
	Pos: 2 Result: 1
	Pos: 3 Result: 0
	Pos: 4 Result: 0


run: nvcc -I/usr/local/cuda/include -I. -lineinfo -arch=sm_53 --ptxas-options=-v -g   -c cmpe297_hw2_rabin_karp_singlePattern.cu -o cmpe297_hw2_rabin_karp_singlePattern.o

*/

#include<stdio.h>
#include<iostream>
#include <cuda_runtime.h>

/*ADD CODE HERE: Implement the parallel version of the sequential Rabin-Karp*/
__device__ int
memcpy(char* input, int index, char* pattern, int pattern_length)
{
	for(int i = 0; i< pattern_length; i++)
		if(pattern[i] != input[index+i])
			return 1;
	return 0;
}

__global__ void
findIfExistsCu(char* input, int input_length, char* pattern, int pattern_length, int patHash, int* result, int *runtime)
{
	int start_time = clock64();
	
	int tid = threadIdx.x;
	int inputHash = 0;	
	
	for(int i = tid; i< tid+pattern_length; i++)
		inputHash = (inputHash*256 + input[i]) % 997;
	if(inputHash == patHash && memcpy(input,tid,pattern,pattern_length)==0)
		result[tid]=1;
	else
		result[tid]=0;
		
	int stop_time = clock64();
	runtime[tid] = (int)(stop_time - start_time);
}

int main()
{

	// host variables
	char input[] = "HEABAL"; 	/*Sample Input*/
	char pattern[] = "AB"; 		/*Sample Pattern*/
	int patHash = 0; 			/*hash for the pattern*/
	int* result; 				/*Result array*/
	int* runtime; 				/*Exection cycles*/
	int pattern_length = 2;		/*Pattern Length*/
	int input_length = 6; 		/*Input Length*/
	/*ADD CODE HERE*/;
	int match_times = input_length - pattern_length +1; 		/*Match Times*/
 	cudaError_t err = cudaSuccess;
 
	// device variables
	char* d_input;
	char* d_pattern;
	int* d_result;
	int* d_runtime;

	// measure the execution time by using clock() api in the kernel as we did in Lab3
	/*ADD CODE HERE*/;
	int runtime_size = match_times*sizeof(int); 
    cudaMalloc((void**)&d_runtime, runtime_size);	
	runtime = (int *) malloc(runtime_size);
	memset(runtime, 0, runtime_size);
	
	result = (int *) malloc((match_times)*sizeof(int));

	/*Calculate the hash of the pattern*/
	for (int i = 0; i < pattern_length; i++)
    {
        patHash = (patHash * 256 + pattern[i]) % 997;
    }
	printf("patHash %d \n", patHash);
	
	/*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/
	int size = input_length*sizeof(char);
  	err = cudaMalloc((void**)&d_input, size);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device d_input(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Copy input string from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
    
    size = pattern_length*sizeof(char);
	err = cudaMalloc((void**)&d_pattern, size);
	printf("Copy pattern string from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_pattern, pattern, size, cudaMemcpyHostToDevice);
    
	size = match_times*sizeof(int);
	err = cudaMalloc((void**)&d_result, size);

	/*ADD CODE HERE: Launch the kernel and pass the arguments*/
	int blocksPerGrid = 1;// FILL HERE
    int threadsPerBlock = match_times;// FILL HERE
	findIfExistsCu<<<blocksPerGrid, threadsPerBlock>>>(d_input, input_length, d_pattern, pattern_length, patHash, d_result,d_runtime);

		
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	cudaThreadSynchronize();	
	
	/*ADD CODE HERE: COPY the result and print the result as in the HW description*/
	// Copy the device result device memory to the host result matrix
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();
    
	/*ADD CODE HERE: Copy the execution times from the GPU memory to HOST Code*/
	 cudaMemcpy(runtime, d_runtime, runtime_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize(); 
	/*RUN TIME calculation*//*ADD CODE HERE:*/
    unsigned long long elapsed_time = 0;
    for(int i = 0; i < input_length-pattern_length; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];

	//Print
    printf("Kernel Execution Time: %llu cycles\n", elapsed_time);
	printf("Total cycles: %d \n", elapsed_time);
	printf("Kernel Execution Time: %d cycles\n", elapsed_time);
	printf("Searching for a single pattern in a single string\nPrint at which position the pattern was found\n");
    printf("Input string: %s\n", input);
    printf("Pattern: %s\n", pattern);
	//Print Result[];
    for(int i = 0; i < match_times; i++)
       printf("Pos: %d Result: %d\n", i, result[i]);

	// Free device memory
	cudaFree(d_input);
    cudaFree(d_pattern);
    cudaFree(d_result);
    cudaFree(d_runtime);

	// Free host memory
	free(result);
	free(runtime);

	return 0;
}
