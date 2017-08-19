// CMPE297-6 HW2
// CUDA version Rabin-Karp


/* *
 HW2 by Jiongfeng Chen

Parallelize the sequential version of Rabin-Karp String matching algorithm
on GPU: Searching for multiple patterns in the input sequence

Test Case: 
 	input string:"Hello, 297 Class!"
 	pattern 1:"alxxl";
 	pattern 2:"llo";
 	pattern 3:", 297";
 	pattern 4:"97 Cl";
 Output:
	Kernel Execution Time: 5336 cycles
	Total cycles: 5336 
	Kernel Execution Time: 5336 cycles
	Searching for multiple patterns in the input sequence
	Input string: Hello, 297 Class!
	Pattern: "llo" was found.
	Pattern: ", 297" was found.
	Pattern: "97 Cl" was found. 
	
run:nvcc -I/usr/local/cuda/include -I. -lineinfo -arch=sm_53 --ptxas-options=-v -g   -c cmpe297_hw2_rabin_karp_multiPattern.cu -o cmpe297_hw2_rabin_karp_multiPattern.o

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
findIfExistsCu(char* input, int input_length, char* pattern, int* patternLength, int* patHashes, int* result, int *runtime)
{
	int start_time = clock64();
	
	int tid = threadIdx.x;
	int inputHash = 0;	
	int searchSpaceIndex[4];
	int patternIndex[4];
	for(int i = 0; i< 4; i++)
		result[i]=0;
	for(int i = 0; i< 4; i++)
		searchSpaceIndex[i]=0;
	for (int i = 0; i < 4; i++)
		if(i==0)
			searchSpaceIndex[i] = input_length - patternLength[i] +1 ;
		else
			searchSpaceIndex[i] = searchSpaceIndex[i-1] + input_length - patternLength[i] +1 ;
	//printf("C----tid-=%d--------%d-%d-%d-%d\n", tid,searchSpaceIndex[0],searchSpaceIndex[1] ,searchSpaceIndex[2] ,searchSpaceIndex[3]  );
	
	for(int i = 0; i < 4; i++)
    	if(i == 0)
    		patternIndex[i] = 0;
    	else
    		patternIndex[i] = patternIndex[i-1] + patternLength[i-1];

    int searchStart;		
	if(tid>=0 && tid <searchSpaceIndex[0])
	{
		searchStart	=tid-searchSpaceIndex[0] +patternIndex[0];
		for(int i = searchStart; i< searchStart +patternLength[0]; i++)
			inputHash = (inputHash*256 + input[i]) % 997;
		if(inputHash == patHashes[0] && 1)
			result[0]=1;	
			
	
	}
	
	if(tid>=searchSpaceIndex[0] && tid <searchSpaceIndex[1])
	{
		searchStart	=tid-searchSpaceIndex[1] +patternIndex[1];
		for(int i = searchStart; i< searchStart +patternLength[1]; i++)
			inputHash = (inputHash*256 + input[i]) % 997;
		if(inputHash == patHashes[1] && 1)
			result[1]=1;	
	}
	if(tid>=searchSpaceIndex[1] && tid <searchSpaceIndex[2])
	{
		searchStart	=tid-searchSpaceIndex[2] +patternIndex[2];
		for(int i = searchStart; i< searchStart +patternLength[2]; i++)
			inputHash = (inputHash*256 + input[i]) % 997;
		if(inputHash == patHashes[2] && memcpy(&(input[searchStart]),0, pattern+patternIndex[2], patternLength[2])==0)
			result[2]=1;	
	}
	if(tid>=searchSpaceIndex[2] && tid <searchSpaceIndex[3])
	{
		searchStart	=tid-searchSpaceIndex[3] +patternIndex[3];
		for(int i = searchStart; i< searchStart +patternLength[3]; i++)
			inputHash = (inputHash*256 + input[i]) % 997;
		if(inputHash == patHashes[3] && memcpy(&(input[searchStart]),0, pattern+patternIndex[3], patternLength[3])==0)
			result[3]=1;	
	}
		
	int stop_time = clock64();
	runtime[tid] = (int)(stop_time - start_time);
}

int main()
{

	// host variables
	//char input[] = "HEABAL"; 				/*Sample Input*/
	char input[] = "Hello, 297 Class!"; 	/*Multiple Patter Version: Input string*/
	char pattern[] = "AB"; 		/*Sample Pattern*/
	int patHash = 0; 			/*hash for the pattern*/
	int* result; 				/*Result array*/
	int* runtime; 				/*Exection cycles*/
	
	int input_length = 17; 		/*Input Length*/
	/*ADD CODE HERE*/;
	int pattern_number = 4; 									/*Multiple Patter Version: Pattern Number*/
	//int pattern_length = 2;		/*Pattern Length*/
	int match_times = 0; 		/*Match Times*/
 	cudaError_t err = cudaSuccess;
 	int * patternsLength;
 	int patHashes[4]; 			/*hash for the pattern*/
	int patternFound[4];		/*Multiple Patter Version: Result Array*/
 	char* patterns[4];			/*Multiple Patter Version: Pattern String*/
 	patterns[0] ="alxxl";
 	patterns[1] ="llo";
 	patterns[2] =", 297";
 	patterns[3] ="97 Cl";
 
	// device variables
	char* d_input;
	char* d_pattern;
	int* d_result;
	int* d_runtime;
	
	int* d_patHashes;
	int* d_patternsLength;
	char* d_patterns;
	
	//convert the string arrary to 1D string and pass to cuda fucntion
	//1D string	
	char patternLong[100] = "";  
	int* patternIndex;
	int p_size = 4*sizeof(int); 
	patternsLength = (int *) malloc(p_size);
	memset(patternsLength, 0, p_size);
	for(int i = 0; i < pattern_number; i++)
   		patternsLength[i] = strlen(patterns[i]);   		
    patternIndex = (int *) malloc(pattern_number*sizeof(int));
    for(int i = 0; i < pattern_number; i++)
    	if(i == 0)
    		patternIndex[i] = 0;
    	else
    		patternIndex[i] = patternIndex[i-1] + patternsLength[i-1];
   	for(int i = 0; i < pattern_number; i++)
    	strcat(patternLong, patterns[i]); 
    printf("Pattern: \"%s\" ...\n", patternLong);
     printf("Pattern: \"%s\"\n", patterns[0]);
               

   	for(int i = 0; i < pattern_number; i++)
   		printf("Pattern: \"%s\", lenth: %d.\n", patterns[i], patternsLength[i]);
	int totalPatternLength=0;
    for(int i = 0; i < pattern_number; i++)
    	totalPatternLength += patternsLength[i];    	
    	
	// measure the execution time by using clock() api in the kernel as we did in Lab3
	/*ADD CODE HERE*/;
	match_times=0;
	for (int i = 0; i < pattern_number; i++)
		match_times += strlen(input) - patternsLength[i] +1 ;
		
	int runtime_size = match_times*sizeof(int); 
    cudaMalloc((void**)&d_runtime, runtime_size);	
	runtime = (int *) malloc(runtime_size);
	memset(runtime, 0, runtime_size);
	          
	result = (int *) malloc((match_times)*sizeof(int));
    		
	/*Calculate the hash of the pattern*/
	for (int i = 0; i < pattern_number; i++)
	{
		int tmp=patternIndex[i];
		patHashes[i] =0;
		for (int j = 0; j < patternsLength[i]; j++)
		{			
			//if(i==3) printf("xxx %c \n", patternLong[tmp+j]);
		    patHashes[i] = (patHashes[i] * 256 + patternLong[tmp+j]) % 997;
		}
		printf("patHash %d \n", patHashes[i]);
	}
		
    	
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
    
    size = totalPatternLength*sizeof(char);
	err = cudaMalloc((void**)&d_pattern, size);
	printf("Copy pattern string from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_pattern, patternLong, size, cudaMemcpyHostToDevice);
    
    size = pattern_number*sizeof(int);
	err = cudaMalloc((void**)&d_patHashes, size);
	printf("Copy Hashes from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_patHashes, patHashes, size, cudaMemcpyHostToDevice);
    
    size = pattern_number*sizeof(int);
	err = cudaMalloc((void**)&d_patternsLength, size);
	printf("Copy patternsLength from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_patternsLength, patternsLength, size, cudaMemcpyHostToDevice);
if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device d_input(error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	size = pattern_number*sizeof(int);
	err = cudaMalloc((void**)&d_result, size);
	
	/*ADD CODE HERE: Launch the kernel and pass the arguments*/
	int blocksPerGrid = 1;// FILL HERE
    int threadsPerBlock = match_times;// FILL HERE
    //printf("C--------=%d--------%s\n",match_times, "");
	findIfExistsCu<<<blocksPerGrid, threadsPerBlock>>>(d_input, input_length, d_pattern, d_patternsLength, d_patHashes, d_result,d_runtime);

		
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
    for(int i = 0; i < match_times; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];

	//Print
    printf("Kernel Execution Time: %llu cycles\n", elapsed_time);
	printf("Total cycles: %d \n", elapsed_time);
	printf("Kernel Execution Time: %d cycles\n", elapsed_time);
    printf("Searching for multiple patterns in the input sequence\n");
    printf("Input string: %s\n", input);
	//Print Result[];
    for(int i = 0; i < pattern_number; i++)
    	if(result[i]==1)
    		printf("Pattern: \"%s\" was found.\n", patterns[i]);

	// Free device memory
	cudaFree(d_input);
    cudaFree(d_pattern);
    cudaFree(d_result);
    cudaFree(d_runtime);
    cudaFree(d_patHashes);
    cudaFree(d_patternsLength);
    cudaFree(d_patterns);
	
	// Free host memory
	free(result);
	free(runtime);

	return 0;
}
