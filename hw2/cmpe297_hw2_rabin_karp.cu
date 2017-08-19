// CMPE297-6 HW2
// CUDA version Rabin-Karp

#include<stdio.h>
#include<iostream>


/*ADD CODE HERE: Implement the parallel version of the sequential Rabin-Karp*/
__global__ void 
findIfExistsCu(char* input, int input_length, char* pattern, int pattern_length, int patHash, int* result)
{ 
	
	
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

	// device variables
	char* d_input;
	char* d_pattern;
	int* d_result;
	int* d_runtime;

	// measure the execution time by using clock() api in the kernel as we did in Lab3
	int runtime_size = /*FILL CODE HERE*/;

	result = (int *) malloc((input_length-pattern_length)*sizeof(int));
	runtime = (int *) malloc(runtime_size);
	
	/*Calculate the hash of the pattern*/
	for (int i = 0; i < M; i++)
    {
        patHash = (patHash * 256 + pattern[i]) % 997;
    }

	/*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/

	
	/*ADD CODE HERE: Launch the kernel and pass the arguments*/
		
		
	/*ADD CODE HERE: Copy the execution times from the GPU memory to HOST Code*/		
	
	
	/*RUN TIME calculation*/
    unsigned long long elapsed_time = 0;
    for(int i = 0; i < input_length-pattern_length; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];

    printf("Kernel Execution Time: %llu cycles\n", elapsed_time);
	printf("Total cycles: %d \n", elapsed_time);
	printf("Kernel Execution Time: %d cycles\n", elapsed_time);

	
	/*ADD CODE HERE: COPY the result and print the result as in the HW description*/
	
	return 0;
}

