// Simple SUDOKU probram in C
// cmpe297_hw3_easysudoku.cpp

#include<stdio.h>
#include<string.h>
#include <windows.h>

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

#define IS_OPTION(row, col, k) \
			((context.not_in_row[row][k] == 1) && \
			(context.not_in_col[col][k] == 1) && \
			(context.not_in_cell[row/3+(col/3)*3][k] == 1))? 1 : 0;

#define FINISHED()	(memcmp(context.num_options, big_2x, sizeof(big_2x)) == 0? 1: 0)

// rule: numbers should be unique in each sub-array, each row, and each column
void c_Sudoku()
{
	// Execution finishes when all the entries in the matrix have 1 value
	while(!FINISHED())
	{
		for(int row = 0; row < 9; row++)
		{
			for(int col = 0; col < 9; col++)
			{
				if(context.num_options[row][col] > 1)
				{
					// Find values that are not in the row, col, and the 
					// 3x3 cell that (row, col) is belonged to.			
					int value = 0, temp;
					context.num_options[row][col] = 0;

					for(int k = 0; k < 9; k++)
					{
						temp = IS_OPTION(row, col, k);
						if(temp == 1)
						{
							context.num_options[row][col]++;
							value = k;

						}
					}

					// If the above loop found only one value, 
					// set the value to (row, col)
					if(context.num_options[row][col] == 1)
					{
						context.not_in_row[row][value] = 0;
						context.not_in_col[col][value] = 0;
						context.not_in_cell[(row)/3+((col)/3)*3][value] = 0;
						context.val[row][col] = value+1;
					}
				}
				
			}
		}
	}
}

int main(int argc, char **argv)
{
	initialize_all();
	print_all();

	c_Sudoku();

	print_all();

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
	int val;

	for(i=0; i<9; i++){
	   for(j=0; j<9; j++){
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

