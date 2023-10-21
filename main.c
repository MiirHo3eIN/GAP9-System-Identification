#define MODEL_ORDER 56 //Set AR/ARMA model order, multiple of 8
#define NSMP 1960

#define MAX_ROWS 4096
#define MAX_COLUMNS 6
#define NUM_CORES 8

#include "pmsis.h"
#include <stdio.h>
#include <math.h>
#include "test.inc" // load data input[MAX_COLUMNS][MAX_ROWS]
#include <stdlib.h>
#include "Gap.h"
#include "DSP_Lib.h"
#include "TwiddlesDef.h"
#include "measurments_utils.h"
#include "time.h"
#include <unistd.h>

#define PAD_GPIO    (PI_PAD_086) // Needed for the power measurement

#define NCHUNK 8 //Set maximum number of chunks (equal to the number of partitions in the 1st step) -- fixed in all our considerations
#define NO (NSMP-MODEL_ORDER) // Has to be divisible by NCHUNK (8) -- Number of rows of the regression matrix of the AR model

#define NO2 (NO-2*MODEL_ORDER) // Identical to NO, rows of the regression matrix of the ARMA model

//PI_L2 float input[MAX_COLUMNS][MAX_ROWS];

PI_L2 float theta[MODEL_ORDER];
PI_L2 float sigma = 0;

uint32_t g_lfsr_state = 0x356167u; // Needed only for the random

struct pi_device uart;

PI_L2 float u[NO+MODEL_ORDER]; // Error vector, needed to build the ARMA regression matrix
PI_L2 float sigma; // Noise variance

PI_L2 float n;
PI_L2 int i2;;
PI_L2 static float buffer[NUM_CORES];
PI_L2 static int idx[NUM_CORES];
PI_L2 float rk=0.0;
PI_L2 static float temp[NUM_CORES];
PI_L2 static float one=1.0;

PI_L2 float Qi_0[NO/NCHUNK][NO/NCHUNK], Ri_0[NO/NCHUNK][MODEL_ORDER];
PI_L2 float Qi_next[2*MODEL_ORDER][2*MODEL_ORDER], Ri_next[2*MODEL_ORDER][MODEL_ORDER];
PI_L2 float Yi[NCHUNK*MODEL_ORDER], R[NCHUNK*MODEL_ORDER][MODEL_ORDER];
PI_L2 float phi_i_0_transposed[MODEL_ORDER][NO/NCHUNK];
PI_L2 float phi_i_next_transposed[MODEL_ORDER][2*MODEL_ORDER];
PI_L2 float phiA[NO][MODEL_ORDER];
PI_L2 float outi_next[NCHUNK*MODEL_ORDER];
//outi_temp[NO/NCHUNK];

PI_L2 float Qi_0_2[NO2/NCHUNK][NO2/NCHUNK], Ri_0_2[NO2/NCHUNK][MODEL_ORDER];
PI_L2 float phi_i_0_transposed_2[MODEL_ORDER][NO2/NCHUNK];

void qr_gramSmidt_next(float Q[2*MODEL_ORDER][2*MODEL_ORDER], float R[2*MODEL_ORDER][MODEL_ORDER], float input[MODEL_ORDER][2*MODEL_ORDER]); // Both AR and MA
void qr_gramSmidt_0(float Q[NO/NCHUNK][NO/NCHUNK], float R[NO/NCHUNK][MODEL_ORDER], float input[MODEL_ORDER][NO/NCHUNK]); // AR model
void qr_gramSmidt_0_2(float Q[NO2/NCHUNK][NO2/NCHUNK], float R[NO2/NCHUNK][MODEL_ORDER], float input[MODEL_ORDER][NO2/NCHUNK]); // ARMA model (ARX)
//float norm(float *v, int row, int column);
static uint32_t lfsr_step(uint32_t reg); // rand
static uint32_t rand(); // test
float** task_CreateMatrix(int rows, int columns); // not used
void task_FreeMatrix(float** matrix, int rows, int columns); // not used
inline float Sqrt(float x);

#pragma GCC push_options
#pragma GCC optimize ("-O3")
__attribute__ ((noinline))
float norm(float *v, int row, int column){ 
	i2=0;
	n = 0.0f;
	int j;
	#if NUM_CORES > 1

	int blockSize_column = column/NUM_CORES;
	int start_column = pi_core_id()*blockSize_column;
	int start_matrice = pi_core_id()*blockSize_column;

	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_column = column - (NUM_CORES - 1)* blockSize_column;
	}
	
	buffer[pi_core_id()]=0;
	idx[pi_core_id()]=0;
	
	for(j = start_column; (j<column) && (j<start_column + blockSize_column); j++){
		buffer[pi_core_id()] = buffer[pi_core_id()] + v[idx[pi_core_id()]+start_matrice]*v[idx[pi_core_id()]+start_matrice];
		idx[pi_core_id()]+=1;
	}
		
	pi_cl_team_barrier();
			
	if(pi_core_id()==0)
		for(j=0; j<NUM_CORES; j++){
			n += buffer[j];
		}
	pi_cl_team_barrier();
	#else

	for(j=0; (j+1)<column; j+=2){
		float t0 = v[i2];float t1 = v[i2+1];
		float temp = t0*t0;
		temp = temp + t1*t1;
		n = n + temp;
		i2+=2*1;
	}
		
	if(j<column){
		float t0 = v[i2];
		float temp = t0*t0;
		n = n + temp;
		i2+=1*1;
	}

	#endif
	return sqrt(n);
	
}

void cluster_main();

void pe_entry(void *arg){
	cluster_main();
}

void cluster_entry(void *arg){
	pi_cl_team_fork((NUM_CORES), pe_entry, 0);
}

static int test_entry(){

	pi_perf_conf(1 << PI_PERF_CYCLES);


    pi_pad_function_set(PAD_GPIO, PI_PAD_FUNC1); // functions for the Nordic

    pi_gpio_flags_e flags = PI_GPIO_OUTPUT;
    pi_gpio_pin_configure(PAD_GPIO, flags);

//	RANDOM testing
//	for(int h=0; h<MAX_ROWS; h++){
//		for(int l=0; l<MAX_COLUMNS; l++)
//			input[l][h] = (float) rand();
//	}
/*	float sum=0;
	for(int i=0; i<MAX_ROWS; i++){
		sum += input[0][i];
	}
	sum = sum/MAX_ROWS;
	for(int i=0; i<MAX_ROWS; i++){
		input[0][i]=input[0][i]-sum;
	}
	sum = 0;
*/
	struct pi_device cluster_dev;
	struct pi_cluster_conf cl_conf;
	struct pi_cluster_task cl_task;

	pi_cluster_conf_init(&cl_conf);
	pi_open_from_conf(&cluster_dev, &cl_conf);
	if (pi_cluster_open(&cluster_dev)){
		return -1;
	}

	pi_perf_start(); // Performance profiling
	pi_cluster_send_task_to_cl(&cluster_dev, pi_cluster_task(&cl_task, cluster_entry, NULL));
	pi_perf_stop(); // Performance profiling

	pi_cluster_close(&cluster_dev);

	uint32_t tim_cycles = pi_perf_read(PI_PERF_CYCLES); // Performance profiling
/*
	printf("%d\n", MODEL_ORDER);
	for(uint32_t i=0; i<MODEL_ORDER; i++)
		printf("%f\n", theta[i]);
	printf("%f\n", sigma);
	printf("%d\n", tim_cycles);
*/
	return 0;
}


void cluster_main()
{
	if(pi_core_id()==0)
	        pi_gpio_pin_write(PAD_GPIO, 1);  // For the Power Measurement with Nordic
	pi_cl_team_barrier();
/*	if(pi_core_id()==0)
		printf("Start of the computation\n");
	pi_cl_team_barrier();*/
	int i, j, w;
	int iIter, iCh, Nsmp_block, end_loop=NCHUNK;

	#if NUM_CORES > 1

	// Block for the parallelized version

/*	int blockSize_NO = NO/NUM_CORES;
	int start_NO = pi_core_id()*blockSize_NO;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NO = NO - (NUM_CORES - 1)* blockSize_NO;}
*/
	int blockSize_NO_NC = NO/(NUM_CORES*NCHUNK);
	int start_NO_NC = pi_core_id()*blockSize_NO_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NO_NC = (NO/NCHUNK) - (NUM_CORES - 1)* blockSize_NO_NC;}

	int blockSize_NO2_NC = NO2/(NUM_CORES*NCHUNK);
	int start_NO2_NC = pi_core_id()*blockSize_NO2_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NO2_NC = (NO2/NCHUNK) - (NUM_CORES - 1)* blockSize_NO2_NC;}

	int blockSize_NO2 = NO2/NUM_CORES;
	int start_NO2 = pi_core_id()*blockSize_NO2;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NO2 = NO2 - (NUM_CORES - 1)* blockSize_NO2;}

	#endif

	// Start looking at the binary tree

	for(iIter = 1; iIter <= log2(NCHUNK)+1; iIter++){

		if(iIter != 1) // Prepare regression matrix Phi (the integer one to be decomposed)

		// NOTE: The if(iIter==1) wasn't needed, Nsmp_block variable is useful only or the other iterations

		{
			//Ni = (int) floor(Ni/2); //general
			if(iIter == 2)
				Nsmp_block = end_loop*MODEL_ORDER*2 / NCHUNK; // is a const after the first iteration with our bonds on the variables
			// NOTE: I added the if(iIter==2) becuase of the observation done in the line above
			// NOTE: I eliminated Ni as a variable cause it wasn't sintattically needed

			#if NUM_CORES > 1
			if(pi_core_id()==0){
				for(i = 0; i < end_loop*MODEL_ORDER; i++)
					outi_next[i] = Yi[i]; // Saves Y in an array because it'll be overwritten
			}
			pi_cl_team_barrier();
			#else
			for(i = 0; i < end_loop*MODEL_ORDER; i++)
				outi_next[i] = Yi[i];
			#endif
			//Update R and Y values at each iteration
		}

		//Start processing considering the available chunks as a power of 2

		// end_loop = (int) floor(pow(2, log2(NCHUNK)-iIter+1)); // General expression

		if(iIter!=1)
			end_loop = end_loop/2; // epression considering the constraints decided (NCHUNK was always fixed as 8)
		// end_loop is needed to know how many QR are needed at that particular step (basically at every step is divided by 2)
		// NOTE: I changed this structure to avoid the complex expression, it should work fine

		for(iCh = 1; iCh <= end_loop; iCh++){

			if(iIter == 1){

				#if NUM_CORES > 1
				for(j=0; j<MODEL_ORDER;j++){
					for(i=start_NO_NC; i<start_NO_NC + blockSize_NO_NC; i++)
						phi_i_0_transposed[j][i] = -input[0][(iCh-1)*NO/NCHUNK+i-j+MODEL_ORDER-1];
				// phi_i_0 is a chunk of the regression matrix, this step builds a chunk of the regression matrix at every iteration starting directly from the raw data (input)
				}
				pi_cl_team_barrier();

				#else
				for(j = 0; j < MODEL_ORDER; j++){
					for(i = 0; i < NO/NCHUNK; i++)
						phi_i_0_transposed[j][i] = -input[0][(iCh-1)*NO/NCHUNK+i-j+MODEL_ORDER-1];
				}
				#endif

				qr_gramSmidt_0(Qi_0, Ri_0, phi_i_0_transposed);

				#if NUM_CORES > 1
				if(pi_core_id()==0){
					for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
						for(j=0; j < MODEL_ORDER; j++)
							R[(iCh-1)*MODEL_ORDER+i][j] = Ri_0[i][j];
					}

					for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
						Yi[(iCh-1)*MODEL_ORDER+i] = 0;
						for(w=0; w<NO/NCHUNK; w++)
							Yi[(iCh-1)*MODEL_ORDER+i] += Qi_0[i][w] * input[0][(iCh-1)*NO/NCHUNK+w+MODEL_ORDER];
					}
				}

				pi_cl_team_barrier();

				#else
				for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
					for(j=0; j < MODEL_ORDER; j++)
						R[(iCh-1)*MODEL_ORDER+i][j] = Ri_0[i][j];
				}

				for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
					Yi[(iCh-1)*MODEL_ORDER+i] = 0;
					for(w=0; w<NO/NCHUNK; w++)
						Yi[(iCh-1)*MODEL_ORDER+i] += Qi_0[i][w] * input[0][(iCh-1)*NO/NCHUNK+w+MODEL_ORDER];
				}
				#endif
			}

			else{
				#if NUM_CORES > 1
				if(pi_core_id()==0){
					for(i=0; i < Nsmp_block; i++){
						for(j=0; j < MODEL_ORDER; j++)
							phi_i_next_transposed[j][i] = R[(iCh-1)*Nsmp_block+i][j];
				// Chunks the matrix to be decomposed (in the steps after the first one the concatenation of the Rs of the previous iterations is the input of the next PTSQR
					}
				}
				pi_cl_team_barrier();
				#else
				for(i=0; i < Nsmp_block; i++){
					for(j=0; j < MODEL_ORDER; j++)
						phi_i_next_transposed[j][i] = R[(iCh-1)*Nsmp_block+i][j];
				}
				#endif

				qr_gramSmidt_next(Qi_next, Ri_next, phi_i_next_transposed);

				#if NUM_CORES > 1
				if(pi_core_id()==0){
					for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
						for(j=0; j < MODEL_ORDER; j++)
							R[(iCh-1)*MODEL_ORDER+i][j] = Ri_next[i][j];
					}

					for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
						Yi[(iCh-1)*MODEL_ORDER+i] = 0;
						for(w=0; w < 2*MODEL_ORDER; w++)
							Yi[(iCh-1)*MODEL_ORDER+i] += Qi_next[i][w]*outi_next[(iCh-1)*Nsmp_block+w];
					}
				}
				pi_cl_team_barrier();
				#else
				for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
					for(j=0; j < MODEL_ORDER; j++)
						R[(iCh-1)*MODEL_ORDER+i][j] = Ri_next[i][j];
				}

				for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
					Yi[(iCh-1)*MODEL_ORDER+i] = 0;
					for(w=0; w < 2*MODEL_ORDER; w++)
						Yi[(iCh-1)*MODEL_ORDER+i] += Qi_next[i][w]*outi_next[(iCh-1)*Nsmp_block+w];
				}
				#endif
			}
		}
	}


	// AR PARAMETER ESTIMATE

	if(pi_core_id()==0){
		float sum = 0;
		for(int k=MODEL_ORDER-1; k>-1; k--){
			if(k==MODEL_ORDER-1)
				theta[k] = Yi[k]/R[k][k];
			else{
				sum = 0;
				for(i=k+1; i<MODEL_ORDER;i++)
					sum+=theta[i]*R[k][i];
				theta[k]=(Yi[k]-sum)/R[k][k];
			}
		}

	// NOISE VARIANCE ESTIMATE

		for(j=MODEL_ORDER; j<NO+MODEL_ORDER; j++){
			float sum_theta_y = 0;
			for(i=0; i<MODEL_ORDER; i++)
				sum_theta_y += theta[i] * (-input[0][j-i-1]);
			u[j] = input[0][j] - sum_theta_y;
			sigma += u[j] * u[j];
		}
		sigma /= (NO-MODEL_ORDER-1);
	}
	pi_cl_team_barrier();
	if(pi_core_id()==0)
        	pi_gpio_pin_write(PAD_GPIO, 0);
	pi_cl_team_barrier();


	// Build of the regression matrix of the ARMA model (Input + Error vector)
	// Differently from the AR part, here we need to build a regression matrix completely, and then chunk it
	// (Maybe it's possible also to directly do it as done in the AR part, but it would be syntactically really complex)

	// From now on basically the considerations are the same

	#if NUM_CORES > 1
	for(j=0;j<MODEL_ORDER/2;j++){
		for(i=start_NO2;i<start_NO2+blockSize_NO2;i++){
			phiA[i][j] = -input[0][i+MODEL_ORDER/2-j+MODEL_ORDER-1];
			phiA[i][j+MODEL_ORDER/2] =  u[i+MODEL_ORDER/2-j+MODEL_ORDER-1];
		}
	}
	pi_cl_team_barrier();
	#else
	for(i=MODEL_ORDER/2;i<NO2+MODEL_ORDER/2;i++){
		for(j=0;j<MODEL_ORDER/2;j++){
			phiA[i-MODEL_ORDER/2][j] = -input[0][i-j+MODEL_ORDER-1];
			phiA[i-MODEL_ORDER/2][j+MODEL_ORDER/2] =  u[i-j+MODEL_ORDER-1];
		}
	}
	#endif

	// Second TSQR

	// Start looking at the binary tree
	end_loop = NCHUNK;
	// NOTE: added considering the modification above

	for(iIter = 1; iIter <= log2(NCHUNK)+1; iIter++){

		if(iIter != 1)
		// Same observation as above in the AR section
		{
			//Ni = (int) floor(Ni/2); //general
			if(iIter==2)
				Nsmp_block = end_loop*MODEL_ORDER*2 / NCHUNK; // is a const after the first iteration with our bonds on the variables
			// Same Notes of the AR section

			#if NUM_CORES > 1
			if(pi_core_id()==0){
				for(i = 0; i < end_loop*MODEL_ORDER; i++)
					outi_next[i] = Yi[i];
			}
			pi_cl_team_barrier();
			#else
			for(i = 0; i < end_loop*MODEL_ORDER; i++)
				outi_next[i] = Yi[i];
			#endif
			//Update R and Y values at each iteration
		}

		//Start processing considering the available chunks as a power of 2

		if(iIter != 1) // Same Notes as above
			end_loop = end_loop/2;

		for(iCh = 1; iCh <= end_loop; iCh++){

			if(iIter == 1){
				#if NUM_CORES > 1
				for(j=0; j<MODEL_ORDER;j++){
					for(i=start_NO2_NC; i<start_NO2_NC + blockSize_NO2_NC; i++)
						phi_i_0_transposed_2[j][i] = phiA[(iCh-1)*NO2/NCHUNK+i][j];
				}
				pi_cl_team_barrier();

				#else
				for(j = 0; j < MODEL_ORDER; j++){
					for(i = 0; i < NO2/NCHUNK; i++)
						phi_i_0_transposed_2[j][i] = phiA[((iCh-1)*NO2/NCHUNK)+i][j];
				}
				#endif
				qr_gramSmidt_0_2(Qi_0_2, Ri_0_2, phi_i_0_transposed_2);

				pi_cl_team_barrier();

				#if NUM_CORES > 1
				if(pi_core_id()==0){
					for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
						for(j=0; j < MODEL_ORDER; j++)
							R[(iCh-1)*MODEL_ORDER+i][j] = Ri_0_2[i][j];
					}
					for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
						Yi[(iCh-1)*MODEL_ORDER+i] = 0;
						for(w=0; w<NO2/NCHUNK; w++)
							Yi[(iCh-1)*MODEL_ORDER+i] += Qi_0_2[i][w]* input[0][(iCh-1)*NO2/NCHUNK+w+3*MODEL_ORDER/2];
					}
				}

				pi_cl_team_barrier();

				#else
				for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
					for(j=0; j < MODEL_ORDER; j++)
						R[(iCh-1)*MODEL_ORDER+i][j] = Ri_0_2[i][j];
				}
				for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
					Yi[(iCh-1)*MODEL_ORDER+i] = 0;
					for(w=0; w<NO2/NCHUNK; w++)
						Yi[(iCh-1)*MODEL_ORDER+i] += Qi_0_2[i][w]* input[0][(iCh-1)*NO2/NCHUNK+w+3*MODEL_ORDER/2];
				}
				#endif
			}

			else{
				#if NUM_CORES > 1
				if(pi_core_id()==0){
					for(i=0; i < Nsmp_block; i++){
						for(j=0; j < MODEL_ORDER; j++)
							phi_i_next_transposed[j][i] = R[(iCh-1)*Nsmp_block+i][j];
					}
				}
				pi_cl_team_barrier();
				#else
				for(i=0; i < Nsmp_block; i++){
					for(j=0; j < MODEL_ORDER; j++)
						phi_i_next_transposed[j][i] = R[(iCh-1)*Nsmp_block+i][j];
				}
				#endif

				qr_gramSmidt_next(Qi_next, Ri_next, phi_i_next_transposed);

				pi_cl_team_barrier();
				#if NUM_CORES > 1
				if(pi_core_id()==0){
					for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
						for(j=0; j < MODEL_ORDER; j++)
							R[(iCh-1)*MODEL_ORDER+i][j] = Ri_next[i][j];
					}

					for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
						Yi[(iCh-1)*MODEL_ORDER+i] = 0;
						for(w=0; w < 2*MODEL_ORDER; w++)
							Yi[(iCh-1)*MODEL_ORDER+i] += Qi_next[i][w]*outi_next[(iCh-1)*Nsmp_block+w];
					}
				}
				pi_cl_team_barrier();
				#else
				for(i=0; i < MODEL_ORDER; i++){ // Concatenate new R matrix
					for(j=0; j < MODEL_ORDER; j++)
						R[(iCh-1)*MODEL_ORDER+i][j] = Ri_next[i][j];
				}

				for(i=0; i < MODEL_ORDER; i++){ // Update vector coefficient
					Yi[(iCh-1)*MODEL_ORDER+i] = 0;
					for(w=0; w < 2*MODEL_ORDER; w++)
						Yi[(iCh-1)*MODEL_ORDER+i] += Qi_next[i][w]*outi_next[(iCh-1)*Nsmp_block+w];
				}
				#endif
			}
		}
	}

	if(pi_core_id()==0){
		float sum = 0;
		for(int k=MODEL_ORDER-1; k>-1; k--){
			if(k==MODEL_ORDER-1)
				theta[k] = Yi[k]/R[k][k];
			else{
				sum = 0;
				for(i=k+1; i<MODEL_ORDER;i++)
					sum+=theta[i]*R[k][i];
				theta[k]=(Yi[k]-sum)/R[k][k];
			}
		}

	// NOISE VARIANCE ESTIMATE

		sigma = 0;
		for(j=0; j<NO2; j++){
			float sum_theta_y = 0;
			for(i=0;i<MODEL_ORDER;i++)
				sum_theta_y += theta[i] * phiA[j][i];
			u[j] = input[0][j+3*MODEL_ORDER/2] - sum_theta_y;
			sigma += u[j] * u[j];
		}
		sigma /= (NO2-3*MODEL_ORDER/2);
	}
	pi_cl_team_barrier();

	if(pi_core_id()==0) // For the power analysis with Nordic
        	pi_gpio_pin_write(PAD_GPIO, 0);
	pi_cl_team_barrier();

//	if(pi_core_id()==0)
//		printf("End of the computation\n");
//	pi_cl_team_barrier();

}

static void test_kickoff(void *arg)
{
    int ret = test_entry();
    pmsis_exit(ret);
}

int main()
{
	return pmsis_kickoff((void *)test_kickoff);
}

static uint32_t lfsr_step(uint32_t reg)
{
reg = ((((reg >> 31) ^ (reg >> 6) ^ (reg >> 4) ^ (reg >> 2) ^ (reg >> 1) ^ reg) & 0x0000001) <<31) | (reg >> 1);
return reg;
}

static uint32_t rand()
{
	g_lfsr_state = lfsr_step(g_lfsr_state);
	return g_lfsr_state;
}

float** task_CreateMatrix(int rows, int columns){
    int i;
    float** matrix;

    matrix = (float **) pi_l2_malloc(rows * sizeof(float *));
    for (i = 0; i < rows; i++) {
        matrix[i] = (float *) pi_l2_malloc(columns * sizeof(float));
	}
	return matrix;
}

void task_FreeMatrix(float** matrix, int rows, int columns){
	int i;
	for (i=0; i<rows; i++) {
        pi_l2_free(matrix[i], columns*sizeof(float));
    }
    pi_l2_free(matrix, rows*sizeof(float*));
}
/*
inline float Sqrt(float x) {
	float res;
	asm("fsqrt.s %0, %1":"=f"(res):"f"(x));
	return res;
}

float norm(float *v, int row, int column){
	i2=0;
	n = 0.0f;
	int j;
	#if NUM_CORES > 1

	int blockSize_column = column/NUM_CORES;
	int start_column = pi_core_id()*blockSize_column;
	int start_matrice = pi_core_id()*blockSize_column;

	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_column = column - (NUM_CORES - 1)* blockSize_column;
	}

	buffer[pi_core_id()]=0;
	idx[pi_core_id()]=0;

	for(j = start_column; (j<column) && (j<start_column + blockSize_column); j++){
		buffer[pi_core_id()] = buffer[pi_core_id()] + v[idx[pi_core_id()]+start_matrice]*v[idx[pi_core_id()]+start_matrice];
		idx[pi_core_id()]+=1;
	}

	pi_cl_team_barrier();

	if(pi_core_id()==0)
		for(j=0; j<NUM_CORES; j++){
			n += buffer[j];
		}
	pi_cl_team_barrier();
	#else

	for(j=0; (j+1)<column; j+=2){
		float t0 = v[i2];float t1 = v[i2+1];
		float temp = t0*t0;
		temp = temp + t1*t1;
		n = n + temp;
		i2+=2*1;
	}

	if(j<column){
		float t0 = v[i2];
		float temp = t0*t0;
		n = n + temp;
		i2+=1*1;
	}

	#endif
	return sqrt(n);

}

void qr_gramSmidt_0(float Q[NO/NCHUNK][NO/NCHUNK], float R[NO/NCHUNK][MODEL_ORDER], float input[MODEL_ORDER][NO/NCHUNK]){
	int j;
	int N_CHANNELS=NO/NCHUNK;
	int EV_WINDOWS_SIZE=MODEL_ORDER;

	#if NUM_CORES > 1
	int blockSize_NC = N_CHANNELS/NUM_CORES;
	int start_NC = pi_core_id()*blockSize_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NC = N_CHANNELS - (NUM_CORES - 1)* blockSize_NC;}
	#endif

	#if NUM_CORES > 1
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=start_NC; j<start_NC+blockSize_NC; j++)
			Q[j][i]=0;
	}


	for(int j=0; j<MODEL_ORDER; j++){
		for(int i=start_NC; i<start_NC+blockSize_NC; i++)
			R[i][j]=0;
	}

	pi_cl_team_barrier();
	#else
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<NO/NCHUNK; j++){
			Q[j][i]=0;
		}
	}

	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<MODEL_ORDER; j++){
			R[i][j]=0;
		}
	}
	#endif

	for(int k=0; k<EV_WINDOWS_SIZE; k++){

		#if NUM_CORES > 1
		for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<start_NC + blockSize_NC){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#else
		for(j=0; (j+1)<N_CHANNELS; j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<N_CHANNELS){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#endif

		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		for(int i=0; i<k; i++){
			#if NUM_CORES > 1
			temp[pi_core_id()]=0;
			for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				float ji1 = Q[i][j+1];float jk1 = Q[k][j+1];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0) + (ji1 * jk1);
			}

			if((j<start_NC + blockSize_NC)){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0);
			}

			pi_cl_team_barrier();

			if(pi_core_id()==0)
				for(j=0; j<NUM_CORES; j++){
			    		R[i][k]+=temp[j];
			    	}
			pi_cl_team_barrier();

			#else

			for(j=0; (j+1)<N_CHANNELS; j+=2){
				float Qji0 = Q[i][j];
				float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];
				float Qjk1 = Q[k][j+1];
				float Rik  = R[i][k];
				R[i][k] = Rik + (Qji0 * Qjk0) + (Qji1 * Qjk1);
            }
            if(j<N_CHANNELS){
				float Qji0 = Q[i][j]; float Qjk0 = Q[k][j];
				float Rik  = R[i][k];
				float temp = (Qji0 * Qjk0);
				R[i][k] = Rik + temp;
            }
			#endif

			#if NUM_CORES > 1
			for(j = start_NC; (j+1<start_NC + blockSize_NC); j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				//hal_compiler_barrier();

				Q[k][j] = Qjk0 - Rik*Qji0;
                Q[k][j+1] = Qjk1 - Rik*Qji1;
            }
			if(j<start_NC + blockSize_NC){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#else
			for(j=0; j+1<N_CHANNELS; j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				Q[k][j] = Qjk0 - Rik*Qji0;
				Q[k][j+1] = Qjk1 - Rik*Qji1;
			}
			if(j<N_CHANNELS){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#endif

			#if NUM_CORES > 1
			pi_cl_team_barrier();
			#endif
		}
		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		R[k][k] = norm(&Q[k][0],N_CHANNELS,N_CHANNELS);
		rk = one/R[k][k];

		#if NUM_CORES > 1
		for( j = start_NC; (j+1)<start_NC + (blockSize_NC & 0xFFFE); j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;

		} if (j<start_NC + (blockSize_NC & 0xFFFE)){
			float Qjk0 = Q[j][k];
			Q[k][j] = Qjk0 * rk;
		}
		#else
		for( j=0; (j+1)<N_CHANNELS; j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
		} if (j<N_CHANNELS){
			float Qjk0 = Q[k][j];
			Q[k][j] = Qjk0 * rk;
		}
		#endif

		#if NUM_CORES > 1
		if(blockSize_NC & 0x0001){
			Q[k][j] = Q[k][j]*rk;}
		pi_cl_team_barrier();
		#endif

	}
	return;
}

void qr_gramSmidt_next(float Q[2*MODEL_ORDER][2*MODEL_ORDER], float R[2*MODEL_ORDER][MODEL_ORDER], float input[MODEL_ORDER][2*MODEL_ORDER]){
	int j;
	int N_CHANNELS=2*MODEL_ORDER;
	int EV_WINDOWS_SIZE=MODEL_ORDER;

	#if NUM_CORES > 1
	if(pi_core_id()==0){
	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<2*MODEL_ORDER; j++)
			Q[j][i]=0;
	}


	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<MODEL_ORDER; j++)
			R[i][j]=0;
	}
	}
	pi_cl_team_barrier();
	#else
	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<2*MODEL_ORDER; j++){
			Q[j][i]=0;
		}
	}

	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<MODEL_ORDER; j++){
			R[i][j]=0;
		}
	}
	#endif

	#if NUM_CORES > 1
	int blockSize_NC = N_CHANNELS/NUM_CORES;
	int start_NC = pi_core_id()*blockSize_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NC = N_CHANNELS - (NUM_CORES - 1)* blockSize_NC;}
	#endif

	for(int k=0; k<EV_WINDOWS_SIZE; k++){
		#if NUM_CORES > 1
		for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<start_NC + blockSize_NC){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#else
		for(j=0; (j+1)<N_CHANNELS; j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<N_CHANNELS){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#endif

		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		for(int i=0; i<k; i++){
			#if NUM_CORES > 1
			temp[pi_core_id()]=0;
			for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				float ji1 = Q[i][j+1];float jk1 = Q[k][j+1];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0) + (ji1 * jk1);
			}
			if((j<start_NC + blockSize_NC)){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0);
			}

			pi_cl_team_barrier();

			if(pi_core_id()==0)
				for(j=0; j<NUM_CORES; j++){
			    		R[i][k]+=temp[j];
			    	}
			pi_cl_team_barrier();

			#else

			for(j=0; (j+1)<N_CHANNELS; j+=2){
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];
				float Rik  = R[i][k];
				R[i][k] = Rik + (Qji0 * Qjk0) + (Qji1 * Qjk1);
            }
            if(j<N_CHANNELS){
				float Qji0 = Q[i][j]; float Qjk0 = Q[k][j];
				float Rik  = R[i][k];
				float temp = (Qji0 * Qjk0);
				R[i][k] = Rik + temp;
            }
			#endif

			#if NUM_CORES > 1
			for(j = start_NC; (j+1<start_NC + blockSize_NC); j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				//hal_compiler_barrier();

				Q[k][j] = Qjk0 - Rik*Qji0;
                Q[k][j+1] = Qjk1 - Rik*Qji1;
            }
			if(j<start_NC + blockSize_NC){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#else
			for(j=0; j+1<N_CHANNELS; j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				Q[k][j] = Qjk0 - Rik*Qji0;
				Q[k][j+1] = Qjk1 - Rik*Qji1;
			}
			if(j<N_CHANNELS){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#endif

			#if NUM_CORES > 1
			pi_cl_team_barrier();
			#endif
		}
		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		R[k][k] = norm(&Q[k][0],N_CHANNELS,N_CHANNELS);
		rk = one/R[k][k];

		#if NUM_CORES > 1
		for( j = start_NC; (j+1)<start_NC + (blockSize_NC & 0xFFFE); j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;

		} if (j<start_NC + (blockSize_NC & 0xFFFE)){
			float Qjk0 = Q[j][k];
			Q[k][j] = Qjk0 * rk;
		}
		#else
		for( j=0; (j+1)<N_CHANNELS; j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
		} if (j<N_CHANNELS){
			float Qjk0 = Q[k][j];
			Q[k][j] = Qjk0 * rk;
		}
		#endif

		#if NUM_CORES > 1
		if(blockSize_NC & 0x0001){
			Q[k][j] = Q[k][j]*rk;}
		pi_cl_team_barrier();
		#endif

	}
	return;
}
void qr_gramSmidt_0_2(float Q[NO2/NCHUNK][NO2/NCHUNK], float R[NO2/NCHUNK][MODEL_ORDER], float input[MODEL_ORDER][NO2/NCHUNK]){
	int j;
	int N_CHANNELS=NO2/NCHUNK;
	int EV_WINDOWS_SIZE=MODEL_ORDER;

	#if NUM_CORES > 1
	int blockSize_NC = N_CHANNELS/NUM_CORES;
	int start_NC = pi_core_id()*blockSize_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NC = N_CHANNELS - (NUM_CORES - 1)* blockSize_NC;}
	#endif

	#if NUM_CORES > 1
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=start_NC; j<start_NC+blockSize_NC; j++)
			Q[j][i]=0;
	}


	for(int j=0; j<MODEL_ORDER; j++){
		for(int i=start_NC; i<start_NC+blockSize_NC; i++)
			R[i][j]=0;
	}

	pi_cl_team_barrier();
	#else
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<NO/NCHUNK; j++){
			Q[j][i]=0;
		}
	}

	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<MODEL_ORDER; j++){
			R[i][j]=0;
		}
	}
	#endif

	for(int k=0; k<EV_WINDOWS_SIZE; k++){

		#if NUM_CORES > 1
		for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<start_NC + blockSize_NC){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#else
		for(j=0; (j+1)<N_CHANNELS; j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<N_CHANNELS){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#endif

		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		for(int i=0; i<k; i++){
			#if NUM_CORES > 1
			temp[pi_core_id()]=0;
			for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				float ji1 = Q[i][j+1];float jk1 = Q[k][j+1];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0) + (ji1 * jk1);
			}

			if((j<start_NC + blockSize_NC)){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0);
			}

			pi_cl_team_barrier();

			if(pi_core_id()==0)
				for(j=0; j<NUM_CORES; j++){
			    		R[i][k]+=temp[j];
			    	}
			pi_cl_team_barrier();

			#else

			for(j=0; (j+1)<N_CHANNELS; j+=2){
				float Qji0 = Q[i][j];
				float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];
				float Qjk1 = Q[k][j+1];
				float Rik  = R[i][k];
				R[i][k] = Rik + (Qji0 * Qjk0) + (Qji1 * Qjk1);
            }
            if(j<N_CHANNELS){
				float Qji0 = Q[i][j]; float Qjk0 = Q[k][j];
				float Rik  = R[i][k];
				float temp = (Qji0 * Qjk0);
				R[i][k] = Rik + temp;
            }
			#endif

			#if NUM_CORES > 1
			for(j = start_NC; (j+1<start_NC + blockSize_NC); j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				//hal_compiler_barrier();

				Q[k][j] = Qjk0 - Rik*Qji0;
                Q[k][j+1] = Qjk1 - Rik*Qji1;
            }
			if(j<start_NC + blockSize_NC){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#else
			for(j=0; j+1<N_CHANNELS; j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				Q[k][j] = Qjk0 - Rik*Qji0;
				Q[k][j+1] = Qjk1 - Rik*Qji1;
			}
			if(j<N_CHANNELS){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#endif

			#if NUM_CORES > 1
			pi_cl_team_barrier();
			#endif
		}
		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		R[k][k] = norm(&Q[k][0],N_CHANNELS,N_CHANNELS);
		rk = one/R[k][k];

		#if NUM_CORES > 1
		for( j = start_NC; (j+1)<start_NC + (blockSize_NC & 0xFFFE); j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;

		} if (j<start_NC + (blockSize_NC & 0xFFFE)){
			float Qjk0 = Q[j][k];
			Q[k][j] = Qjk0 * rk;
		}
		#else
		for( j=0; (j+1)<N_CHANNELS; j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
		} if (j<N_CHANNELS){
			float Qjk0 = Q[k][j];
			Q[k][j] = Qjk0 * rk;
		}
		#endif

		#if NUM_CORES > 1
		if(blockSize_NC & 0x0001){
			Q[k][j] = Q[k][j]*rk;}
		pi_cl_team_barrier();
		#endif

	}
	return;
}*/

inline float Sqrt(float x) {
	float res;
	asm("fsqrt.s %0, %1":"=f"(res):"f"(x));
	return res;
}

void qr_gramSmidt_next(float Q[2*MODEL_ORDER][2*MODEL_ORDER], float R[2*MODEL_ORDER][MODEL_ORDER], float input[MODEL_ORDER][2*MODEL_ORDER]){
	int j;
	int N_CHANNELS=2*MODEL_ORDER;
	int EV_WINDOWS_SIZE=MODEL_ORDER;

	#if NUM_CORES > 1
	if(pi_core_id()==0){
	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<2*MODEL_ORDER; j++)
			Q[j][i]=0;
	}


	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<MODEL_ORDER; j++)
			R[i][j]=0;
	}
	}
	pi_cl_team_barrier();
	#else
	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<2*MODEL_ORDER; j++){
			Q[j][i]=0;
		}
	}

	for(int i=0; i<2*MODEL_ORDER; i++){
		for(j=0; j<MODEL_ORDER; j++){
			R[i][j]=0;
		}
	}
	#endif

	#if NUM_CORES > 1
	int blockSize_NC = N_CHANNELS/NUM_CORES;
	int start_NC = pi_core_id()*blockSize_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NC = N_CHANNELS - (NUM_CORES - 1)* blockSize_NC;}
	#endif

	for(int k=0; k<EV_WINDOWS_SIZE; k++){
		#if NUM_CORES > 1
		for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<start_NC + blockSize_NC){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#else
		for(j=0; (j+1)<N_CHANNELS; j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<N_CHANNELS){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#endif		

		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		for(int i=0; i<k; i++){
			#if NUM_CORES > 1
			temp[pi_core_id()]=0;
			for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				float ji1 = Q[i][j+1];float jk1 = Q[k][j+1];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0) + (ji1 * jk1);				
			}
			if((j<start_NC + blockSize_NC)){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0);
			}
			
			pi_cl_team_barrier();

			if(pi_core_id()==0)
				for(j=0; j<NUM_CORES; j++){
			    		R[i][k]+=temp[j];
			    	}
			pi_cl_team_barrier();

			#else

			for(j=0; (j+1)<N_CHANNELS; j+=2){
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];
				float Rik  = R[i][k];
				R[i][k] = Rik + (Qji0 * Qjk0) + (Qji1 * Qjk1);
            }
            if(j<N_CHANNELS){				
				float Qji0 = Q[i][j]; float Qjk0 = Q[k][j];
				float Rik  = R[i][k];
				float temp = (Qji0 * Qjk0);
				R[i][k] = Rik + temp;                	
            }
			#endif
			
			#if NUM_CORES > 1
			for(j = start_NC; (j+1<start_NC + blockSize_NC); j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];
				
				//hal_compiler_barrier();
                		
				Q[k][j] = Qjk0 - Rik*Qji0;
                Q[k][j+1] = Qjk1 - Rik*Qji1;
            }
			if(j<start_NC + blockSize_NC){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#else
			for(j=0; j+1<N_CHANNELS; j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				Q[k][j] = Qjk0 - Rik*Qji0;
				Q[k][j+1] = Qjk1 - Rik*Qji1;
			}
			if(j<N_CHANNELS){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#endif
			
			#if NUM_CORES > 1 
			pi_cl_team_barrier();
			#endif
		}
		#if NUM_CORES > 1 
		pi_cl_team_barrier();
		#endif
		
		R[k][k] = norm(&Q[k][0],N_CHANNELS,N_CHANNELS);
		rk = one/R[k][k];
		
		#if NUM_CORES > 1
		for( j = start_NC; (j+1)<start_NC + (blockSize_NC & 0xFFFE); j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
			
		} if (j<start_NC + (blockSize_NC & 0xFFFE)){
			float Qjk0 = Q[j][k];
			Q[k][j] = Qjk0 * rk;
		}
		#else
		for( j=0; (j+1)<N_CHANNELS; j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
		} if (j<N_CHANNELS){
			float Qjk0 = Q[k][j];
			Q[k][j] = Qjk0 * rk;
		}
		#endif
		
		#if NUM_CORES > 1 
		if(blockSize_NC & 0x0001){
			Q[k][j] = Q[k][j]*rk;}
		pi_cl_team_barrier();
		#endif
		
	}
	return;
}

void qr_gramSmidt_0(float Q[NO/NCHUNK][NO/NCHUNK], float R[NO/NCHUNK][MODEL_ORDER], float input[MODEL_ORDER][NO/NCHUNK]){
	int j;
	int N_CHANNELS=NO/NCHUNK;
	int EV_WINDOWS_SIZE=MODEL_ORDER;

	#if NUM_CORES > 1
	int blockSize_NC = N_CHANNELS/NUM_CORES;
	int start_NC = pi_core_id()*blockSize_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NC = N_CHANNELS - (NUM_CORES - 1)* blockSize_NC;}
	#endif

	#if NUM_CORES > 1
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=start_NC; j<start_NC+blockSize_NC; j++)
			Q[j][i]=0;
	}


	for(int j=0; j<MODEL_ORDER; j++){
		for(int i=start_NC; i<start_NC+blockSize_NC; i++)
			R[i][j]=0;
	}

	pi_cl_team_barrier();
	#else
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<NO/NCHUNK; j++){
			Q[j][i]=0;
		}
	}

	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<MODEL_ORDER; j++){
			R[i][j]=0;
		}
	}
	#endif

	for(int k=0; k<EV_WINDOWS_SIZE; k++){
		#if NUM_CORES > 1
		for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<start_NC + blockSize_NC){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#else
		for(j=0; (j+1)<N_CHANNELS; j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<N_CHANNELS){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#endif		

		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		for(int i=0; i<k; i++){
			#if NUM_CORES > 1
			temp[pi_core_id()]=0;
			for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				float ji1 = Q[i][j+1];float jk1 = Q[k][j+1];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0) + (ji1 * jk1);				
			}
			if((j<start_NC + blockSize_NC)){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0);
			}
			
			pi_cl_team_barrier();

			if(pi_core_id()==0)
				for(j=0; j<NUM_CORES; j++){
			    		R[i][k]+=temp[j];
			    	}
			pi_cl_team_barrier();

			#else

			for(j=0; (j+1)<N_CHANNELS; j+=2){
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];
				float Rik  = R[i][k];
				R[i][k] = Rik + (Qji0 * Qjk0) + (Qji1 * Qjk1);
            }
            if(j<N_CHANNELS){				
				float Qji0 = Q[i][j]; float Qjk0 = Q[k][j];
				float Rik  = R[i][k];
				float temp = (Qji0 * Qjk0);
				R[i][k] = Rik + temp;                	
            }
			#endif
			
			#if NUM_CORES > 1
			for(j = start_NC; (j+1<start_NC + blockSize_NC); j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];
				
				//hal_compiler_barrier();
                		
				Q[k][j] = Qjk0 - Rik*Qji0;
                Q[k][j+1] = Qjk1 - Rik*Qji1;
            }
			if(j<start_NC + blockSize_NC){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#else
			for(j=0; j+1<N_CHANNELS; j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				Q[k][j] = Qjk0 - Rik*Qji0;
				Q[k][j+1] = Qjk1 - Rik*Qji1;
			}
			if(j<N_CHANNELS){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#endif
			
			#if NUM_CORES > 1 
			pi_cl_team_barrier();
			#endif
		}
		#if NUM_CORES > 1 
		pi_cl_team_barrier();
		#endif
		
		R[k][k] = norm(&Q[k][0],N_CHANNELS,N_CHANNELS);
		rk = one/R[k][k];
		
		#if NUM_CORES > 1
		for( j = start_NC; (j+1)<start_NC + (blockSize_NC & 0xFFFE); j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
			
		} if (j<start_NC + (blockSize_NC & 0xFFFE)){
			float Qjk0 = Q[j][k];
			Q[k][j] = Qjk0 * rk;
		}
		#else
		for( j=0; (j+1)<N_CHANNELS; j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
		} if (j<N_CHANNELS){
			float Qjk0 = Q[k][j];
			Q[k][j] = Qjk0 * rk;
		}
		#endif
		
		#if NUM_CORES > 1 
		if(blockSize_NC & 0x0001){
			Q[k][j] = Q[k][j]*rk;}
		pi_cl_team_barrier();
		#endif
		
	}
	return;
}

void qr_gramSmidt_0_2(float Q[NO2/NCHUNK][NO2/NCHUNK], float R[NO2/NCHUNK][MODEL_ORDER], float input[MODEL_ORDER][NO2/NCHUNK]){
	int j;
	int N_CHANNELS=NO2/NCHUNK;
	int EV_WINDOWS_SIZE=MODEL_ORDER;

	#if NUM_CORES > 1
	int blockSize_NC = N_CHANNELS/NUM_CORES;
	int start_NC = pi_core_id()*blockSize_NC;
	if(pi_core_id()==(NUM_CORES - 1)){
		blockSize_NC = N_CHANNELS - (NUM_CORES - 1)* blockSize_NC;}
	#endif

	#if NUM_CORES > 1
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=start_NC; j<start_NC+blockSize_NC; j++)
			Q[j][i]=0;
	}


	for(int j=0; j<MODEL_ORDER; j++){
		for(int i=start_NC; i<start_NC+blockSize_NC; i++)
			R[i][j]=0;
	}

	pi_cl_team_barrier();
	#else
	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<NO/NCHUNK; j++){
			Q[j][i]=0;
		}
	}

	for(int i=0; i<NO/NCHUNK; i++){
		for(int j=0; j<MODEL_ORDER; j++){
			R[i][j]=0;
		}
	}
	#endif

	for(int k=0; k<EV_WINDOWS_SIZE; k++){
		#if NUM_CORES > 1
		for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<start_NC + blockSize_NC){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#else
		for(j=0; (j+1)<N_CHANNELS; j+=2){
			float in0 = input[k][j];float in1 = input[k][j+1];
			Q[k][j] = in0;
			Q[k][j+1] = in1;
		}
		if(j<N_CHANNELS){
			float in0 = input[k][j];
			Q[k][j] = in0;
		}
		#endif		

		#if NUM_CORES > 1
		pi_cl_team_barrier();
		#endif

		for(int i=0; i<k; i++){
			#if NUM_CORES > 1
			temp[pi_core_id()]=0;
			for(j = start_NC; ((j+1)<start_NC + blockSize_NC); j+=2){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				float ji1 = Q[i][j+1];float jk1 = Q[k][j+1];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0) + (ji1 * jk1);				
			}
			if((j<start_NC + blockSize_NC)){
				float ji0 = Q[i][j];float jk0 = Q[k][j];
				temp[pi_core_id()] = temp[pi_core_id()] + (ji0 * jk0);
			}
			
			pi_cl_team_barrier();

			if(pi_core_id()==0)
				for(j=0; j<NUM_CORES; j++){
			    		R[i][k]+=temp[j];
			    	}
			pi_cl_team_barrier();

			#else

			for(j=0; (j+1)<N_CHANNELS; j+=2){
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];
				float Rik  = R[i][k];
				R[i][k] = Rik + (Qji0 * Qjk0) + (Qji1 * Qjk1);
            }
            if(j<N_CHANNELS){				
				float Qji0 = Q[i][j]; float Qjk0 = Q[k][j];
				float Rik  = R[i][k];
				float temp = (Qji0 * Qjk0);
				R[i][k] = Rik + temp;                	
            }
			#endif
			
			#if NUM_CORES > 1
			for(j = start_NC; (j+1<start_NC + blockSize_NC); j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];
				
				//hal_compiler_barrier();
                		
				Q[k][j] = Qjk0 - Rik*Qji0;
                Q[k][j+1] = Qjk1 - Rik*Qji1;
            }
			if(j<start_NC + blockSize_NC){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#else
			for(j=0; j+1<N_CHANNELS; j+=2){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				float Qji1 = Q[i][j+1];float Qjk1 = Q[k][j+1];

				Q[k][j] = Qjk0 - Rik*Qji0;
				Q[k][j+1] = Qjk1 - Rik*Qji1;
			}
			if(j<N_CHANNELS){
				float Rik = R[i][k];
				float Qji0 = Q[i][j];float Qjk0 = Q[k][j];
				Q[k][j] = Qjk0 - Rik*Qji0;
			}
			#endif
			
			#if NUM_CORES > 1 
			pi_cl_team_barrier();
			#endif
		}
		#if NUM_CORES > 1 
		pi_cl_team_barrier();
		#endif
		
		R[k][k] = norm(&Q[k][0],N_CHANNELS,N_CHANNELS);
		rk = one/R[k][k];
		
		#if NUM_CORES > 1
		for( j = start_NC; (j+1)<start_NC + (blockSize_NC & 0xFFFE); j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
			
		} if (j<start_NC + (blockSize_NC & 0xFFFE)){
			float Qjk0 = Q[j][k];
			Q[k][j] = Qjk0 * rk;
		}
		#else
		for( j=0; (j+1)<N_CHANNELS; j+=2){
			float Qjk0 = Q[k][j];float Qjk1 = Q[k][j+1];
			Q[k][j] = Qjk0 * rk;
			Q[k][j+1] = Qjk1 * rk;
		} if (j<N_CHANNELS){
			float Qjk0 = Q[k][j];
			Q[k][j] = Qjk0 * rk;
		}
		#endif
		
		#if NUM_CORES > 1 
		if(blockSize_NC & 0x0001){
			Q[k][j] = Q[k][j]*rk;}
		pi_cl_team_barrier();
		#endif
		
	}
	return;
}
