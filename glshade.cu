/******************************************************************************/
/////////////////////////////GL-SHADE//////////////////////////////////
/******************************************************************************/
/* 
Algorithm: GL-SHADE
Author: CINVESTAV-IPN (Evolutionary Computation Group)
Implemented using: C++/CUDA + OpenMP
Year: 2020
Requirements: CUDA version 7.0 or higher (since syntax c++11 is used) 
Compile: nvcc glshade.cu -O2 -D=f$ -Xcompiler -fopenmp 
	$: is the objective funtion identifier [1, 2, ...., 15]
Run: a.out arg1 arg2 arg3 
	arg1: size of population 1
	arg2: size of population 2
	arg3: random number generator seed    
*/

/******************************************************************************/
/////////////////////////////NORMAL HEADERS//////////////////////////////////
/******************************************************************************/
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <omp.h> 
#include <stdio.h>
#include <unistd.h>
#include <random>
#include <string>
#include <chrono>
using namespace std::chrono;
using namespace std;

/******************************************************************************/
///////////////////////CONSTANTS AND OUTPUT FILES////////////////////////////////
/******************************************************************************/
//readable by both host and device
#define PI (3.141592653589793238462643383279)
#define E  (2.718281828459045235360287471352)
const int N_threads = 256; //number of threads [CUDA]
const int N_blocks = 32; //number of blocks [CUDA]
#define maxThreads 6 //max number of cpu cores activated [OMP]
const int dim = 1000; //objective function dimension
FILE *file_results; //output file: for recording best solution so far by checkpoint.
FILE *file_report;  //output file: fot recording the evolutionary history.
struct evol_data_struct{double Cr,F; int j_rand,a,b,p_best;};//used by: shade
struct evol_data_struct2{double F,Cr; int a,b,p_best;int Jrand,Jend;};//used by: eshadels 
struct rank_ind {int id;double fitness;};//used by: shade/eshadels

/******************************************************************************/
/////////////////////////////MY HEADERS//////////////////////////////////
/******************************************************************************/
/*
Useful functions for generating random numbers. The variable Rseed is declared in this
header so it can be thought as a global variables as well. Rseed is the random number 
generator (rng) seed for starting up both rng gpu and rng cpu.  
*/
#include "rand.c"

/*
The objective_function.h includes all the data and functions needed for 
running the kernel of the objective function named F_D. In particular, 
we can do so by invoking the set_objective_function() procedure.  
*/
#include "cec13/objective_function.h" 

/*
Kernel functions  
*/ 
#include "kernel.cu"

/*
Import classes
*/ 
#include "shade.cu"
#include "eshade_ls.cu"
#include "mts_ls1.cpp"

/*
Error handler 
*/
#include "error_handler.c"


int main(int argc, char const *argv[])
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now(); //start timer
	/******************************************************************************/
	/////////////////////////////PRELIMINARIES//////////////////////////////////
	/******************************************************************************/
    //Make sure 4 input arguments were given 
    if((argc-1) != 3)
    {
        printf("Error: 3 input arguments expected\n");
        exit(EXIT_FAILURE);
    } 
    /* 
    Define population sizes and random number generator (rng) seed
    */ 
    int NP1,NP2; //problem id and popsizes
    //Rseed is the rng seed <- rand.c
    // ID is the objective function identifier determined at compilation time <- objective_function.h
    check_and_set_popsize(argv,1,NP1); //popsize 1 
    check_and_set_popsize(argv,2,NP2); //popsize 2
    check_and_set_Rseed(argv,3,Rseed); //Define seed for cpu RNG  

    /* 
    Define objetive function kernel F_D (implementation of the objective function on device)
    and set bounds accordingly. F_D can be used at any scope.
    F_H is the OMP version of F_D which have the same scope that the latter. 
    */ 
    float lb,ub; //bounds
    set_objective_function(lb,ub); //<- objective_function.h 

    /* 
	RNG 
    */ 
    int gpuseed = int(Rseed*100);// seed for gpu rng
    int cpuseed = int(Rseed*10);// seed for cpu rng
    curandState *state_D;//it stores the rng state in device
    cudaMalloc(&state_D,N_blocks*N_threads*sizeof(curandState));
    randomize_gpu<<<N_blocks,N_threads>>>(gpuseed,state_D); //start up gpu rng
    randomize(); //start up cpu rng
    default_random_engine rng (cpuseed); //rng state in host

    /* 
	Output files to record results. 
    */ 
    string file_results_name = "glshade_results_"+ to_string(NP1) + "pop1_" + to_string(NP2) + "pop2_f" + to_string(ID) + ".csv";
    string file_report_name = "glshade_report_"+ to_string(NP1) + "pop1_" + to_string(NP2) + "pop2_f" + to_string(ID) + ".txt";

    //Open and init output files
    if( access( file_results_name.c_str(), F_OK ) == -1 ) //if file does not exist then...
    {    
        file_results = fopen (file_results_name.c_str(),"a");
        fprintf(file_results, "FEs,f,seed,solution_value\n"); //initialize it
    }
    else
        file_results = fopen (file_results_name.c_str(),"a");
    file_report = fopen(file_report_name.c_str(),"a");
	/******************************************************************************/
	//////////////////////////////START ALGORITHM///////////////////////////////////
	/******************************************************************************/
	/*
	Set components and gl-shade control parameters
	*/
    double before,after;
    ind global_best;  // best candidate to solution; ind data type is declared in objective_function.h 
    int stop_criterion = 3000000;
    int G_FEs = 25000;
    int L_FEs = 25000;
    int current_FEs = 0;
    int it = 0;
    shade DE1(lb,ub,NP1,100,dim,G_FEs); //the 4th argument is H_max1
    eshade_ls DE2(lb,ub,NP2,100,dim,L_FEs); //the 4th argument is H_max2
    mts_ls1 LS1(lb,ub,dim,L_FEs); 

	/*
	Initialization
	*/
	DE1.init_population_in_device(state_D,global_best,current_FEs); // population 1
	DE2.init_population_in_device(state_D,global_best,current_FEs); // population 2
    //Report 
    fprintf(file_report,"**************************** Initialization ********************************\n");
    fprintf(file_report,"\t\tImprovement %d to %d ====> doesn't apply\n",it,it); 
    fprintf(file_report,"\t\tglobal_best ====> f(X):%.4e\n",global_best.fx); 
    fprintf(file_report,"\t\tFEs status ====> %d\n",current_FEs); 

    before = global_best.fx;
	LS1.enhance(global_best,current_FEs,stop_criterion); //enhance best candidate to solution
    after = global_best.fx;
    //Report
    fprintf(file_report,"************************* Early Local Search: Iteration %d *************************\n",it);
    fprintf(file_report,"\t\tImprovement after early local search ====> %.3lf %%\n",100*(before-after)/before); 
    fprintf(file_report,"\t\tglobal_best ====> f(X):%.4e\n",global_best.fx); 
    fprintf(file_report,"\t\tFEs status ====> %d\n",current_FEs);

	/*
	Searching engine
	*/
	while (current_FEs < stop_criterion)
	{
		//Update iteration counter
		it++;

        //Record fitness at the beginning of global exploration
        before = global_best.fx;

        //Apply a specialized global search evolution scheme 
        DE1.evolve_in_device(rng,state_D,global_best,current_FEs,stop_criterion);

        //Apply a specialized local search evolution scheme
        DE2.evolve_in_device(rng,state_D,global_best,current_FEs,stop_criterion);

        //Record fitness at the end of local search
        after = global_best.fx;

        /******************* Report ************************/
        fprintf(file_report,"**************************** Iteration %d ****************************\n",it);
        fprintf(file_report,"\t\tImprovement %d to %d ====> %.3lf %%\n",it-1,it,100*(before-after)/before);
        fprintf(file_report,"\t\tglobal_best ====> f(X):%.4e\n",global_best.fx); 
        fprintf(file_report,"\t\tFEs status ====> %d\n",current_FEs);
	}
	/******************************************************************************/
	////////////////////////////////THE CLOSING/////////////////////////////////
	/******************************************************************************/
    high_resolution_clock::time_point t2 = high_resolution_clock::now(); //stop timer
    auto duration = duration_cast<seconds>( t2 - t1 ).count();

    /* Final report */
    fprintf(file_report,"**************************** GL-SHADE REPORT ****************************\n");
    fprintf(file_report,"Benchmark = CEC 2013\n");
    fprintf(file_report,"Objective Function = f%d \n",ID);
    fprintf(file_report,"Lower Bound = %.1f and Upper Bound = %.1f\n",lb,ub);
    fprintf(file_report,"RNG Seed = %.3f\n",Rseed);
    fprintf(file_report,"Parameters:\n");
    fprintf(file_report,"\tmaxFEs = 3e6\n");
    fprintf(file_report,"\tG_FEs = L_FEs = 25e3\n");
    fprintf(file_report,"\tNP1 = %d and NP2 = %d\n",NP1,NP2);
    fprintf(file_report,"\tH_maxsize1 = 100 and H_maxsize2 = 100 \n");
    fprintf(file_report,"\tw_min = 0.0 and w_max = 0.2 \n");
    fprintf(file_report,"Did it find a feasible solution?: %d\n",VerifyBestSolution(global_best.x,lb,ub));
    fprintf(file_report,"Solution after computing %d objective function evaluations\n",stop_criterion);
    fprintf(file_report,"Solution found when the objective function evaluations status was %d\n",global_best.FEs_when_found);
    fprintf(file_report,"Execution time = %ld seconds\n",duration);
    fprintf(file_report,"Solution value = %.6e\n",global_best.fx);
    fprintf(file_report,"Solution: [");
    for (it = 0; it < dim-1; ++it) fprintf(file_report,"%.6lf,",global_best.x[it]);
    fprintf(file_report,"%.6lf]\n\n,",global_best.x[dim-1]);

    //Console ouput
    printf("Execution Time: %ld sec\n",duration);

    //Close output files
    fclose (file_results);
    fclose(file_report);

    //free space
    cudaFree(state_D);
    free_objective_function_data(); //<- objective_function.h
    DE1.free_memory();
    DE2.free_memory();

	return 0;
}





