/* 
Algorithm: GL-SHADE
Author: CINVESTAV-IPN (Evolutionary Computation Group)
Implemented using: C++/CUDA + OpenMP
Year: 2020
Requirements: CUDA version 7.0 or higher (since syntax c++11 is used)   
*/

#include <stdio.h>
#include <errno.h>   
#include <stdlib.h>
#include <unistd.h>
#include <random>
#include <string>
#include "Benchmark/HeaderGPU.h"
#include <curand_kernel.h>
#include "rand.c"

// For measuring running time
#include <chrono>
using namespace std::chrono;

//Custom structs 
/*
-> struct ind {double x[dim]; double fx; unsigned FEs_when_found;} is 
defined in Benchmark/HeaderGPU.h 
This struct is used by glshade/mtsls1/shade/eshadels
*/
struct evol_data_struct{double Cr,F; int j_rand,a,b,p_best;};//used by: shade
struct evol_data_struct2{double F,Cr; int a,b,p_best;int Jrand,Jend;};//used by: eshadels 
struct rank_ind {int id;double fitness;};//used by: shade/eshadels

//output files 
FILE *file_results;
FILE *file_report;

#include "error_handler.c"
#include "kernel.cu" //all kernel procedures are here 
#include "mts_ls1.cpp"
#include "shade.cu"
#include "eshade_ls.cu"

int main(int argc, char const *argv[])
{
    //Make sure four input arguments were given 
    if((argc-1) != 4)
    {
        printf("Error: 4 input arguments expected\n");
        exit(EXIT_FAILURE);
    }

    //Start timer
    high_resolution_clock::time_point t1 = high_resolution_clock::now(); 

    //Declare objetive function and main parameters   
    int DIM; //dimension
    float lb,ub; //lower and upper bound
    int NP1,NP2;//popsizes 

    //Define objetive function and main parameters
    //Population sizes  
    check_and_set_popsize(argv,1,NP1);//popsize 1 
    check_and_set_popsize(argv,2,NP2);//popsize 2
    //Define seed for cpu RNG.
    check_and_set_Rseed(argv,3); 
    //Objective function 
    check_and_set_FunctionID(argv,4);

    //Define output files names
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

    //Random number generator stuff
    int GPUseed = int(Rseed*100);// define seed for gpu RNG
    curandState *state_D;//it stores the RNG state in device
    cudaMalloc(&state_D,N_blocks*N_threads*sizeof(curandState));//Number of blocks and threads are defined in HeaderGPU.h
    randomize_gpu<<<N_blocks,N_threads>>>(GPUseed,state_D); //Start up gpu RNG
    randomize(); //Start up cpu RNG
    default_random_engine rng (int(Rseed*10)); //Start another cpu RNG

    //Set benchmark parameters to work with F=ID. This procedure will set F_D and F_H properly. So after procedure finishes, they can be used at any scope.
    //F_D is the gpu implementation of the function and F_H the cpu + openMP implementation.
    SetBenchFramework(DIM,lb,ub);//defined in Benchmark/Aux.cu

    /******************************* Start algorithm ******************************/
    //Declarations
    unsigned maxFEs,G_FEs,L_FEs,current_FEs,it;
    double before,after;
    ind current_best; // the best currently.
    ind global_best;  // the best seen so far
    shade DE1;
    eshade_ls DE2;
    mts_ls1 LS1; //multi trajectory search - local search 1

    //Definitions
    maxFEs = 3000000;
    G_FEs = L_FEs = 25000;
    current_FEs = it = 0;

    //Initializing
    LS1.init(DIM,maxFEs,L_FEs,lb,ub);
    DE1.set_parameters(NP1,DIM,maxFEs,G_FEs,lb,ub); 
    DE2.set_parameters(NP2,DIM,maxFEs,L_FEs,lb,ub);
    DE1.init_population_in_device(state_D,current_FEs,current_best);//main population
    DE2.init_population_in_device(state_D,current_FEs);//secondary population
    global_best = current_best; //take as global_best the best that was born in population 1
    //Report 
    fprintf(file_report,"**************************** Initialization ********************************\n");
    fprintf(file_report,"\t\tImprovement %u to %u ====> doesn't apply\n",it,it); 
    fprintf(file_report,"\t\tglobal_best ====> f(X):%.4e\n",global_best.fx); 
    fprintf(file_report,"\t\tFEs status ====> %u\n",current_FEs); 

    /******************* Early local search ************************/
    before = global_best.fx;
    LS1.search(current_FEs,DE1.pop[DE1.best_location].x,DE1.pop[DE1.best_location].fx,current_best);
    if(current_best.fx<global_best.fx) global_best=current_best; //Update global_best
    after = global_best.fx; 
    //Report
    fprintf(file_report,"************************* Early Local Search: Iteration %u *************************\n",it);
    fprintf(file_report,"\t\tImprovement after early local search ====> %.3lf %%\n",100*(before-after)/before); 
    fprintf(file_report,"\t\tglobal_best ====> f(X):%.4e\n",global_best.fx); 
    fprintf(file_report,"\t\tFEs status ====> %u\n",current_FEs);

    /******************* evolving procedure ************************/
    // Iterate
    while (current_FEs < maxFEs)
    {   
        //Advance iteration counter
        it += 1;
    
        /******************* Apply SHADE ************************/
        //Record fitness at the beginning of global exploration
        before = global_best.fx;

        //migrate back current_best from pop2 to pop1
        DE1.receive(global_best);

        //Apply a specialized global search evolution scheme 
        DE1.evolve_in_device(current_FEs,rng,current_best,state_D);

        //Update global_best
        if(current_best.fx<global_best.fx) global_best=current_best;

        /******************* Apply eSHADE-ls ************************/
        //migrate current_best from pop1 to pop2
        DE2.receive(global_best);

        //Apply a specialized local search evolution scheme 
        DE2.evolve_in_device(current_FEs,rng,current_best);

        //Update global_best
        if(current_best.fx<global_best.fx) global_best=current_best;

        //Record fitness at the end of local search
        after = global_best.fx;

        /******************* Report ************************/
        fprintf(file_report,"**************************** Iteration %u ****************************\n",it);
        fprintf(file_report,"\t\tImprovement %d to %d ====> %.3lf %%\n",it-1,it,100*(before-after)/before);
        fprintf(file_report,"\t\tglobal_best ====> f(X):%.4e\n",global_best.fx); 
        fprintf(file_report,"\t\tFEs status ====> %u\n",current_FEs);
    }
    //Stop timer
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
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
    fprintf(file_report,"\tNP1 = %u and NP2 = %u\n",NP1,NP2);
    fprintf(file_report,"\tH_maxsize1 = 100 and H_maxsize2 = 100 \n");
    fprintf(file_report,"\tw_min = 0.0 and w_max = 0.2 \n");
    fprintf(file_report,"Did it find a feasible solution?: %d\n",VerifyBestSolution(global_best.x,lb,ub));
    fprintf(file_report,"Solution after computing %d objective function evaluations\n",maxFEs);
    fprintf(file_report,"Solution found when the objective function evaluations status was %d\n",global_best.FEs_when_found);
    fprintf(file_report,"Execution time = %ld seconds\n",duration);
    fprintf(file_report,"Solution value = %.6e\n",global_best.fx);
    fprintf(file_report,"Solution: [");
    for (it = 0; it < DIM-1; ++it) fprintf(file_report,"%.6lf,",global_best.x[it]);
    fprintf(file_report,"%.6lf]\n\n,",global_best.x[DIM-1]);
    
    //Console ouput
    printf("Execution Time: %ld sec\n",duration);//overall time, printf("%ld\n",duration);
    /******************************************************************************/
    //Close output files
    fclose (file_results);
    fclose(file_report);
    //Free memory space
    DE1.free_memory();
    DE2.free_memory();
    cudaFree(state_D);
    FreeBenchData();//defined in Benchmark/HeaderGPU.h
    //End
    return 0;
} 
