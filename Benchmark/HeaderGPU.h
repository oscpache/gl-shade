/*
This file implements the benchmark functions provided in the following technical report: 
X. Li, K. Tang, M. Omidvar, Z. Yang and K. Qin, 
“Benchmark Functions for the CEC’2013 Special Session and Competition on Large Scale Global Optimization,” 
Technical Report, Evolutionary Computation and Machine Learning Group,
RMIT University, Australia, 2013.

There already is a CPU based implementation of such benchmark set (visit:http://www.tflsgo.org/special_sessions/cec2019#benchmark-competition),
however the present implementation is GPU based. 
*/
/******************************************************************************/
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
using namespace std;
#include <cuda.h>
#include <omp.h>

/*****************************GLOBAL DATA****************************************/

//This data is visible for both: host and device.
#define PI (3.141592653589793238462643383279)
#define E  (2.718281828459045235360287471352)
const int N_threads = 256;//for CUDA
const int N_blocks = 32;//for CUDA
const int dim = 1000;//dimension for all problems except F13-F14
const int one = 1;
const int maxThreads = 4;// for OMP
const int dim_ovl = 905;//dimension for F13-F14
const int overlap = 5;//overlap size

//Only visible by host 
int s_size;//number of subcomponents. It is set at run time 
int ID;//objective function identifier 

//Only visible by device (stored in device constant memory)  
__constant__ int s_size_d; //number of subcomponents. It is set at run time
__constant__ float lb_d;//lower bound
__constant__ float ub_d;//upper bound 

//Custom structs
struct ind {double x[dim]; double fx; unsigned FEs_when_found;};
struct row {double col[100];};//  

//read only data [HOST]
double *Ovector;
int    *Pvector;
double *r25;
double *r50;
double *r100;
int    *s;
double *w;
double **OvectorVec;

//read only data [DEVICE]
double *Ovector_D;
int    *Pvector_D;
double *r25_D;
double *r50_D;
double *r100_D;
int    *s_D;
double *w_D;
row *OvectorVec_D;

//read and write [DEVICE]
struct storage {double anotherz[dim]; double anotherz1[dim];}; storage *mem_D;
/******************************************************************************/
/*
Benchmark.cu includes all base procedures for implementing f"ID".cu and Aux.cu    
*/
#include "Benchmark.cu"

/*
It is selected a problem at compilation time. The file f"ID".cu includes the corresponding 
F_D and F_H procedures; F_D is the gpu implementation of the function and F_H the cpu + openMP implementation.
Here is the reason why the following flags are added when compiling: -D=fID -Xcompiler -fopenmp
where ID is an integer ranging from 1 to 15.
*/
#if f1
# include "f1.cu"
#elif f2
# include "f2.cu"
#elif f3
# include "f3.cu"
#elif f4
# include "f4.cu"
#elif f5
# include "f5.cu"
#elif f6
# include "f6.cu"
#elif f7
# include "f7.cu"
#elif f8
# include "f8.cu"
#elif f9
# include "f9.cu"
#elif f10
# include "f10.cu"
#elif f11
# include "f11.cu"
#elif f12
# include "f12.cu"
#elif f13
# include "f13.cu"
#elif f14
# include "f14.cu"
#elif f15
# include "f15.cu"
#endif

/*
The file Aux.cu includes the SetBenchFramework procedure which sets all 
the required data(lower bound, upper bound, rotation vectors, shifting vector, 
weigth vector, number of subcomponents, etc.) in both host and device  
according to the objective function selected
*/
#include "Aux.cu"


