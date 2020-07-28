/*
- This file defines all data needed to compute any of the 15 objective functions
since such the objective function is determined at compilation time, in other words, 
the problem to be chosen is not known in advance.
- Additionally, all functions needed for setting correctly the lower and 
upper bounds, dimension, number of subcomponents, etc. corresponding 
to the objective function selected are implemented here.
*/

/******************************************************************************/
/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*****************************DATA*********************************************/
//Readable and modifiable only by host
int s_size; //number of components  
const int overlap = 5; //overlaping size

/* 
For overlapping class functions F13~F14. This data is readable only 
by device; stored in constant memory 
*/  
__constant__ int dim_ovl = 905; //dimension of overlaping functions
__constant__ int s_size_d; //number of components: it has to be set later at run time.
__constant__ float lb_d;//lower bound: it has to be set later at run time.
__constant__ float ub_d;//upper bound: it has to be set later at run time.

//Custom structs
struct row {double col[100];}; 
struct ind {double x[dim]; double fx; int FEs_when_found;};  

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
/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*****************************PROBLEM*********************************************/
#include "Benchmark.cu"//includes all basic functions needed to run any of the problems

//Select a problem at compilation time
#if f1
# include "f1.cu"
const int ID = 1;

#elif f2
# include "f2.cu"
const int ID = 2;

#elif f3
# include "f3.cu"
const int ID = 3;

#elif f4
# include "f4.cu"
const int ID = 4;

#elif f5
# include "f5.cu"
const int ID = 5;

#elif f6
# include "f6.cu"
const int ID = 6;

#elif f7
# include "f7.cu"
const int ID = 7;

#elif f8
# include "f8.cu"
const int ID = 8;

#elif f9
# include "f9.cu"
const int ID = 9;

#elif f10
# include "f10.cu"
const int ID = 10;

#elif f11
# include "f11.cu"
const int ID = 11;

#elif f12
# include "f12.cu"
const int ID = 12;

#elif f13
# include "f13.cu"
const int ID = 13;

#elif f14
# include "f14.cu"
const int ID = 14;

#elif f15
# include "f15.cu"
const int ID = 15;

#endif
/******************************************************************************/
/////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*****************************SET DATA ACCORDINGLY*******************************/
//Important note!!!: all reading functions are implemented in Benchmark.cu

void set_bounds(int id,float &lb,float &ub)
{
	if (id==1 || id==4 || id==7 || id==8 || id==12 || id==13 || id==14 || id==15 || id==9)
	{
		lb = -100;
		ub = 100;
	}
	else if (id==2 || id==5 || id==10)
	{
		lb = -5;
		ub = 5;
	}
	else //3,6,11
	{
		lb = -32;
		ub = 32;
	}

	return;
}

void set_objective_function(float &lb,float &ub)
{
	int i; //just an index

	//Setting bounds and number of subcomponents accordingly on host
	set_bounds(ID,lb,ub);
	if(ID <= 7) 
		s_size = 7; 
	else 
		s_size = 20;

	//Setting bounds and number of subcomponents accordingly on device
	cudaMemcpyToSymbol(s_size_d, &s_size, sizeof(s_size_d));
    cudaMemcpyToSymbol(lb_d, &lb, sizeof(lb_d));
    cudaMemcpyToSymbol(ub_d, &ub, sizeof(ub_d));

	//Allocate extra storage to work on device
	cudaMalloc(&mem_D,N_blocks*sizeof(storage));//stored on global memory

	/*
	Read data, allocate space on device and load data from host to device.
	*/

	//Set Pvector,r25,r50,r100,s,w
	if ((ID>=4 && ID<=11) or ID==13 or ID==14)
	{
		//Allocate memory space on device
		cudaMalloc(&Pvector_D,dim*sizeof(int));
		cudaMalloc(&r25_D,25*25*sizeof(double));
		cudaMalloc(&r50_D,50*50*sizeof(double));
		cudaMalloc(&r100_D,100*100*sizeof(double));
		cudaMalloc(&s_D,s_size*sizeof(int));
		cudaMalloc(&w_D,s_size*sizeof(double));

		//Read data and load it to device asynchronously
		cudaMallocHost(&Pvector,dim*sizeof(int));
		readPermVector(Pvector,dim,ID);
		cudaMemcpyAsync(Pvector_D,Pvector,dim*sizeof(int),cudaMemcpyDefault);

		cudaMallocHost(&r25,25*25*sizeof(double));
		readR(r25,25,ID);
		cudaMemcpyAsync(r25_D,r25,25*25*sizeof(double),cudaMemcpyDefault);

		cudaMallocHost(&r50,50*50*sizeof(double));
		readR(r50,50,ID);
		cudaMemcpyAsync(r50_D,r50,50*50*sizeof(double),cudaMemcpyDefault);

		cudaMallocHost(&r100,100*100*sizeof(double));
		readR(r100,100,ID);
		cudaMemcpyAsync(r100_D,r100,100*100*sizeof(double),cudaMemcpyDefault);

		cudaMallocHost(&s,s_size*sizeof(int));
		readS(s,s_size,ID);
		cudaMemcpyAsync(s_D,s,s_size*sizeof(int),cudaMemcpyDefault);

		cudaMallocHost(&w,s_size*sizeof(double));
		readW(w,s_size,ID);
		cudaMemcpy(w_D,w,s_size*sizeof(double),cudaMemcpyDefault);		
	}

	//Set Ovector and OvectorVec properly
	if (ID != 14)
	{
		//Read data
		Ovector = readOvector(dim,ID);

		//Allocate memory space on device
		cudaMalloc(&Ovector_D,dim*sizeof(double));

		//Load data to device
		cudaMemcpy(Ovector_D,Ovector,dim*sizeof(double),cudaMemcpyDefault);
	}
	else
	{
		//Read data
		OvectorVec = readOvectorVec(ID);

		//Allocate memory space on device
		cudaMalloc(&OvectorVec_D,s_size*sizeof(row));

		//Load data to device
		cudaMemcpy (OvectorVec_D, OvectorVec, s_size*sizeof(row), cudaMemcpyDefault);
		for (i = 0; i < s_size; ++i) cudaMemcpy(OvectorVec_D[i].col, OvectorVec[i],s[i]*sizeof(double), cudaMemcpyDefault);
	}
}

void free_objective_function_data()
{
	int i;
	cudaFree(mem_D);

	if ((ID>=4 && ID<=11) || ID==13 || ID==14)
	{
		//Free device
		cudaFree(Pvector_D);
		cudaFree(r25_D);
		cudaFree(r50_D);
		cudaFree(r100_D);
		cudaFree(s_D);
		cudaFree(w_D);
		//Free host
		cudaFreeHost(Pvector);
		cudaFreeHost(r25);
		cudaFreeHost(r50);
		cudaFreeHost(r100);
		cudaFreeHost(w);
		cudaFreeHost(s);
	}

	if (ID != 14) 
		{cudaFree(Ovector_D); free(Ovector);}
	else
	{
		cudaFree(OvectorVec_D); 
		for (i = 0; i < s_size; i++) free(OvectorVec[i]); 
		free(OvectorVec);
	}
}

int VerifyBestSolution(double *solution,float lb,float ub)
{
  int ans = 1;
  int j;
  for (j = 0; j < dim; ++j)
  {
    if(solution[j]<lb || solution[j]>ub)
    {
      ans = 0;
      break;
    }
  }
  return ans;
}
