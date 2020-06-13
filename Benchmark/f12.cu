__global__ void F_D(double *Ovector,storage *mem,int *Pvector,double *r25,
double *r50,double *r100,int *s,double *w,row *OvectorVec,ind *pop,int popsize)
{ 
  __shared__ double cache[N_threads];
  double acc,reg,oz,tmp;
  int i = blockIdx.x;
  int j = threadIdx.x;

  while (i < popsize) 
  {
    //////////////////////////Process an individual/////////////////////////////////////
    j = threadIdx.x;
    //Compute anotherz
    while(j<dim)
    {
      mem[blockIdx.x].anotherz[j] = pop[i].x[j] - Ovector[j];
      j += blockDim.x;
    }
    __syncthreads();

    //Save anotherz[j+1] in anotherz1[j]
    j = threadIdx.x;
    while(j<(dim-1))
    {
      mem[blockIdx.x].anotherz1[j] = mem[blockIdx.x].anotherz[j+1];
      j += blockDim.x;
    }
    __syncthreads();

    //Compute rosenbrock
    acc = 0.0;
    j = threadIdx.x;
    while(j<(dim-1))
    {
      reg = mem[blockIdx.x].anotherz[j];
      oz = mem[blockIdx.x].anotherz1[j];
      tmp = 100*( (reg*reg) - oz )*( (reg*reg) - oz );
      acc += ( tmp + ((reg-1)*(reg-1)) );
      j += blockDim.x;
    }
    cache[threadIdx.x] = acc;
    __syncthreads();

    //Reduction step
    reduce(cache);
    ////////////////////////////////////////////////////////////////////
    if(threadIdx.x==0) pop[i].fx = cache[0];
    __syncthreads();
    i += gridDim.x;
  }
}

//OMP
double F_H(double *x)
{
  int j;
  double sum,tmp;
  double reg,oz;
  double anotherz[dim];
  double anotherz1[dim];

  //Compute anotherz
  #pragma omp parallel for num_threads(maxThreads)
  for (j=0; j<dim; ++j) anotherz[j] = x[j] - Ovector[j];

  //Save anotherz[j+1] in anotherz1[j]
  #pragma omp parallel for num_threads(maxThreads)
  for (j=0; j<(dim-1); ++j) anotherz1[j] = anotherz[j+1];

  //rosenbrock
  sum = 0.0;
  #pragma omp parallel for reduction(+:sum) private(reg,oz,tmp) num_threads(maxThreads)
  for (j=0; j<(dim-1); ++j)
  {
    reg = anotherz[j];
    oz = anotherz1[j];
    tmp = 100*( (reg*reg) - oz )*( (reg*reg) - oz );
    tmp = tmp + ((reg-1)*(reg-1));
    sum += tmp;
  }
  return sum;
}

