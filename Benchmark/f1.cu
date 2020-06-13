__global__ void F_D(double *Ovector,storage *mem,int *Pvector,double *r25,
double *r50,double *r100,int *s,double *w,row *OvectorVec,ind *pop,int popsize)
{ 
  __shared__ double cache[N_threads];
  double acc,reg;
  int i = blockIdx.x;
  int j = threadIdx.x;

  while (i<popsize) 
  {
    //////////////////////////Process an individual/////////////////////////////////////
    acc = 0.0;
    j = threadIdx.x;
    //Compute elliptic
    while(j<dim)
    { 
      reg = pop[i].x[j] - Ovector[j];//load the difference to a register
      reg = sign(reg) * exp( hat(reg) + 0.049 * ( sin( c1(reg) * hat(reg) ) + sin( c2(reg)* hat(reg) )  ) ) ;//transform_osz
      acc += ( pow(1.0e6,  j/((double)(dim - 1)) ) * reg * reg );//accumulate
      j += blockDim.x;
    }
    cache[threadIdx.x] = acc;//save partial result to cache
    __syncthreads();

    // reduction step
    // At the end: cache[0] = cache[0]+cache[1]+cache[2]+.....+cache[NT]
    reduce(cache);//[NT stands for Number of threads]
    ////////////////////////////////////////////////////////////////////
    if(threadIdx.x==0) pop[i].fx = cache[0];//set fitness
    __syncthreads();
    i += gridDim.x;
  }
}


//OMP
double F_H(double *x)
{
  int j;
  double sum;
  double reg;

  //elliptic
  sum = 0.0;
  #pragma omp parallel for reduction(+:sum) private(reg) num_threads(maxThreads)
  for (j=0; j<dim; ++j)
  {
    reg = x[j] - Ovector[j];
    reg = sign_h(reg) * exp( hat_h(reg) + 0.049 * ( sin( c1_h(reg) * hat_h(reg) ) + sin( c2_h(reg)* hat_h(reg) )  ) ) ;//transform_osz
    reg = ( pow(1.0e6,  j/((double)(dim - 1)) ) * reg * reg );
    sum += reg;
  }
  return sum;
}




