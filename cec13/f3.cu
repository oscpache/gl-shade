__global__ void F_D(double *Ovector,storage *mem,int *Pvector,double *r25,
double *r50,double *r100,int *s,double *w,row *OvectorVec,ind *pop,int popsize)
{ 
  __shared__ double cache[N_threads];
  __shared__ double cache5[N_threads];
  double acc,reg,acc2;
  int i = blockIdx.x;
  int j = threadIdx.x;

  while (i < popsize) 
  {
    //////////////////////////Process an individual/////////////////////////////////////
    acc = 0.0;
    acc2 = 0.0;
    j = threadIdx.x;
    //Compute ackley
    while(j<dim)
    { 
      reg = pop[i].x[j] - Ovector[j];//anotherz[j]
      reg = sign(reg) * exp( hat(reg) + 0.049 * ( sin( c1(reg) * hat(reg) ) + sin( c2(reg)* hat(reg) )  ) ) ;//transform_osz
      if(reg>0) reg = pow(reg, 1 + 0.2 * j/((double) (dim-1)) * sqrt(reg));//transform_asy
      reg = reg * pow(10, 0.5 * j/((double) (dim-1)) ); //lambda
      acc += ( reg*reg ); //accumulate1
      acc2 += ( cos(2.0 * PI * reg) );//accumulate2
      j += blockDim.x;
    }
    cache[threadIdx.x] = acc;
    cache5[threadIdx.x] = acc2;
    __syncthreads();

    // reduction step
    // At the end: cache[0] = cache[0]+cache[1]+cache[2]+.....+cache[NT]
    // At the end: cache5[0] = cache5[0]+cache5[1]+cache5[2]+.....+cache5[NT]
    reduce_twice(cache,cache5);//[NT stands for Number of threads]
    if (threadIdx.x == 0) 
      cache[0] = -20.0 * exp(-0.2 * sqrt(cache[0] / dim)) - exp(cache5[0] / dim) + 20.0 + E;
    __syncthreads();
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
  double sum1,sum2;
  double reg,reg1,reg2;

  //ackley
  sum1 = sum2 = 0.0;
  #pragma omp parallel for reduction(+:sum1,sum2) private(reg,reg1,reg2) num_threads(maxThreads)
  for (j=0; j<dim; ++j)
  {
    reg = x[j] - Ovector[j];
    reg = sign_h(reg) * exp( hat_h(reg) + 0.049 * ( sin( c1_h(reg) * hat_h(reg) ) + sin( c2_h(reg)* hat_h(reg) )  ) ) ;//transform_osz
    if(reg>0) reg = pow(reg, 1 + 0.2 * j/((double) (dim-1)) * sqrt(reg));//transform_asy
    reg = reg * pow(10, 0.5 * j/((double) (dim-1)) );//lambda
    reg1 = ( reg*reg );
    reg2 = ( cos(2.0 * PI * reg) );
    sum1 += reg1;
    sum2 += reg2;
  }
  sum1 = -20.0 * exp(-0.2 * sqrt(sum1/dim)) - exp(sum2/dim) + 20.0 + E;
  
  return sum1;
}

