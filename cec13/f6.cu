__global__ void F_D(double *Ovector,storage *mem,int *Pvector,double *r25,
double *r50,double *r100,int *s,double *w,row *OvectorVec,ind *pop,int popsize)
{ 
  __shared__ double cache[N_threads];
  __shared__ int c,t,mempointer,rowcount;
  __shared__ double result;
  __shared__ double Z[100];
  __shared__ double Y[100];
  __shared__ double cache3[625];
  __shared__ double cache4[1000];
  __shared__ double cache5[1500];
  double acc,reg,acc2;
  int i = blockIdx.x;
  int j = threadIdx.x;

  //Download r25 to cache
  while(j<625)
  {
    cache3[j] = r25[j];
    j+=blockDim.x;
  }
  __syncthreads();

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
    if(threadIdx.x == 0){c=0; t=0; result=0.0;}
    __syncthreads();

    // s_size non-separable part with rotation
    /*************************************************/
    while(t<s_size_d)
    {
      ////////////////////////Rotate vector/////////////////////////////////
      j = threadIdx.x;
      while(j<(c+s[t]))
      {
        if(j>=c) Z[j-c] = mem[blockIdx.x].anotherz[Pvector[j]];
        j += blockDim.x;
      }
      __syncthreads();

      /*For r50 and r100 pagination procedure is followed. A page is held in cache*/
      if (s[t]==25) multiplyR25_normal(Z, cache3,s[t],threadIdx.x,Y);
      else if(s[t]==50) multiplyR50_pagination(Z, r50,s[t],threadIdx.x,Y,cache4,1000,mempointer,rowcount);
      else multiplyR100_pagination(Z, r100,s[t],threadIdx.x,Y,cache5,1500,mempointer,rowcount);

      if(threadIdx.x==0) c=c+s[t];
      __syncthreads();
      //////////////////////ackley////////////////////////////
      acc = 0.0;
      acc2 = 0.0;
      j = threadIdx.x;
      while(j<s[t])
      { //anotherz[j]
        reg = Y[j];
        reg = sign(reg) * exp( hat(reg) + 0.049 * ( sin( c1(reg) * hat(reg) ) + sin( c2(reg)* hat(reg) )  ) ) ;//transform_osz
        if(reg>0) reg = pow(reg, 1 + 0.2 * j/((double) (s[t]-1)) * sqrt(reg));//transform_asy
        reg = reg * pow(10, 0.5 * j/((double) (s[t]-1)) ); //lambda
        acc += ( reg*reg ); //accumulate1
        acc2 += ( cos(2.0 * PI * reg) );//accumulate2
        j += blockDim.x;
      }
      cache[threadIdx.x] = acc;
      cache5[threadIdx.x] = acc2;
      __syncthreads();

      // reduction step
      reduce_twice(cache,cache5);
      if (threadIdx.x == 0) 
        cache[0] = -20.0 * exp(-0.2 * sqrt(cache[0] / s[t])) - exp(cache5[0] / s[t]) + 20.0 + E;
      __syncthreads();
      //////////////////////////////////////////////////////
      if(threadIdx.x==0){result += (w[t] * cache[0]); t += 1;}
      __syncthreads();
    }

    // one separable part without rotation
    // Compute ackley
    /*************************************************/
    acc = 0.0;
    j = threadIdx.x;
    while(j<dim)
    {
      if(j>=c) 
      {
        reg = mem[blockIdx.x].anotherz[Pvector[j]];
        reg = sign(reg) * exp( hat(reg) + 0.049 * ( sin( c1(reg) * hat(reg) ) + sin( c2(reg)* hat(reg) )  ) ) ;//transform_osz
        if(reg>0) reg = pow(reg, 1 + 0.2 * (j-c)/((double) (dim-c-1)) * sqrt(reg));//transform_asy
        reg = reg * pow(10, 0.5 * (j-c)/((double) (dim-c-1)) ); //lambda
        acc += ( reg*reg ); //accumulate1
        acc2 += ( cos(2.0 * PI * reg) );//accumulate2
      }
      j += blockDim.x;
    }

    cache[threadIdx.x] = acc;
    cache5[threadIdx.x] = acc2;
    __syncthreads();

    //Reduce
    reduce_twice(cache,cache5);
    if (threadIdx.x == 0) 
      cache[0] = -20.0 * exp(-0.2 * sqrt(cache[0] / (dim-c))) - exp(cache5[0] / (dim-c)) + 20.0 + E;
    __syncthreads();
    /*************************************************/
    if(threadIdx.x==0)
    {
      result += cache[0];
      cache[0] = result;
    }
    __syncthreads();

    //////////////////////////////////////////////////////////////
    if(threadIdx.x==0) pop[i].fx = cache[0];
    __syncthreads();
    i += gridDim.x;
  }
}

//OMP 
double F_H(double *x)
{
  double anotherz[dim];
  double Z[100];
  double Y[100];
  int j,t,c;
  double reg,result,reg1,reg2,sum1,sum2;

  //Compute anotherz
  #pragma omp parallel for num_threads(maxThreads)
  for (j=0; j<dim; ++j) anotherz[j] = x[j] - Ovector[j];

  c = 0;
  result = 0.0;
  for (t=0; t<s_size; ++t)
  {
    /*****************Rotate vector***********************/
    #pragma omp parallel for num_threads(maxThreads)
    for (j=c; j<(c+s[t]); ++j) Z[j-c] = anotherz[Pvector[j]];

    if (s[t]==25) multiply_h(Z, r25,s[t],Y);
    else if(s[t]==50) multiply_h(Z, r50,s[t],Y);
    else multiply_h(Z,r100,s[t],Y);
    c = c + s[t];

    /*****************Ackley***********************/
    sum1 = sum2 = 0.0;
    #pragma omp parallel for reduction(+:sum1,sum2) private(reg,reg1,reg2) num_threads(maxThreads)
    for (j=0; j<s[t]; ++j)
    {
      reg = Y[j];
      reg = sign_h(reg) * exp( hat_h(reg) + 0.049 * ( sin( c1_h(reg) * hat_h(reg) ) + sin( c2_h(reg)* hat_h(reg) )  ) ) ;//transform_osz
      if(reg>0) reg = pow(reg, 1 + 0.2 * j/((double) (s[t]-1)) * sqrt(reg));//transform_asy
      reg = reg * pow(10, 0.5 * j/((double) (s[t]-1)) );//lambda
      reg1 = ( reg*reg );
      reg2 = ( cos(2.0 * PI * reg) );
      sum1 += reg1;
      sum2 += reg2;
    }
    sum1 = -20.0 * exp(-0.2 * sqrt(sum1/s[t])) - exp(sum2/s[t]) + 20.0 + E;
    result += (w[t]*sum1); 
  }

  // one separable part without rotation
  // Compute ackley
  /*************************************************/
  sum1 = sum2 = 0.0;
  #pragma omp parallel for reduction(+:sum1,sum2) private(reg,reg1,reg2) num_threads(maxThreads)
  for (j=c; j<dim; ++j)
  {
    reg = anotherz[Pvector[j]];
    reg = sign_h(reg) * exp( hat_h(reg) + 0.049 * ( sin( c1_h(reg) * hat_h(reg) ) + sin( c2_h(reg)* hat_h(reg) )  ) ) ;//transform_osz
    if(reg>0) reg = pow(reg, 1 + 0.2 * (j-c)/((double) (dim-c-1)) * sqrt(reg));//transform_asy
    reg = reg * pow(10, 0.5 * (j-c)/((double) (dim-c-1)) );//lambda
    reg1 = ( reg*reg );
    reg2 = ( cos(2.0 * PI * reg) );
    sum1 += reg1;
    sum2 += reg2;
  }
  sum1 = -20.0 * exp(-0.2 * sqrt(sum1/(dim-c))) - exp(sum2/(dim-c)) + 20.0 + E;
  result += sum1;

  return result;
}

