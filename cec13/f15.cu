__global__ void F_D(double *Ovector,storage *mem,int *Pvector,double *r25,
double *r50,double *r100,int *s,double *w,row *OvectorVec,ind *pop,int popsize)
{ 
  __shared__ double cache[1000];
  double s1,s2;
  int l;
  int i = blockIdx.x;
  int j = threadIdx.x;

  while (i < popsize) 
  {
    //////////////////////////Process an individual/////////////////////////////////////
    //////////////////////schwefel////////////////////////////
    j = threadIdx.x;
    while(j<dim)
    {
      cache[j] =  pop[i].x[j] - Ovector[j];
      cache[j] = sign(cache[j]) * exp( hat(cache[j]) + 0.049 * ( sin( c1(cache[j]) * hat(cache[j]) ) + sin( c2(cache[j])* hat(cache[j]) )  ) ) ;//transform_osz
      if(cache[j]>0) cache[j] = pow(cache[j], 1 + 0.2 * j/((double) (dim-1)) * sqrt(cache[j]));//transform_asy
      j += blockDim.x;
    }
    __syncthreads();

    // reduction step
    if(threadIdx.x == 0)
    {
      s1 = s2 = 0.0;
      for (l=0; l<dim; ++l)
      {
        s1 += cache[l];
        s2 += (s1 * s1);
      }
    }
    __syncthreads();
    //////////////////////////////////////////////////////////////
    if(threadIdx.x==0) pop[i].fx = s2;
    i += gridDim.x;
  }
}


//OMP 
double F_H(double *x)
{
  int j,l;
  double s1,s2;
  double Y[dim];

  /*****************Schwefel***********************/
  #pragma omp parallel for num_threads(maxThreads)
  for (j=0; j<dim; ++j)
  {
    Y[j] = x[j] - Ovector[j];
    Y[j] = sign_h(Y[j]) * exp( hat_h(Y[j]) + 0.049 * ( sin( c1_h(Y[j]) * hat_h(Y[j]) ) + sin( c2_h(Y[j])* hat_h(Y[j]) )  ) ) ;//transform_osz
    if(Y[j]>0) Y[j] = pow(Y[j], 1 + 0.2 * j/((double) (dim-1)) * sqrt(Y[j]));//transform_asy
  }
  //Reduction step
  s1 = s2 = 0.0;
  for (l=0; l<dim; ++l)
  {
    s1 += Y[l];
    s2 += (s1 * s1);
  }

  return s2;
}


