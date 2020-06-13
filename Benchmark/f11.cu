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
  double s1,s2;
  int l;
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

      /*For r50 and r100 a pagination procedure is executed due to they are heavy data. A page
      is held in cache. Inexact means that: for example (size*size)/pagesize is not an integer.
      size=50, pagesize = 1000 for R50 and size=100, pagesize=1500 for R100.
      The idea of all this mess is to speed up the multiplication part.*/
      if (s[t]==25) multiply_normal(Z, cache3,s[t],threadIdx.x,Y);
      else if(s[t]==50) multiply_byPagination_inexactR50(Z, r50,s[t],threadIdx.x,Y,cache4,1000,mempointer,rowcount);
      else multiply_byPagination_inexactR100(Z, r100,s[t],threadIdx.x,Y,cache5,1500,mempointer,rowcount);

      if(threadIdx.x==0) c=c+s[t];
      __syncthreads();
      //////////////////////schwefel////////////////////////////
      j = threadIdx.x;
      while(j<s[t])
      { 
        Y[j] = sign(Y[j]) * exp( hat(Y[j]) + 0.049 * ( sin( c1(Y[j]) * hat(Y[j]) ) + sin( c2(Y[j])* hat(Y[j]) )  ) ) ;//transform_osz
        if(Y[j]>0) Y[j] = pow(Y[j], 1 + 0.2 * j/((double) (s[t]-1)) * sqrt(Y[j]));//transform_asy
        j += blockDim.x;
      }
      __syncthreads();

      // reduction step
      if(threadIdx.x == 0)
      {
        s1 = s2 = 0.0;
        for (l=0; l<s[t]; ++l)
        {
          s1 += Y[l];
          s2 += (s1 * s1);
        }
        cache[0] = s2;
      }
      __syncthreads();
      //////////////////////////////////////////////////////
      if(threadIdx.x==0){result += (w[t] * cache[0]); t += 1;}
      __syncthreads();
    }
    //////////////////////////////////////////////////////////////
    if(threadIdx.x==0) pop[i].fx = result;
    i += gridDim.x;
  }
}

//OMP 
double F_H(double *x)
{
  double anotherz[dim];
  double Z[100];
  double Y[100];
  int j,t,c,l;
  double s1,s2,reg,result;

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

    /*****************Schwefel***********************/
    #pragma omp parallel for num_threads(maxThreads)
    for (j=0; j<s[t]; ++j)
    {
      Y[j] = sign_h(Y[j]) * exp( hat_h(Y[j]) + 0.049 * ( sin( c1_h(Y[j]) * hat_h(Y[j]) ) + sin( c2_h(Y[j])* hat_h(Y[j]) )  ) ) ;//transform_osz
      if(Y[j]>0) Y[j] = pow(Y[j], 1 + 0.2 * j/((double) (s[t]-1)) * sqrt(Y[j]));//transform_asy
    }
    //Reduction step
    s1 = s2 = 0.0;
    for (l=0; l<s[t]; ++l)
    {
      s1 += Y[l];
      s2 += (s1 * s1);
    }
    result += (w[t]*s2); 
  }

  return result;
}

