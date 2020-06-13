
/***************** Randomize on GPU kernel *********************/
__global__ void randomize_gpu(int GPUseed,curandState *state)
{	// this function starts random number generator engine on gpu.
	int randState_tid = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(GPUseed,randState_tid,0,&state[randState_tid]);
}
/***************************************************************/

/***************** Init population kernel *********************/
__global__ void init_population(curandState *state,ind *pop,unsigned FEs_begin,int NP)
{
	int i = blockIdx.x;
	int j = threadIdx.x;
	int randState_tid = blockIdx.x*blockDim.x + threadIdx.x;
	curandState localState; 

	while (i<NP) 
	{
		/*************************** Init chromosome *****************************/
		// Init individual
		j = threadIdx.x;
		while (j < dim)
		{
	  		// Initialize individual i, variable j
			localState = state[randState_tid];// Copy state to local memory for efficiency 
			pop[i].x[j] = lb_d + curand_uniform(&localState)*(ub_d - lb_d);//curand_uniform gives a random number between (0.0,1.0]
			state [randState_tid] = localState;// Copy state back to global memory
	  		j += blockDim.x;
		}
		__syncthreads();
		i += gridDim.x;
	}
}
/***************************************************************/

__global__ void shade_engine(curandState *state,evol_data_struct *dat,
ind *pop,rank_ind *rank,ind *memory,ind *child,int NP)
{
	// A block will process an individual so that every thread will process a variable.
	int i = blockIdx.x;
	int j = threadIdx.x;
	int randState_tid = blockIdx.x*blockDim.x + threadIdx.x;
	curandState localState; 
	__shared__ double F,Cr;
	__shared__ int j_rand,p_best,a,b;

  while (i<NP) 
  {
  	if (threadIdx.x == 0)
  	{
  		F = dat[i].F;
  		Cr = dat[i].Cr;
  		p_best = dat[i].p_best;
  		a = dat[i].a;
  		b = dat[i].b;
  		j_rand = dat[i].j_rand;
  	}
  	__syncthreads();
  	/****************Mutation and Bin-Crossover************************/
  	j = threadIdx.x;
    while (j < dim)
    {
    	localState = state[randState_tid]; // load current state 
    	if (curand_uniform(&localState)<=Cr || j == j_rand)
    	{	// mutate gene
            child[i].x[j] = pop[i].x[j] + F*(pop[rank[p_best].id].x[j] - pop[i].x[j])
            + F*(pop[a].x[j] - memory[b].x[j]);
            // making sure a gen isn't out of boundary
            if (child[i].x[j] > ub_d)
                child[i].x[j] = (ub_d+pop[i].x[j])/2;
            else if (child[i].x[j] < lb_d)
                child[i].x[j] = (lb_d+pop[i].x[j])/2;
    	}
    	else child[i].x[j] = pop[i].x[j];
    	state [randState_tid] = localState; // update state 
      	j += blockDim.x;
    }
    __syncthreads();
    i += gridDim.x;
  }
}
/***************************************************************/


/***************************************************************/

__global__ void eshade_ls_engine(evol_data_struct2 *dat,ind *pop,rank_ind *rank,
ind *memory,ind *child,int NP)
{
	// A block will process an individual so that every thread will process a variable.
	int i = blockIdx.x;
	int j = threadIdx.x;
	__shared__ double F;
	__shared__ int p_best,a,b,Jrand,Jend;

  while (i<NP) 
  {
  	if (threadIdx.x == 0)
  	{
  		F = dat[i].F;
  		p_best = dat[i].p_best;
  		a = dat[i].a;
  		b = dat[i].b;
  		Jrand = dat[i].Jrand;
  		Jend = dat[i].Jend;
  	}
  	__syncthreads();
  	/****************Mutation and Exp-Crossover************************/
  	j = threadIdx.x;
    while (j<dim)
    {
    	if((Jend>=Jrand && j>=Jrand && j<=Jend) || (Jend<Jrand && j<=Jend) || (Jend<Jrand && j>=Jrand))
    	{
			// mutate gene
	        child[i].x[j] = pop[rank[p_best].id].x[j] + F*(pop[a].x[j] - memory[b].x[j]);
	        // making sure a gen isn't out of boundary
	        if (child[i].x[j] > ub_d)
	            child[i].x[j] = (ub_d+pop[i].x[j])/2;
	        else if (child[i].x[j] < lb_d)
	            child[i].x[j] = (lb_d+pop[i].x[j])/2;
	    }
    	else 
    		child[i].x[j] = pop[i].x[j];
      	j += blockDim.x;
    }
    __syncthreads();    
    i += gridDim.x;
  }
}
/***************************************************************/
__global__ void mean_WAWL(double *Scr,double *Sf,double *improvement,double *mean)
{	//2 blocks and 64 threads per block
	//int i = blockIdx.x;
	int j = threadIdx.x;
	__shared__ int size;
	__shared__ double acc;
	__shared__ double cache1[64];
	__shared__ double cache2[64];
	int k;
	double sum1,sum2;

	//Set storage length
	if(threadIdx.x == 0) size = int(mean[0]);
	__syncthreads();

	if (blockIdx.x == 0)
	{	//Mean WA
		///////////////////////////////////////////////////////////////////
		sum1 = 0.0;
		while(j<size)
		{
			sum1 += improvement[j];
			j += blockDim.x;
		}
		cache1[threadIdx.x] = sum1;
		__syncthreads();
		// reduction step
		k = blockDim.x/2;
		while (k != 0) 
		{
			if (threadIdx.x < k) cache1[threadIdx.x] += cache1[threadIdx.x + k];
			__syncthreads();
			k /= 2; 
		}
		if(threadIdx.x == 0) acc = cache1[0];
		__syncthreads();
	  	///////////////////////////////////////////////////////////////////
	  	j = threadIdx.x;
	  	sum1 = 0.0;
		while(j<size)
		{
			sum1 += (improvement[j]/acc)*Scr[j];
			j += blockDim.x;
		}
		cache1[threadIdx.x] = sum1;
		__syncthreads();
		// reduction step
		k = blockDim.x/2;
		while (k != 0) 
		{
			if (threadIdx.x < k) cache1[threadIdx.x] += cache1[threadIdx.x + k];
			__syncthreads();
			k /= 2; 
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			if(cache1[0]>1) cache1[0] = 1;
			else if(cache1[0]<0) cache1[0] = 0;
			mean[0] = cache1[0];
		}
		///////////////////////////////////////////////////////////////////
	}
	else
	{	//Mean WL
		///////////////////////////////////////////////////////////////////
		sum1 = 0.0;
		while(j<size)
		{
			sum1 += improvement[j];
			j += blockDim.x;
		}
		cache1[threadIdx.x] = sum1;
		__syncthreads();
		// reduction step
		k = blockDim.x/2;
		while (k != 0) 
		{
			if (threadIdx.x < k) cache1[threadIdx.x] += cache1[threadIdx.x + k];
			__syncthreads();
			k /= 2; 
		}
		if(threadIdx.x == 0) acc = cache1[0];
		__syncthreads();
	  	///////////////////////////////////////////////////////////////////
	  	j = threadIdx.x;
	  	sum1 = sum2 = 0.0;
		while(j<size)
		{
			sum1 += (improvement[j]/acc)*(Sf[j]*Sf[j]);
			sum2 += (improvement[j]/acc)*Sf[j];
			j += blockDim.x;
		}
		cache1[threadIdx.x] = sum1;
		cache2[threadIdx.x] = sum2;
		__syncthreads();
		// reduction step
		k = blockDim.x/2;
		while (k != 0) 
		{
			if (threadIdx.x < k) 
			{
				cache1[threadIdx.x] += cache1[threadIdx.x + k];
				cache2[threadIdx.x] += cache2[threadIdx.x + k];
			}
			__syncthreads();
			k /= 2; 
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			acc = cache1[0]/cache2[0];
			if(acc>1) acc = 1.0;
			else if(acc<0) acc = 0.0;
			mean[1] = acc;
		}
	}
}
