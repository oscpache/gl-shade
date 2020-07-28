/***************** Randomize on GPU kernel *********************/
__global__ void randomize_gpu(int GPUseed,curandState *state)
{	// this function starts random number generator engine on gpu.
	int randState_tid = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(GPUseed,randState_tid,0,&state[randState_tid]);
}
/***************************************************************/

/***************** Init population kernel *********************/
__global__ void init_population(curandState *state,ind *pop,int NP)
{
	/*This function, randomly with uniform distribution, initializes a given population.
	An individual is processed by a block and threads within that block process that individual's
	variables.
	This means that we're adopting a grid which configuration of blocks is linear as well as
	the configuration of the threads within a block*/   
	int i = blockIdx.x; //block identifier (bid) 
	int j = threadIdx.x; // thread identifier (tid) within block 
	int randState_tid = blockIdx.x*blockDim.x + threadIdx.x; //rng state tid  
	curandState localState; 

	//For every individual do...
	while (i<NP) //as much individuals as number of blocks are processed by iteration 
	{
		/*************************** Init chromosome *****************************/
		//For every dimension do...
		j = threadIdx.x;
		while (j < dim) //as much variables as number of threads are processed by iteration
		{
	  		// Initialize individual i, variable j
			localState = state[randState_tid];// Copy state to local memory for efficiency 
			pop[i].x[j] = lb_d + curand_uniform(&localState)*(ub_d - lb_d);//curand_uniform gives a random number between (0.0,1.0]
			state [randState_tid] = localState;// Copy state back to global memory
			/* blockDim.x is the number of threads per block. When the threads of a block have a tabular (matrix) shape: blockDim.x is
			the number of rows and blockDim.y the number of columns but since here we're using a linear configuration, blockDim.x is
			the only that matters since blockDim.y is zero in this particular case*/ 
	  		j += blockDim.x; 
		}
		__syncthreads();
		/* gridDim.x is the number of blocks in the grid. When the blocks of a grid have a tabular (matrix) shape the total number
		of blocks is given for gridDim.x*gridDim.y similarly the number of threads within a block that have a matrix shape threads distribution
		is given by blockDim.x*blockDim.y */
		i += gridDim.x;
	}
}
/***************************************************************/

__global__ void shade_engine(curandState *state,evol_data_struct *dat,ind *pop,rank_ind *rank,ind *memory,ind *child,int NP)
{
	/*This function evolves population by applying mutation and recombination operators 
	according to shade algorithm. An individual is processed by a block*/
	int i = blockIdx.x; //block identifier (bid) 
	int j = threadIdx.x; // thread identifier (tid) within block
	int randState_tid = blockIdx.x*blockDim.x + threadIdx.x;
	curandState localState; 
	__shared__ double F,Cr; ////stored in shared memory 
	__shared__ int j_rand,p_best,a,b; ////stored in shared memory

	//For every individual do...	
	while (i<NP) 
	{
		//Let thread #0 to set evolution data 
		if (threadIdx.x == 0)
		{
			F = dat[i].F;
			Cr = dat[i].Cr;
			p_best = dat[i].p_best;
			a = dat[i].a;
			b = dat[i].b;
			j_rand = dat[i].j_rand;
		}
		__syncthreads(); //barrier: waiting until thread #0 finishes  
		/****************Mutation and binomial crossover both at once************************/
		//For every dimension do...
		j = threadIdx.x;
		while (j < dim)
		{
			localState = state[randState_tid]; // load current state 
			if (curand_uniform(&localState)<=Cr || j == j_rand)
			{	// mutate
		        child[i].x[j] = pop[i].x[j] + F*(pop[rank[p_best].id].x[j] - pop[i].x[j])
		        + F*(pop[a].x[j] - memory[b].x[j]);
		        // making sure isn't out of boundary
		        if (child[i].x[j] > ub_d)
		            child[i].x[j] = (ub_d+pop[i].x[j])/2;
		        else if (child[i].x[j] < lb_d)
		            child[i].x[j] = (lb_d+pop[i].x[j])/2;
			}
			else child[i].x[j] = pop[i].x[j];
			state [randState_tid] = localState; // update state 
		  	j += blockDim.x;
		}
		__syncthreads(); //waiting until all block threads finish to go over the next individual
		i += gridDim.x;
	}
}

/***************************************************************/
__global__ void eshade_ls_engine(evol_data_struct2 *dat,ind *pop,rank_ind *rank,ind *memory,ind *child,int NP)
{
	/*This function evolves population by applying mutation and recombination operators 
	according to eshade_ls algorithm. An individual is processed by a block*/
	int i = blockIdx.x; //block identifier (bid)
	int j = threadIdx.x; // thread identifier (tid) within block
	__shared__ double F; //stored in shared memory
	__shared__ int p_best,a,b,Jrand,Jend; //stored in shared memory

	//For every individual do...
	while (i<NP) 
	{
		//Let thread #0 to set evolution data
		if (threadIdx.x == 0)
		{
			F = dat[i].F;
			p_best = dat[i].p_best;
			a = dat[i].a;
			b = dat[i].b;
			Jrand = dat[i].Jrand;
			Jend = dat[i].Jend;
		}
		__syncthreads(); //barrier: waiting until thread #0 finishes
		/****************Mutation and exponential crossover both at once************************/
		//For every dimension do...
		j = threadIdx.x;
		while (j<dim)
		{
			if((Jend>=Jrand && j>=Jrand && j<=Jend) || (Jend<Jrand && j<=Jend) || (Jend<Jrand && j>=Jrand))
			{
				// mutate
		        child[i].x[j] = pop[rank[p_best].id].x[j] + F*(pop[a].x[j] - memory[b].x[j]);
		        // making sure isn't out of boundary
		        if (child[i].x[j] > ub_d)
		            child[i].x[j] = (ub_d+pop[i].x[j])/2;
		        else if (child[i].x[j] < lb_d)
		            child[i].x[j] = (lb_d+pop[i].x[j])/2;
		    }
			else 
				child[i].x[j] = pop[i].x[j];
		  	j += blockDim.x;
		}
		__syncthreads(); //waiting until all block threads finish to go over the next individual    
		i += gridDim.x;
	}
}

/***************************************************************/
__global__ void mean_WAWL(double *Scr,double *Sf,double *improvement,int size,double *mean)
{
	/*This function computes the weighted arithmetic mean (WA) over Scr and the 
	weighted Lehmer mean (WL) over Sf. This kernel is specifically designed to be 
	executed using a grid configuration (2,64): 2 blocks and 64 threads per block.
	Here block #0 computes WA and block #1 computes WL in such a way Scr and Sf 
	are processed concurrently*/
	int j = threadIdx.x;
	__shared__ double acc; //accumulator in shared memory 
	//we'll be working using shared memory space for faster computation
	__shared__ double cache1[64];
	__shared__ double cache2[64];
	int k; //just an index
	double sum1,sum2; //private accumulators 

	//block #0 
	if (blockIdx.x == 0)
	{	// WA mean
		////////////////////////Sum all improvements to get delta_sum////////////////////////////////
		sum1 = 0.0; //thread sets its private sum to zero
		//For every entry in improvement array do...
		while(j<size)
		{
			sum1 += improvement[j]; //accumulate in private sum
			j += blockDim.x; 
		}
		//thread stores its private sum to its corresponding space in cache
		cache1[threadIdx.x] = sum1;
		__syncthreads(); //waiting until all block threads finish before move on

		/*Perform a dichotomous reduction: cache1[0] = cache1[0] + cache1[1] + ... + cache1[64-1].
		Thread cooperation and coordination is needed*/
		k = blockDim.x/2;
		while (k != 0) 
		{
			if (threadIdx.x < k) cache1[threadIdx.x] += cache1[threadIdx.x + k];
			__syncthreads();
			k /= 2; 
		}
		//Let thread #0 to store cache1[0] (delta_sum) in acc
		if(threadIdx.x == 0) acc = cache1[0];
		__syncthreads(); //waiting until thread #0 finishes
	  	//////////////////Compute SUM(w_j*Scr_j)///////////////////////////
	  	j = threadIdx.x; //reset j 
	  	sum1 = 0.0; //reset private sum

	  	//For j in [0,...,size-1]
		while(j<size)
		{
			sum1 += (improvement[j]/acc)*Scr[j]; //compute and accumulate in private sum
			j += blockDim.x;
		}
		//thread stores its private sum to its corresponding space in cache
		cache1[threadIdx.x] = sum1; 
		__syncthreads(); //wait

		/*Perform a dichotomous reduction: cache1[0] = cache1[0] + cache1[1] + ... + cache1[64-1].
		Thread cooperation and coordination is needed*/
		k = blockDim.x/2;
		while (k != 0) 
		{
			if (threadIdx.x < k) cache1[threadIdx.x] += cache1[threadIdx.x + k];
			__syncthreads();
			k /= 2; 
		}
		__syncthreads();
		//Let thread #0 to store cache1[0] (WA mean) in mean[0].
		if (threadIdx.x == 0)
		{
			if(cache1[0]>1) cache1[0] = 1;
			else if(cache1[0]<0) cache1[0] = 0;
			mean[0] = cache1[0];
		}
		//no synchronization needed here since we're done
	}
	//block #1
	else
	{	//WL mean
		////////////////////////Sum all improvements to get delta_sum////////////////////////////////
		sum1 = 0.0; //thread sets its private sum to zero
		//For every entry in improvement array do...
		while(j<size) 
		{
			sum1 += improvement[j]; //accumulate in private sum
			j += blockDim.x;
		}
		//thread stores its private sum to its corresponding space in cache
		cache1[threadIdx.x] = sum1;
		__syncthreads();

		/*Perform a dichotomous reduction: cache1[0] = cache1[0] + cache1[1] + ... + cache1[64-1].
		Thread cooperation and coordination is needed*/
		k = blockDim.x/2;
		while (k != 0) 
		{
			if (threadIdx.x < k) cache1[threadIdx.x] += cache1[threadIdx.x + k];
			__syncthreads();
			k /= 2; 
		}
		//Let thread #0 to store cache1[0] (delta_sum) in acc
		if(threadIdx.x == 0) acc = cache1[0];
		__syncthreads();
	  	///////////////////// Compute SUM(w_j*[Sf_j]^2)/SUM(w_j*Sf_j) /////////////////////////////
	  	j = threadIdx.x; //reset j
	  	sum1 = sum2 = 0.0; //set both private accumulators to zero

	  	////For j in [0,...,size-1]
		while(j<size)
		{
			sum1 += (improvement[j]/acc)*(Sf[j]*Sf[j]); //compute and accumulate in private sum1
			sum2 += (improvement[j]/acc)*Sf[j]; //compute and accumulate in private sum2
			j += blockDim.x;
		}
		//thread stores its private sum1 and sum 2 to its corresponding space in cache 1 and cache2 respectively 
		cache1[threadIdx.x] = sum1;
		cache2[threadIdx.x] = sum2;
		__syncthreads(); //wait for all 

		// Perform a dichotomous reduction on both cache1 and cache2
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
		__syncthreads(); //wait for all 

		//Let thread #0 to store cache1[0]/cache2[0] (WL mean) in mean[1].
		if (threadIdx.x == 0)
		{
			acc = cache1[0]/cache2[0];
			if(acc>1) acc = 1.0;
			else if(acc<0) acc = 0.0;
			mean[1] = acc;
		}
	}
}
