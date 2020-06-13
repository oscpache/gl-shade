/*
Tanabe, R.; Fukunaga, A., "Success-history based parameter adaptation
for Differential Evolution," Evolutionary Computation (CEC), 2013 IEEE
Congress on , vol., no., pp.71,78, 20-23 June 2013
doi:10.1109/CEC.2013.655755510.1109/CEC.2013.6557555

Wan-li Xiang, Xue-lei Meng, Mei-qing An, Yinzhen Li, and Ming-xia Gao.
An enhanced diffe- rential evolution algorithm based on multiple mutation 
strategies. 
Computational Intelligence and Neuroscience, 2015:1–15, 11 2015.
*/
/***************** eshade_ls class *********************/
class eshade_ls
{	// Private stuff 
	int H_maxsize;
	int k;
	int counter;
	float p_min;
	float p_max;
	double *M_Cr;
	double *M_F;
	vector<ind> A;
	vector<ind> A_tmp;
	vector<ind> memory;
	evol_data_struct2 *evol_data;
	rank_ind *ranklist;
    ind *child;
	double *S_F;
	double *S_Cr;
	double *W;
	int S_size;
	double uF,uCr;
	int r,L;
    double mean[2];
    double *mu;
	//Device data//
	evol_data_struct2 *evol_data_D;
	rank_ind *rank_D;
	ind *memory_D;
	ind *child_D;
	double *S_F_D;
	double *S_Cr_D;
	double *W_D;
	double *mean_D;
	ind *child_mu_D;
public:
	ind *pop;
	ind *pop_D;
	int i,j;
	int best_location;
	float lb,ub;
	int NP,D;
	unsigned maxFEs,maxFEs_per_round;
	void set_parameters(int,int,unsigned,unsigned,float,float);
	void init_population_in_device(curandState*,unsigned&);
	void update_best();
	void free_memory();
	void evolve_in_device(unsigned&,default_random_engine&,ind&);
	void apply_A_maintenance();
	double mean_WA();
	double mean_WL();
	void receive(ind&);
	void ls_search(unsigned&,int,ind&);
	void sort(rank_ind*);
	
};

void eshade_ls::set_parameters(int popsize,int dimension,unsigned maxEvals,unsigned FEs_per_application,
float lowerbound,float upperbound)
{
	D = dimension;
	NP = popsize;
	maxFEs = maxEvals;
	maxFEs_per_round = FEs_per_application;
	lb = lowerbound;
	ub = upperbound;
	H_maxsize = 100;//NP
	best_location = 0;
	k = 0;
	counter = 0;
	p_min = 2.0/NP;
	p_max = 0.1; 
	pop = (ind *)malloc(NP*sizeof(ind));
	ranklist = (rank_ind *)malloc(NP*sizeof(rank_ind));
	evol_data = (evol_data_struct2 *)malloc(NP*sizeof(evol_data_struct2));
	child = (ind *)malloc(NP*sizeof(ind));
	S_F = (double *)malloc(NP*sizeof(double));
	S_Cr = (double *)malloc(NP*sizeof(double));
	W = (double *)malloc(NP*sizeof(double));
	M_F = (double *)malloc(H_maxsize*sizeof(double));
	M_Cr = (double *)malloc(H_maxsize*sizeof(double));
	mu = (double *)malloc(D*sizeof(double));
}

void eshade_ls::init_population_in_device(curandState *state_D,unsigned &current_FEs)
{
	// 1. allocate memory 
	cudaMalloc(&pop_D,NP*sizeof(ind));

	// 3. Running kernel
	init_population<<<N_blocks,N_threads>>>(state_D,pop_D,current_FEs,NP);
	current_FEs += NP;
	F_D<<<N_blocks,N_threads>>>(Ovector_D,mem_D,Pvector_D,r25_D,r50_D,r100_D,s_D,w_D,OvectorVec_D,pop_D,NP);

  	// Concurrently init Cr and F storage 
	for (i = 0; i < H_maxsize; ++i) {M_F[i] = 0.5; M_Cr[i] = 0.5;}

	//4. copy data from device to host
	cudaMemcpy(pop,pop_D,NP*sizeof(ind),cudaMemcpyDefault);

	//5. free memory
	cudaFree(pop_D);

	//Update best_location 
	update_best();
}

void eshade_ls::evolve_in_device(unsigned &current_FEs,default_random_engine &rng,ind &global_best)
{
	//Set counter and storage size counter
	counter = 0;
	S_size = 0;

    //Allocate memory on device
    cudaMalloc(&evol_data_D,NP*sizeof(evol_data_struct2));
    cudaMalloc(&pop_D,NP*sizeof(ind));
    cudaMalloc(&child_mu_D,one*sizeof(ind));
    cudaMalloc(&rank_D,NP*sizeof(rank_ind));
    cudaMalloc(&memory_D,2*NP*sizeof(ind));
    cudaMalloc(&child_D,NP*sizeof(ind));
    cudaMalloc(&S_F_D,NP*sizeof(double));
    cudaMalloc(&S_Cr_D,NP*sizeof(double));
    cudaMalloc(&W_D,NP*sizeof(double));
    cudaMalloc(&mean_D,2*sizeof(double));

    //While stopping condition is not met:
	while(counter<maxFEs_per_round && current_FEs<maxFEs)
	{
		/******************* SHADE ************************/
		// Join Population and external archive
		memory.insert(memory.end(), &pop[0], &pop[NP]); //memory = pop;
		memory.insert(memory.end(), A.begin(), A.end());

    	// Rank population by fitness
	    for (i = 0; i < NP; ++i)
	    {
	        ranklist[i].id = i; ranklist[i].fitness = pop[i].fx;

			/*******************Setting F and Cr************************/
			// Generate F and Cr using a normal distribution with mean
			// taken randomly from storage and std. 0.1
			r = rnd(0,H_maxsize-1);
			uF = M_F[r]; normal_distribution<double> Ndistribution_F(uF,0.1);
			uCr = M_Cr[r]; normal_distribution<double> Ndistribution_Cr(uCr,0.1);

	        evol_data[i].Cr = Ndistribution_Cr(rng); 
	        if (evol_data[i].Cr > 1.0) evol_data[i].Cr = 1.0; 
	        else if(evol_data[i].Cr < 0.0) evol_data[i].Cr = 0.0;

	        evol_data[i].F = Ndistribution_F(rng); 
	        if (evol_data[i].F > 1.0) evol_data[i].F = 1.0; 
	        while (evol_data[i].F <= 0.0) evol_data[i].F = Ndistribution_F(rng);

	        /*******************Setting p_best************************/
	        evol_data[i].p_best = rnd(0,int(rndreal(p_min,0.1)*NP)); // take an index within best pop range

	        /*******************Choosing a and b************************/
	        // randomly pick 2 different members
	        do evol_data[i].a = rnd(0,NP-1); while(evol_data[i].a==i); // from pop
	        do evol_data[i].b = rnd(0,memory.size()-1); while(evol_data[i].b==i || evol_data[i].b==evol_data[i].a); // from pop U archive

	        /*******************Get exp crossover window************************/
	        evol_data[i].Jrand = j = rnd(0,D-1);
	        L = 0;
	        do {evol_data[i].Jend = j; j = (j+1)%D; L++;} while(flip(evol_data[i].Cr) and L<D);
	    }
	    sort(ranklist);//sort by fitness min => ranklist[0].fitness

		//Load generated data and current population to device 
		cudaMemcpy(evol_data_D,evol_data,NP*sizeof(evol_data_struct2),cudaMemcpyDefault);
		cudaMemcpy(pop_D,pop,NP*sizeof(ind),cudaMemcpyDefault);
		cudaMemcpy(rank_D,ranklist,NP*sizeof(rank_ind),cudaMemcpyDefault);
		cudaMemcpy(memory_D,memory.data(),memory.size()*sizeof(ind),cudaMemcpyDefault);

		//Lauch kernel: mutation,recombination and function evaluation 
		eshade_ls_engine<<<N_blocks,N_threads>>>(evol_data_D,pop_D,rank_D,memory_D,child_D,NP);
		F_D<<<N_blocks,N_threads>>>(Ovector_D,mem_D,Pvector_D,r25_D,r50_D,r100_D,s_D,w_D,OvectorVec_D,child_D,NP);
		cudaMemcpy(child,child_D,NP*sizeof(ind),cudaMemcpyDefault);

		//Selection
		for (i = 0; i < NP; ++i)
		{
			//Update FEs counter 
			current_FEs += 1; counter += 1;
	        if (child[i].fx <= pop[i].fx) // if better than target vector then:
	        {	
	        	//if strictly better then:
	        	if (child[i].fx < pop[i].fx)
	        	{
	        		A_tmp.push_back(pop[i]);//add defeated parent to external archive
	        		S_F[S_size] = evol_data[i].F;//record F
	        		S_Cr[S_size] = evol_data[i].Cr;//record Cr
	        		W[S_size] = pop[i].fx - child[i].fx;//record improvement
	        		S_size++;//increase storage size counter
	        	}

	        	//update global_best if needed 
	        	if (child[i].fx<global_best.fx && current_FEs<=maxFEs)
	        	{
	        		global_best = child[i];
	        		global_best.FEs_when_found = current_FEs;
	        	}
	        	//Advance child to next generation
	        	pop[i] = child[i];
	        }
			if(current_FEs==1.2e5 || current_FEs==3e5 || current_FEs==6e5 || current_FEs==9e5 || current_FEs==1.2e6 || current_FEs==1.5e6
			|| current_FEs==1.8e6 || current_FEs==2.1e6 || current_FEs==2.4e6 || current_FEs==2.7e6 || current_FEs==3e6)
				fprintf(file_results,"%u,%d,%.2f,%.6e\n",current_FEs,ID,Rseed,global_best.fx);
		}
		//If F and Cr storages are non-empty
		if (S_size > 0)
		{	//Load F, Cr and W data to device  
			uF = double(S_size);
			cudaMemcpy(S_F_D,S_F,S_size*sizeof(double),cudaMemcpyDefault);
			cudaMemcpy(S_Cr_D,S_Cr,S_size*sizeof(double),cudaMemcpyDefault);
			cudaMemcpy(W_D,W,S_size*sizeof(double),cudaMemcpyDefault);
			cudaMemcpy(&mean_D[0],&uF,1*sizeof(double),cudaMemcpyDefault);
			//Compute mean WA and mean WL in device
			mean_WAWL<<<2,64>>>(S_Cr_D,S_F_D,W_D,mean_D);
		}

		//Concurrently update best solution index
		update_best();

		// Concurrently check external archive
		A.insert(A.end(), A_tmp.begin(), A_tmp.end()); // add defeated parents to A
		apply_A_maintenance();//|A| must be less than or equal to popsize

		// Update M_CR and M_F
		if (S_size > 0)
		{	//Record means 
			cudaMemcpy(mean,mean_D,2*sizeof(double),cudaMemcpyDefault);
			M_F[k] = mean[1];//mean_WL();
			M_Cr[k] = mean[0];//mean_WA();
			k = (k + 1) % H_maxsize;
		}

		// Reset and go again
		S_size = 0;
		A_tmp.clear();
		memory.clear();

		//Apply EDE_LS
		ls_search(current_FEs,D,global_best);
	}
    //Free memory
    cudaFree(evol_data_D);
    cudaFree(pop_D);
    cudaFree(child_mu_D);
    cudaFree(rank_D);
    cudaFree(memory_D);
    cudaFree(child_D);
    cudaFree(S_F_D);
    cudaFree(S_Cr_D);
    cudaFree(W_D);
    cudaFree(mean_D);

}

void eshade_ls::ls_search(unsigned &current_FEs,int maxFEs_LS_in_round,ind &global_best)
{
	int totalevals = 0;
    double score; // trial vector fitness
    float wmax = 0.2; float wmin = 0; float r2; int k0,l,n,j;
	int best = best_location;

	while (totalevals<maxFEs_LS_in_round && current_FEs<maxFEs)
	{
	    for (j = 0; j < D; ++j)
	    { 
	      	if (current_FEs>=maxFEs || totalevals>=maxFEs_LS_in_round) break;

	        // set mu = x_best
	        for (l = 0; l < D; ++l) mu[l] = pop[best].x[l];

	        do k0 = rnd(0,NP-1); while(k0==best);
	        do n = rnd(0,D-1); while(n==j);

	        // compute r2
	        r2 = wmin + ((current_FEs/maxFEs)*(wmax-wmin));

	        // perturb
	        if (flip(r2))
	            mu[j] = pop[best].x[n] + ((2*rndreal(0,1))-1)*(pop[best].x[n] - pop[k0].x[n]);
	        else
	            mu[j] = pop[best].x[j] + ((2*rndreal(0,1))-1)*(pop[best].x[n] - pop[k0].x[n]);

	        // making sure a gen isn't out of boundary
	        if (mu[j] > ub)
	            mu[j] = (ub+pop[best].x[j])/2;
	        else if (mu[j] < lb)
	            mu[j] = (lb+pop[best].x[j])/2;

	        // evaluate mu
	        score = F_H(mu);//fp->compute(mu);
	        current_FEs += 1; totalevals += 1; counter += 1;

	        // choose better{mu,current_best} as new best
	        if (score <= pop[best].fx)
	        {
		      	if (current_FEs<=maxFEs && score<global_best.fx)
		      	{
	        		global_best.FEs_when_found = current_FEs;
	        		global_best.fx = score;
	        		memcpy(global_best.x,mu,D*sizeof(double));
		      	}
	            for (l = 0; l < D; ++l) pop[best].x[l] = mu[l];
	            pop[best].fx = score;	        	
	        }
			if(current_FEs==1.2e5 || current_FEs==3e5 || current_FEs==6e5 || current_FEs==9e5 || current_FEs==1.2e6 || current_FEs==1.5e6
			|| current_FEs==1.8e6 || current_FEs==2.1e6 || current_FEs==2.4e6 || current_FEs==2.7e6 || current_FEs==3e6)
				fprintf(file_results,"%u,%d,%.2f,%.6e\n",current_FEs,ID,Rseed,global_best.fx);
	    }
	}
  return;
}


void eshade_ls::free_memory()
{
	free(pop);
	free(ranklist);
    free(evol_data);
    free(child);
    free(S_F);
    free(S_Cr);
    free(W);
    free(M_F);
    free(M_Cr);
    free(mu);
}

void eshade_ls::receive(ind &global_best)
{
	//Choose a random position to place it
	int rand; 
	do rand = rnd(0,NP-1); while(rand==best_location);

	// Place it at rand index position
	pop[rand] = global_best;
}

void eshade_ls::update_best()
{
  double min = pop[0].fx; int min_id = 0;

  for (i = 1; i < NP; ++i)
  {
    if (pop[i].fx < min)
    {
      min = pop[i].fx;
      min_id = i;
    }

  }
  best_location = min_id;
}


void eshade_ls::apply_A_maintenance()
{	
	int r;
	while (A.size() > NP)
	{
		r = rnd(0,A.size()-1);
		A.erase(A.begin()+r);
	}
}


double eshade_ls::mean_WA()
{	// W -> improvement
    int k,g;
    int size = S_size;
    double delta_sum = 0;
    double tmp = 0;
    
    for (g = 0; g < size; g++)
        delta_sum += W[g];
    
    for (k = 0; k < size; k++)
        tmp += (W[k]/delta_sum) * S_Cr[k];

    if (tmp > 1) tmp = 1;
    else if (tmp < 0) tmp = 0;
    
    return tmp;
}

double eshade_ls::mean_WL()
{	// W -> improvement
    int k,g;
    int size = S_size;
    double delta_sum = 0;
    double tmp1 = 0;
    double tmp2 = 0;
    double res;
    
    for (g = 0; g < size; g++)
        delta_sum += W[g];
    
    for (k = 0; k < size; k++)
        tmp1 += (W[k]/delta_sum) * (S_F[k]*S_F[k]);
    for (k = 0; k < size; k++)
        tmp2 += (W[k]/delta_sum) * S_F[k];
    
    res = tmp1/tmp2;
    if (res > 1) res = 1;
    else if (res < 0) res = 0;

    return res; 
}

void eshade_ls::sort(rank_ind *S)
{	/*
	Insertion sort
	If (NP<=4000) is a good idea to use insertion sort due to its simplicity,
	actually InsertionSort shows better performance in this affair.
	else it'll be required to implement another algorithm with better 
	time complexity.
	*/

	int l,m;
	rank_ind key;

	for (l = 1; l < NP; ++l)
	{
		key = S[l];
		//Insert S[l] 􏱾 into the sorted sequence S[1......l-1]
		m = l-1;
		while(m>=0 && S[m].fitness>key.fitness)
		{
			S[m+1] = S[m];
			m--;
		}
		S[m+1] = key;
	}
}


