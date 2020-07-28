/*
Tanabe, R.; Fukunaga, A., "Success-history based parameter adaptation
for Differential Evolution," Evolutionary Computation (CEC), 2013 IEEE
Congress on , vol., no., pp.71,78, 20-23 June 2013
doi:10.1109/CEC.2013.655755510.1109/CEC.2013.6557555
*/
/******************************************************************************/
/////////////////////////////SHADE//////////////////////////////////
/******************************************************************************/
class shade
{
private:
	//host
	int NP,D,H_maxsize,stop_criterion,k;
	float p_min,p_max;
	vector<ind> A;
	vector<ind> A_tmp;
	vector<ind> memory;
	evol_data_struct *evol_data;
	rank_ind *ranklist;
	ind *child;
	double *S_F;
	double *S_Cr;
	double *W;
	int S_size;
	double uF,uCr;
	double mean[2];

	//device
	evol_data_struct *evol_data_D;
	rank_ind *rank_D;
	ind *pop_D;
	ind *memory_D;
	ind *child_D;
	double *S_F_D;
	double *S_Cr_D;
	double *W_D;
	double *mean_D; 

	void update_best()
	{
		int i;
		double min = pop[0].fx; int min_id = 0;

		for (i = 1; i < NP; ++i)
		{
			if (pop[i].fx < min)
	    	{
	      		min = pop[i].fx;
	      		min_id = i;
	    	}

	  	}
	  	best = min_id;
	}

	void apply_A_maintenance()
	{	
		int r;
		while (A.size() > NP)
		{
			r = rnd(0,A.size()-1);
			A.erase(A.begin()+r);
		}
	}

	void sort(rank_ind *S)
	{	/*Insertion sort*/
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

public:
	int best; //position where the best individual is
	ind *pop; //ind data type is defined in in objective_function.h 
	double *M_Cr;
	double *M_F;

	shade(float lb, float ub, int popsize=100, int memory_size=100, int N_decison_var=1000,int shade_stop_criterion=25000)
	{	
		//__init__
		/* lb and ub are no needed while executing shade on host since
		mutation and recombination is done on device */
		NP = popsize; 
		D = N_decison_var;
		stop_criterion = shade_stop_criterion;
		H_maxsize = memory_size;
		k = 0;
		p_min = 2.0/NP;
		p_max = 0.2;
		pop = (ind *)malloc(NP*sizeof(ind));
		ranklist = (rank_ind *)malloc(NP*sizeof(rank_ind));
		evol_data = (evol_data_struct *)malloc(NP*sizeof(evol_data_struct));
		child = (ind *)malloc(NP*sizeof(ind));
		S_F = (double *)malloc(NP*sizeof(double));
		S_Cr = (double *)malloc(NP*sizeof(double));
		W = (double *)malloc(NP*sizeof(double));
		M_F = (double *)malloc(H_maxsize*sizeof(double));
		M_Cr = (double *)malloc(H_maxsize*sizeof(double));
	}

	void init_population_in_device(curandState *state_D,ind &global_best,int &glshade_current_FEs)
	{
		int i;

		// 1. allocate memory 
		cudaMalloc(&pop_D,NP*sizeof(ind));

		//3. Initializing population on device while updating current evaluations on host 
		init_population<<<N_blocks,N_threads>>>(state_D,pop_D,NP);
		glshade_current_FEs += NP;

		//4. Evaluating the created population on device while initializing M_F and M_Cr on host 
		F_D<<<N_blocks,N_threads>>>(Ovector_D,mem_D,Pvector_D,r25_D,r50_D,r100_D,s_D,w_D,OvectorVec_D,pop_D,NP);
		for (i = 0; i < H_maxsize; ++i) {M_F[i] = 0.5; M_Cr[i] = 0.5;} // Init Cr and F storage 

		//5. copy data from device to host
		cudaMemcpy(pop,pop_D,NP*sizeof(ind),cudaMemcpyDefault);

		//6. free memory
		cudaFree(pop_D);

		//7. update best and record it 
		update_best();
		global_best = pop[best]; //set the best individual of population 1 as global best
	}

	void evolve_in_device(default_random_engine &rng,curandState *state_D,ind &global_best,int &glshade_current_FEs,int glshade_stop_criterion)
	{

		//Integrate global_best to population (receive)
		pop[best] = global_best; // place it at best_id index position

		//Set counter, storage size counter 
		int counter = 0;
		int S_size = 0;
		int r,i;

	    //Allocate memory on device
	    cudaMalloc(&evol_data_D,NP*sizeof(evol_data_struct));
	    cudaMalloc(&pop_D,NP*sizeof(ind));
	    cudaMalloc(&rank_D,NP*sizeof(rank_ind));
	    cudaMalloc(&memory_D,2*NP*sizeof(ind));
	    cudaMalloc(&child_D,NP*sizeof(ind));
	    cudaMalloc(&S_F_D,NP*sizeof(double));
	    cudaMalloc(&S_Cr_D,NP*sizeof(double));
	    cudaMalloc(&W_D,NP*sizeof(double));
	    cudaMalloc(&mean_D,2*sizeof(double));

	    //While stopping condition is not met:
		while(counter<stop_criterion && glshade_current_FEs<glshade_stop_criterion)
		{
			/******************* SHADE ************************/
			// Join Population and external archive
			memory.insert(memory.end(), &pop[0], &pop[NP]); //memory = pop;
			memory.insert(memory.end(), A.begin(), A.end()); 

	    	//Prepare random data
		    for (i = 0; i < NP; ++i)
		    {	
		    	//Fill ranking list
		        ranklist[i].id = i; ranklist[i].fitness = pop[i].fx;

				/*******************Setting F and Cr************************/
				/*Generate F and Cr using a normal distribution with mean 
				taken randomly from storage*/
				r = rnd(0,H_maxsize-1); //take an index randomly  
				uF = M_F[r]; normal_distribution<double> Ndistribution_F(uF,0.1);
				uCr = M_Cr[r]; normal_distribution<double> Ndistribution_Cr(uCr,0.1);

		        evol_data[i].Cr = Ndistribution_Cr(rng); 
		        if (evol_data[i].Cr > 1.0) evol_data[i].Cr = 1.0; 
		        else if(evol_data[i].Cr < 0.0) evol_data[i].Cr = 0.0;

		        evol_data[i].F = Ndistribution_F(rng); 
		        if (evol_data[i].F > 1.0) evol_data[i].F = 1.0; 
		        while (evol_data[i].F <= 0.0) evol_data[i].F = Ndistribution_F(rng);

		        /*******************Setting p_best************************/
		        evol_data[i].p_best = rnd(0,int(rndreal(p_min,p_max)*NP)); //take an index within best range

		        /*******************Choosing a and b************************/
		        // randomly pick 2 different members
		        do evol_data[i].a = rnd(0,NP-1); while(evol_data[i].a==i); // from pop
		        do evol_data[i].b = rnd(0,memory.size()-1); while(evol_data[i].b==i || evol_data[i].b==evol_data[i].a); // from pop U archive

		        /*******************Get j_rand************************/
		        evol_data[i].j_rand = rnd(0,D-1);
		    }
		    //Rank population by fitness
		    sort(ranklist);//sort by fitness min => ranklist[0].fitness 

			//Load generated data and current population to device 
			cudaMemcpy(evol_data_D,evol_data,NP*sizeof(evol_data_struct),cudaMemcpyDefault);
			cudaMemcpy(pop_D,pop,NP*sizeof(ind),cudaMemcpyDefault);
			cudaMemcpy(rank_D,ranklist,NP*sizeof(rank_ind),cudaMemcpyDefault);
			cudaMemcpy(memory_D,memory.data(),memory.size()*sizeof(ind),cudaMemcpyDefault);

			//Lauch kernel: mutation,recombination and function evaluation 
			shade_engine<<<N_blocks,N_threads>>>(state_D,evol_data_D,pop_D,rank_D,memory_D,child_D,NP);
			F_D<<<N_blocks,N_threads>>>(Ovector_D,mem_D,Pvector_D,r25_D,r50_D,r100_D,s_D,w_D,OvectorVec_D,child_D,NP);
			cudaMemcpy(child,child_D,NP*sizeof(ind),cudaMemcpyDefault);
			
			//Selection
			for (i = 0; i < NP; ++i)
			{
				//Update FEs counter 
				glshade_current_FEs += 1; counter += 1;
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
		        	if (child[i].fx<global_best.fx && glshade_current_FEs<=glshade_stop_criterion)
		        	{
		        		global_best = child[i];
		        		global_best.FEs_when_found = glshade_current_FEs;
		        	}
		        	//Advance child to next generation
		        	pop[i] = child[i];
		        }
				if(glshade_current_FEs==1.2e5 || glshade_current_FEs==3e5 || glshade_current_FEs==6e5 || 
				glshade_current_FEs==9e5 || glshade_current_FEs==1.2e6 || glshade_current_FEs==1.5e6
				|| glshade_current_FEs==1.8e6 || glshade_current_FEs==2.1e6 || glshade_current_FEs==2.4e6 || 
				glshade_current_FEs==2.7e6 || glshade_current_FEs==3e6)
					fprintf(file_results,"%d,%d,%.2f,%.6e\n",glshade_current_FEs,ID,Rseed,global_best.fx);

			}
			//If F and Cr storages are non-empty
			if (S_size > 0)
			{	//Load F, Cr and W data to device  
				cudaMemcpy(S_F_D,S_F,S_size*sizeof(double),cudaMemcpyDefault);
				cudaMemcpy(S_Cr_D,S_Cr,S_size*sizeof(double),cudaMemcpyDefault);
				cudaMemcpy(W_D,W,S_size*sizeof(double),cudaMemcpyDefault);
				mean_WAWL<<<2,64>>>(S_Cr_D,S_F_D,W_D,S_size,mean_D); //Compute mean WA and mean WL in device
			}

			// Concurrently update best solution index
			update_best();

			// Concurrently check external archive
			A.insert(A.end(), A_tmp.begin(), A_tmp.end()); // add defeated parents to A
			apply_A_maintenance();//|A| must be less than or equal to popsize

			// Update M_CR and M_F
			if (S_size > 0)
			{	//Record means 
				cudaMemcpy(mean,mean_D,2*sizeof(double),cudaMemcpyDefault);
				M_F[k] = mean[1]; //weighted Lehmer mean (WL)
				M_Cr[k] = mean[0]; //weighted arithmetic mean (WA)
				k = (k + 1) % H_maxsize;
			}

			// Reset and go again
			S_size = 0;
			A_tmp.clear();
			memory.clear();
		}
	    //Free memory
	    cudaFree(evol_data_D);
	    cudaFree(pop_D);
	    cudaFree(rank_D);
	    cudaFree(memory_D);
	    cudaFree(child_D);
	    cudaFree(S_F_D);
	    cudaFree(S_Cr_D);
	    cudaFree(W_D);
	    cudaFree(mean_D);
	}

	void free_memory()
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
	}

};

