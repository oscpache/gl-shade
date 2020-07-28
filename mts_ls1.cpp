#include <iostream>
#include <algorithm>
#include <vector>
#include <random>

/*
 * Implements the MTS-LS1 indicated in MTS
 * http://sci2s.ugr.es/EAMHCO/pdfs/contributionsCEC08/tseng08mts.pdf
 * Lin-Yu Tseng; Chun Chen, "Multiple trajectory search for Large Scale Global Optimization," Evolutionary Computation, 2008. CEC 2008. (IEEE World Congress on Computational Intelligence). IEEE Congress on , vol., no., pp.3052,3059, 1-6 June 2008
 * doi: 10.1109/CEC.2008.4631210
 */

 /******************************************************************************/
/////////////////////////////mts_ls//////////////////////////////////
/******************************************************************************/
class mts_ls1
{
//private
int D,stop_criterion,counter;
float ub,lb;
typedef struct{vector<double> solution; double fitness; int evals;} LSresult;

public:
	vector<double> improvement;
	vector<double> SR;

	mts_ls1(float lowbound, float upbound, int N_decison_var=1000,int mtsls1_stop_criterion=25000)
	{	
		//__init__
		lb = lowbound;
		ub = upbound;
		D = N_decison_var;
		stop_criterion = mtsls1_stop_criterion;
		
		// Set improvement and search range 
		int k;
		for (k = 0; k < dim; ++k)
		{
			improvement.push_back(0.0); //improvement
			SR.push_back((ub - lb) * 0.2); //search range: step size = 20
		}
	}

	LSresult mts_ls1_improve_dim(vector<double> sol,double best_fitness, int i,int &glshade_current_FEs,int &glshade_stop_criterion)
	{
	  vector<double> newsol = sol;
	  newsol[i] -= SR[i];

	  // Check new solution
	  if (newsol[i] > ub) newsol[i] = (ub+sol[i])/2;
	  else if (newsol[i] < lb) newsol[i] = (lb+sol[i])/2;

	  //f(x) on CPU with omp
	  /**********************************/
	  double fitness_newsol = F_H(newsol.data()); //F_H <- objective function with omp
	  glshade_current_FEs += 1; counter += 1;

	  //write results to output file
	  int FEs = glshade_current_FEs; 
	  if(FEs==1.2e5 || FEs==3e5 || FEs==6e5 || FEs==9e5 || FEs==1.2e6 || FEs==1.5e6
	  || FEs==1.8e6 || FEs==2.1e6 || FEs==2.4e6 || FEs==2.7e6 || FEs==3e6)
	  {
	  	if(fitness_newsol<best_fitness)
	  		fprintf(file_results,"%d,%d,%.2f,%.6e\n",FEs,ID,Rseed,fitness_newsol); 
	  	else
	  		fprintf(file_results,"%d,%d,%.2f,%.6e\n",FEs,ID,Rseed,best_fitness); 
	  }
	  /**********************************/

	  int evals = 1;
	  if(fitness_newsol < best_fitness)
	  {
	    best_fitness = fitness_newsol;
	    sol = newsol;
	  } 
	  else if( fitness_newsol > best_fitness )
	  {
	    newsol[i] = sol[i];
	    newsol[i] += 0.5 * SR[i];

	    // Check new solution
	  	if (newsol[i] > ub) newsol[i] = (ub+sol[i])/2;
	  	else if (newsol[i] < lb) newsol[i] = (lb+sol[i])/2;

	  	//f(x) on CPU with omp
	  	/**********************************/
	    fitness_newsol = F_H(newsol.data());
	    glshade_current_FEs += 1; counter += 1; 

	  	//write results to output file
	  	FEs = glshade_current_FEs;
		if(FEs==1.2e5 || FEs==3e5 || FEs==6e5 || FEs==9e5 || FEs==1.2e6 || FEs==1.5e6
		|| FEs==1.8e6 || FEs==2.1e6 || FEs==2.4e6 || FEs==2.7e6 || FEs==3e6)
		{
			if(fitness_newsol<best_fitness)
			{
				fprintf(file_results,"%d,%d,%.2f,%.6e\n",FEs,ID,Rseed,fitness_newsol);  
			}
			else
			{
				fprintf(file_results,"%d,%d,%.2f,%.6e\n",FEs,ID,Rseed,best_fitness); 
			}
		}
	    /**********************************/

	    evals++;
	    if(fitness_newsol < best_fitness )
	    {
	      best_fitness = fitness_newsol;
	      sol = newsol;
	    }
	  }
	  
	  return LSresult{sol, best_fitness, evals};
	}

	void enhance(ind &global_best, int &glshade_current_FEs, int glshade_stop_criterion)
	{
		counter = 0;
		vector<double> sol(D);
		LSresult result;

	  	//Set global best as current best  
	  	double best_fitness = global_best.fx;
	  	memcpy(sol.data(),global_best.x, D*sizeof(double));
	  	LSresult current_best = {sol, best_fitness, 0};

	  	vector<double> dim_sorted(D);
	  	iota(dim_sorted.begin(), dim_sorted.end(), 0);

	  	double improve;
	  	//warm-up
	  	if(counter<stop_criterion && glshade_current_FEs<glshade_stop_criterion)
	  	{
		    next_permutation(dim_sorted.begin(), dim_sorted.end());
		    for( auto it = dim_sorted.begin(); it != dim_sorted.end(); it++ )
		    {
		      	result = mts_ls1_improve_dim(sol, best_fitness, *it,glshade_current_FEs, glshade_stop_criterion);
		      	improve = max(current_best.fitness - result.fitness, 0.0);
		      	improvement[*it] = improve;
		      	if( improve > 0.0 ) current_best = result;
		      	else SR[*it] /= 2.0;
			}
	  	}
		iota(dim_sorted.begin(), dim_sorted.end(), 0);
		sort(dim_sorted.begin(), dim_sorted.end(), [&](unsigned i1, unsigned i2) { return improvement[i1] > improvement[i2]; });

		int i, d = 0, next_d, next_i;
		while(counter<stop_criterion && glshade_current_FEs<glshade_stop_criterion)
		{
			i = dim_sorted[d];
			result = mts_ls1_improve_dim(current_best.solution, current_best.fitness,i,glshade_current_FEs, glshade_stop_criterion);
			improve = max(current_best.fitness - result.fitness, 0.0);
			improvement[i] = improve;
			next_d = (d+1)%D;
			next_i = dim_sorted[next_d];

			if( improve > 0.0 )
			{
			  current_best = result;
			  if( improvement[i] < improvement[next_i] )
			  {
			    iota(dim_sorted.begin(), dim_sorted.end(), 0);
			    sort(dim_sorted.begin(), dim_sorted.end(), [&](unsigned i1, unsigned i2) { return improvement[i1] > improvement[i2]; });
			  }
			}
			else 
			{
			  SR[i] /= 2.0;
			  d = next_d;
			  if( SR[i] < 1e-15 ) SR[i] = (ub - lb) * 0.2;
			}
		}

		// Register new global best 
		memcpy(global_best.x,current_best.solution.data(), D*sizeof(double)); 
		global_best.fx = current_best.fitness;
		global_best.FEs_when_found = glshade_current_FEs;

		return;
	}
};



