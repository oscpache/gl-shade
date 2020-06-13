//Supporting functions
/****************************SUPPORTING*************************************/
void setBounds_cec13(int id,float &lb,float &ub)
{
	if (id==1 || id==4 || id==7 || id==8 || id==12 || id==13 || id==14 || id==15 || id==9)
	{
		lb = -100;
		ub = 100;
	}
	else if (id==2 || id==5 || id==10)
	{
		lb = -5;
		ub = 5;
	}
	else //3,6,11
	{
		lb = -32;
		ub = 32;
	}

	return;
}

void SetBenchFramework(int &DIM,float &lb,float &ub)
{
	int i;
	DIM = dim;

	//Setting bounds and s_size
	setBounds_cec13(ID,lb,ub);
	if(ID <= 7) s_size = 7; else s_size = 20;
    cudaMemcpyToSymbol(lb_d, &lb, sizeof(lb_d));
    cudaMemcpyToSymbol(ub_d, &ub, sizeof(ub_d));
	cudaMemcpyToSymbolAsync(s_size_d, &s_size, sizeof(s_size_d));//set s_size on device

	//Allocate extra storage to work on device
	cudaMalloc(&mem_D,N_blocks*sizeof(storage));//hold on global memory

	/*
	Reading data,
	Allocating memory on device,
	Loading data,
	Freeing memory space that'll not be needed any more.
	*/

	//Set Pvector,r25,r50,r100,s,w
	if ((ID>=4 && ID<=11) or ID==13 or ID==14)
	{
		//Allocate memory space on device
		cudaMalloc(&Pvector_D,dim*sizeof(int));
		cudaMalloc(&r25_D,25*25*sizeof(double));
		cudaMalloc(&r50_D,50*50*sizeof(double));
		cudaMalloc(&r100_D,100*100*sizeof(double));
		cudaMalloc(&s_D,s_size*sizeof(int));
		cudaMalloc(&w_D,s_size*sizeof(double));

		//Read data from host and load it to device asynchronously
		cudaMallocHost(&Pvector,dim*sizeof(int));
		readPermVector(Pvector,dim);
		cudaMemcpyAsync(Pvector_D,Pvector,dim*sizeof(int),cudaMemcpyDefault);

		cudaMallocHost(&r25,25*25*sizeof(double));
		readR(r25,25);
		cudaMemcpyAsync(r25_D,r25,25*25*sizeof(double),cudaMemcpyDefault);

		cudaMallocHost(&r50,50*50*sizeof(double));
		readR(r50,50);
		cudaMemcpyAsync(r50_D,r50,50*50*sizeof(double),cudaMemcpyDefault);

		cudaMallocHost(&r100,100*100*sizeof(double));
		readR(r100,100);
		cudaMemcpyAsync(r100_D,r100,100*100*sizeof(double),cudaMemcpyDefault);

		cudaMallocHost(&s,s_size*sizeof(int));
		readS(s,s_size);
		cudaMemcpyAsync(s_D,s,s_size*sizeof(int),cudaMemcpyDefault);

		cudaMallocHost(&w,s_size*sizeof(double));
		readW(w,s_size);
		cudaMemcpy(w_D,w,s_size*sizeof(double),cudaMemcpyDefault);		
	}


	//Set Ovector and OvectorVec properly
	if (ID != 14)
	{
		//Read data
		Ovector = readOvector(dim);

		//Allocate memory space on device
		cudaMalloc(&Ovector_D,dim*sizeof(double));

		//Load data to device
		cudaMemcpy(Ovector_D,Ovector,dim*sizeof(double),cudaMemcpyDefault);
	}
	else //ID==14
	{
		//Read data
		OvectorVec = readOvectorVec();

		//Allocate memory space on device
		cudaMalloc(&OvectorVec_D,s_size*sizeof(row));

		//Load data to device
		cudaMemcpy (OvectorVec_D, OvectorVec, s_size*sizeof(row), cudaMemcpyDefault);
		for (i = 0; i < s_size; ++i) cudaMemcpy(OvectorVec_D[i].col, OvectorVec[i],s[i]*sizeof(double), cudaMemcpyDefault);
	}
}

void FreeBenchData()
{
	int i;
	cudaFree(mem_D);

	if ((ID>=4 && ID<=11) || ID==13 || ID==14)
	{
		cudaFree(Pvector_D);
		cudaFree(r25_D);
		cudaFree(r50_D);
		cudaFree(r100_D);
		cudaFree(s_D);
		cudaFree(w_D);
		//Free memory space on host
		cudaFreeHost(Pvector);
		cudaFreeHost(r25);
		cudaFreeHost(r50);
		cudaFreeHost(r100);
		cudaFreeHost(w);
		cudaFreeHost(s);
	}

	if (ID != 14) 
		{cudaFree(Ovector_D); free(Ovector);}
	else //ID==14
	{
		cudaFree(OvectorVec_D); 
		for (i = 0; i < s_size; i++) free(OvectorVec[i]); 
		free(OvectorVec);
	}
}

int VerifyBestSolution(double *solution,float lb,float ub)
{
  int ans = 1;
  int j;
  for (j = 0; j < dim; ++j)
  {
    if(solution[j]<lb || solution[j]>ub)
    {
      ans = 0;
      break;
    }
  }
  return ans;
}

/******************************************************************************/

