//Basic and reading functions.
/**************************READING***************************************/
double* readOvector(int dimension, ushort ID)
{
  // read O vector from file in csv format
  double* d = new double[dimension];
  stringstream ss;
  ss<< "cdatafiles/" << "F" << ID << "-xopt.txt";
  ifstream file (ss.str());
  string value;
  string line;
  int c=0;
  
  if (file.is_open())
    {
      stringstream iss;
      while ( getline(file, line) )
        {
          iss<<line;
          while (getline(iss, value, ','))
            {
              d[c++] = stod(value);
            }
          iss.clear();
        }
      file.close();
    }
  else
    {
      cout<<"Cannot open datafiles"<<endl;
    }
  return d;
}
////////////////////////////////////////////////////////////////////////////////
void readPermVector(int *d,int dimension, ushort ID){

  //int* d;
  //d = new int[dimension];

  stringstream ss;
  ss<< "cdatafiles/" << "F" << ID << "-p.txt";
  ifstream file (ss.str());
  int c=0;
  string value;

  if (file.is_open())
    {
      while (getline(file,value,','))
        {
          d[c++] = stod(value) - 1;
        }
    }
  //return(d);
}
////////////////////////////////////////////////////////////////////////////////
void readS(int *s0,int num, ushort ID)
{
  //int *s0 = new int[num];

  stringstream ss;
  ss<< "cdatafiles/" << "F" << ID << "-s.txt";
  ifstream file (ss.str());
  int c=0;
  string value;
  if (file.is_open())
    {
      while (getline(file,value))
        {
          // cout<<stod(value)<<endl;
          s0[c++] = stod(value);
        }
    }
  //return s0;
}
////////////////////////////////////////////////////////////////////////////////
void readR(double *m,int sub_dim, ushort ID)
{
  //double* m;
  //m = new double[sub_dim*sub_dim];
  // for (int i = 0; i< sub_dim; i++)
  //   {
  //     m[i] = new double[sub_dim];
  //   }

  stringstream ss;
  ss<< "cdatafiles/" << "F" << ID << "-R"<<sub_dim<<".txt";
  // cout<<ss.str()<<endl;

  ifstream file (ss.str());
  string value;
  string line;
  int i=0;
  int j;

  if (file.is_open())
    {
      stringstream iss;
      while ( getline(file, line) )
        {
          j=0;
          iss<<line;
          while (getline(iss, value, ','))
            {
              // printf("%d,%d\t%f\n", i,j, stod(value));
              m[i*sub_dim + j] = stod(value);
              // printf("done\n");
              j++;
            }
          iss.clear();
          i++;
        }
      file.close();
    }
  else
    {
      cout<<"Cannot open datafiles"<<endl;
    }
  //return m;
}
////////////////////////////////////////////////////////////////////////////////
void readW(double *w0,int num, ushort ID)
{
  //double *w0 = new double[num];

  stringstream ss;
  ss<< "cdatafiles/" << "F" << ID << "-w.txt";
  ifstream file (ss.str());
  int c=0;
  string value;
  if (file.is_open())
    {
      while (getline(file,value))
        {
          // cout<<stod(value)<<endl;
          w0[c++] = stod(value);
        }
    }

  //return w0;
}
////////////////////////////////////////////////////////////////////////////////
double** readOvectorVec(ushort ID)
{
  // read O vector from file in csv format, seperated by s_size groups
  double** d = (double**) malloc(s_size*sizeof(double*));
  stringstream ss;
  ss<< "cdatafiles/" << "F" << ID << "-xopt.txt";
  ifstream file (ss.str());
  string value;
  string line;
  int c = 0;                      // index over 1 to dim
  int i = -1;                      // index over 1 to s_size
  int up = 0;                   // current upper bound for one group
  
  if (file.is_open())
    {
      stringstream iss;
      while ( getline(file, line) )
        {
          if (c==up)             // out (start) of one group
            {
              // printf("=\n");
              i++;
              d[i] =  (double*) malloc(s[i]*sizeof(double));
              up += s[i];
            }
          iss<<line;
          while (getline(iss, value, ','))
            {
              // printf("c=%d\ts=%d\ti=%d\tup=%d\tindex=%d\n",c,s[i],i,up,c-(up-s[i]));
              d[i][c-(up-s[i])] = stod(value);
              // printf("1\n");
              c++;
            }
          iss.clear();
          // printf("2\n");
        }
      file.close();
    }
  else
    {
      cout<<"Cannot open datafiles"<<endl;
    }
  return d;  
}
/******************************************************************************/
/////////////////////////////////////////////////////////////////////////
/////////////////////////////C///////////////////////////////////////////
/////////////////////////////U///////////////////////////////////////////
/////////////////////////////D///////////////////////////////////////////
/////////////////////////////A///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/*****************************CUDA*******************************************/
__device__ int sign(double x)
{
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
} 
////////////////////////////////////////////////////////////////////////////////
__device__ double hat(double x)
{
  if (x==0) //if (fabs(x)<= 1e-6)
    {
      return 0;
    }
  else
    {
      return log(abs(x));
    }
}
////////////////////////////////////////////////////////////////////////////////
__device__ double c1(double x)
{
  if (x>0)
    {
      return 10;
    }
  else
    {
      return 5.5;
    }
}
////////////////////////////////////////////////////////////////////////////////
__device__ double c2(double x)
{
  if (x>0)
    {
      return 7.9;
    }
  else
    {
      return 3.1;
    }
}
////////////////////////////////////////////////////////////////////////////////
__device__ void multiplyR25_normal(double *vector,double *matrix,int size,int j,double *y)
{
  /*
  Here it's performed a normal matrix multiplication (No pagination). Pagination means 
  downloading data pages from global memory to shared memory. 
  */
  int l;
  //For every matrix row do...
  while(j<size) //let in all threads which private j variable is less than number of rows 
  { // let thread handle a row 
    y[j] = 0.0;
    for(l = size-1; l >=0; l--) 
      y[j] += (vector[l] * matrix[j*size + l]);
    j += blockDim.x; //advance j: this line is equivalent to say j += number of threads in block
  }
  __syncthreads();
}
////////////////////////////////////////////////////////////////////////////////
__device__ void multiplyR50_pagination(double *vector,double *matrix,int size,int j,
double *y,double *cache,int pagesize,int &mempointer,int &rowcount)
{
  /* multiply_s
  Pagination means that a data page (matrix rows) is loaded from global memory to cache as matrix R50 is too heavy to load it completely.
  The pagination were found useful to speed up computation as reading directly matrix entries from global memory is very slow.
  */
  int n,step;

  //Let thread #0 to set shared variables  
  if (threadIdx.x==0) {rowcount = 0; mempointer = 0;}
  /*If pagesize is set to 1000, then we get step = 20 = 1000/50 meaning that we can load 20 rows in one page */ 
  step = pagesize/size;
  __syncthreads(); //wait for all 

  while(rowcount<40)
  { 
    //Loading page from global memory to cache
    j = threadIdx.x;
    while(j<pagesize)
    {
      cache[j] = matrix[mempointer+j];
      j+=blockDim.x;
    }
    __syncthreads();

    //Processing page or every row 
    if (threadIdx.x<step)
    {
      j = rowcount + threadIdx.x;
      y[j] = 0.0;
      for(n=size-1; n>=0; n--) y[j] += (vector[n] * cache[threadIdx.x*size + n]); 
    }
    if (threadIdx.x==step){rowcount += step; mempointer += pagesize;}
    __syncthreads();
  }

  //Loading last page from global memory to cache
  j = threadIdx.x;
  while(j<500)
  {
    cache[j] = matrix[2000+j];
    j+=blockDim.x;
  }
  __syncthreads();
  //Processing page
  if (threadIdx.x<10)
  {
    j = 40 + threadIdx.x;
    y[j] = 0.0;
    for(n=size-1; n>=0; n--) y[j] += (vector[n] * cache[threadIdx.x*size + n]); 
  }
}
////////////////////////////////////////////////////////////////////////////////
__device__ void multiplyR100_pagination(double *vector,double *matrix,int size,int j,
double *y,double *cache,int pagesize,int &mempointer,int &rowcount)
{
  /* multiply_s
  Pagination means that a data page (matrix rows) is loaded from global memory to cache as matrix R100 is too heavy to load it completely.
  The pagination were found useful to speed up computation as reading directly matrix entries from global memory is very slow.
  */
  int n,step;

  if (threadIdx.x==0) {rowcount = 0; mempointer = 0;}
  /*If pagesize is set to 1500, then we get step = 15 = 1500/100 meaning that we can load 15 rows in one page */ 
  step = pagesize/size;
  __syncthreads();

  while(rowcount<90)
  { 
    //Loading page from global memory to cache
    j = threadIdx.x;
    while(j<pagesize)
    {
      cache[j] = matrix[mempointer+j];
      j+=blockDim.x;
    }
    __syncthreads();

    //Processing page
    if (threadIdx.x<step)
    {
      j = rowcount + threadIdx.x;
      y[j] = 0.0;
      for(n=size-1; n>=0; n--) y[j] += (vector[n] * cache[threadIdx.x*size + n]); 
    }
    if (threadIdx.x==step){rowcount += step; mempointer += pagesize;}
    __syncthreads();
  }

  //Loading last page from global memory to cache
  j = threadIdx.x;
  while(j<1000)
  {
    cache[j] = matrix[9000+j];
    j+=blockDim.x;
  }
  __syncthreads();
  //Processing page
  if (threadIdx.x<10)
  {
    j = 90 + threadIdx.x;
    y[j] = 0.0;
    for(n=size-1; n>=0; n--) y[j] += (vector[n] * cache[threadIdx.x*size + n]); 
  }
}
////////////////////////////////////////////////////////////////////////////////
__device__ void reduce(double *cache)
{
  //Perform a dichotomous reduction:cache[0] = cache[0] + cache[1] + ... + cache[number of threads]
  int k = blockDim.x/2;
  while (k != 0) 
  {
    if (threadIdx.x < k) cache[threadIdx.x] += cache[threadIdx.x + k];
    __syncthreads();
    k /= 2; 
  }
  __syncthreads();
}
__device__ void reduce_twice(double *cache,double *cache2)
{
  //Perform a dichotomous reduction: cache[0] = cache[0] + cache[1] + ... + cache[number of threads] 
  int k = blockDim.x/2;
  while (k != 0) 
  {
    if (threadIdx.x < k) 
    {
      cache[threadIdx.x] += cache[threadIdx.x + k];
      cache2[threadIdx.x] += cache2[threadIdx.x + k];
    }
    __syncthreads();
    k /= 2; 
  }
  __syncthreads();
}
/******************************************************************************/
/////////////////////////////////////////////////////////////////////////
/////////////////////////////O///////////////////////////////////////////
/////////////////////////////M///////////////////////////////////////////
/////////////////////////////P///////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/************************OMP*********************************************/
int sign_h(double x)
{
  if (x > 0) return 1;
  if (x < 0) return -1;
  return 0;
} 
////////////////////////////////////////////////////////////////////////////////
double hat_h(double x)
{
  if (x==0) //if (fabs(x)<= 1e-6)
    {
      return 0;
    }
  else
    {
      return log(abs(x));
    }
}
////////////////////////////////////////////////////////////////////////////////
double c1_h(double x)
{
  if (x>0)
    {
      return 10;
    }
  else
    {
      return 5.5;
    }
}
////////////////////////////////////////////////////////////////////////////////
double c2_h(double x)
{
  if (x>0)
    {
      return 7.9;
    }
  else
    {
      return 3.1;
    }
}
////////////////////////////////////////////////////////////////////////////////
void multiply_h(double *vector,double *matrix,int size,double *y)
{
  int m,n;

  #pragma omp parallel for num_threads(maxThreads)
  for (m=0; m<size; ++m)
  {
    y[m] = 0.0;
    for (n=0; n<size; ++n) y[m] += (vector[n] * matrix[m*size + n]);
  }
}
////////////////////////////////////////////////////////////////////////////////

