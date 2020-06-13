# gl-shade
A SHADE-Based Algorithm for Large Scale Global Optimization 

# Overall description
Algorithm: GL-SHADE

    -> Author: CINVESTAV-IPN (Evolutionary Computation Group)

    -> Implemented using: C++/CUDA + OpenMP

    -> Year: 2020

    -> Requirements: CUDA version 7.0 or higher (since syntax c++11 is used)   

Compiling and executing on a linux like operating system

    -> Compile: nvcc glshade.cu -o a.out -std=c++11 -O2 -D=fID -Xcompiler -fopenmp
    
    -> It is no necessary to set the OMP_NUM_THREADS variable since the program uses always 4 threads by default 
    
    -> Run: ./a.out popsize1 popsize2 Rseed ID

Input arguments 

    -> Rseed must range from 0.0 to 1.0 [float]
    
    -> popsize1 and popsize2 must be at most 1000 and at least 4 [int]
    
    -> ID is the objective function identifier; 1 <= ID <= 15 [int] 

Execution example using f4 as objective funtion, popsize1 = 100 , popsize2 = 75 and Rseed = 0.44

    -> $ nvcc glshade.cu -o a.out -std=c++11 -O2 -D=f4 -Xcompiler -fopenmp
    -> $ ./a.out 100 75 0.44 4
 
Important notes

    -> The CEC'13 LSGO benchmark is used

    -> The code was tested using Ubuntu 18.04 as operating system and a GeForce GTX 680 GPU with the CUDA 10.2 version
    
    -> You can use the run_glshade bash file to run this code as follows: bash run_glshade.sh popsize1 popsize2 Rseed ID 
    
    -> You can use the run25executions_glshade bash file to perform 25 GL-SHADE's independent runs of every objective function as follows: bash run25executions_glshade.sh
    
    -> Note how the objective function is chosen at compilation time and such function identifier is confirmed when running the ./a.out program [see the 4th argument] 
