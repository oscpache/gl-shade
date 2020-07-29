# GL-SHADE 
This source code implements the GL-SHADE algorithm presented in the paper "A SHADE-Based Algorithm for Large Scale Global Optimization" at the PPSN 2020 international conference. Here it´s showed a parallell implementation of GL-SHADE using CUDA + OpenMP and adopting the CEC'13 LSGO Benchmark test suite. Futher, all test functions are implemented in both CUDA and OpenMP; the gpu based implementation of a test problem is used most of the time but in very special cases the OpenMP based implementation is employed.

## Getting Started

### Prerequisites
The implementation was tested using a pc with ubuntu 18.04 as operating system and a GeForce GTX 680 GPU with the CUDA 10.2 version. For running the program you need a GPU enabled to work with CUDA version 7.0 or higher (since syntax c++11 is used) and, if possible, a pc powered by a linux like operating system.  

## Running the program 

The number of threads and blocks when using the GPU as well as the number of CPU cores activated when using OpenMP are predefined at the top of the main program named glshade.cu but we encourage you to modify them according to you GPU and CPU hardware. So, for running the main program just 4 parameters have to be defined: population size 1, population size 2, random number generator seed and problem (the one to be adopted as the objective function) identifier.  

### Compile

Explain what these tests test and why

```
nvcc glshade.cu -O2 -D=f7 -Xcompiler -fopenmp
```

### Run

Explain what these tests test and why

```
a.out 85 54 0.76
```
