#!/bin/bash
echo nvcc glshade.cu -o a.out -std=c++11 -O2 -D=f$4 -Xcompiler -fopenmp
nvcc glshade.cu -o a.out -std=c++11 -O2 -D=f$4 -Xcompiler -fopenmp 
export OMP_NUM_THREADS=4
echo ./a.out popsize1=$1 popsize2=$2 Rseed=$3 f_objective=$4 
./a.out $1 $2 $3 $4