#!/bin/bash
echo nvcc glshade.cu -O2 -D=f$4 -Xcompiler -fopenmp
nvcc glshade.cu -O2 -D=f$4 -Xcompiler -fopenmp
echo ./a.out popsize1=$1 popsize2=$2 Rseed=$3 
echo *******************CONSOLE***************************
#export OMP_NUM_THREADS=4
./a.out $1 $2 $3
echo *******************CSV_FILE**************************
cat *f$4.csv 
