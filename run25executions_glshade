#!/bin/bash
for ((f=1;f<=15;f++)) 
do	echo compiling f$f
	nvcc glshade.cu -o a.out -std=c++11 -w -O2 -D=f$f -Xcompiler -fopenmp #compile
	echo processing f$f...
	for ((k=4;k<=100;k+=4))
	do
	    i=$(bc <<<"scale=2; $k / 100" )
	    ./a.out 100 100 $i $f >> glshade_time_100pop1_100pop2_f$f.csv #execute
	    echo $i
	done
done
