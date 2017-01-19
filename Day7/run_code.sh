#!/bin/bash

#PBS -l nodes=1:ppn=20 -l walltime=00:10:00

cd P1.7_seed/Day7/

module load intel/14.0
module load mkl/11.1
 
icc -mkl=sequential -xAVX exercise1-matrixmult.cpp
./a.out > output.txt

