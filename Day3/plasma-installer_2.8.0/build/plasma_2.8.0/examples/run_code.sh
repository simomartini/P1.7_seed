#!/bin/bash

#PBS -l nodes=1:ppn=20 -l walltime=00:10:00

cd P1.7_seed/Day3/plasma-installer_2.8.0/build/plasma_2.8.0/examples

module load mkl/11.1 hwloc
module load python 

./example_dgesv 3000 > output.txt

