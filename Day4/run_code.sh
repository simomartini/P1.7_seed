#!/bin/bash

#PBS -l nodes=1:ppn=20 -l walltime=00:10:00 -q reserved2

cd /home/smartini/magma-2.2.0/example

module load cudatoolkit/7.5
module load openblas

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/smartini/magma-2.2.0/lib
export OMP_NUM_THREADS=20

./example_v1_real > output.txt

