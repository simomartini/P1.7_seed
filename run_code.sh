#!/bin/bash

#PBS -l nodes=1:ppn=20 -q reserved2 -l walltime=01:00:00

cd P1.7_seed

module load gnu/4.9.2
module load cudatoolkit/7.5

./a.out > output.txt