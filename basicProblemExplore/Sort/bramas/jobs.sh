#!/bin/bash
#PBS -q a64fx
#PBS -l select=1:ncpus=48,place=scatter
#PBS -l walltime=24:00:00


## Bérenger Bramas (berenger.bramas@inria.fr)
## ARM SVE SORT
## This file is batch job that tries to run all
## configurations (this cannot be done within 24H
## so it is prefered to use the models+gen.sh system)

source ~/.bashrc

TIMESTAMP=$(date +"%H-%M-%S-%d-%m-%y")
CXX="armclang++"

cd ~/arm-sve-sort/

echo "Create results/$TIMESTAMP-$CXX"
mkdir -p "results/$TIMESTAMP-$CXX"
cd "results/$TIMESTAMP-$CXX"

echo "taskset -c 0 ~/arm-sve-sort/sortSVEperf.$CXX.exe seq > res.txt"
taskset -c 0 ~/arm-sve-sort/sortSVEperf.$CXX.exe seq > res.txt

echo "OMP_NUM_THREADS=48 OMP_PROC_BIND=TRUE OMP_WAIT_POLICY=ACTIVE ~/arm-sve-sort/sortSVEperf.$CXX.exe par > res.txt"
OMP_NUM_THREADS=48 OMP_PROC_BIND=TRUE OMP_WAIT_POLICY=ACTIVE ~/arm-sve-sort/sortSVEperf.$CXX.exe par > res.txt

