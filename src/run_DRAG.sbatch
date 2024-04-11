#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --output=../output/slurm-%j.out
 
make main
##./main ./datasets/RandomWalk/RW_1M.txt 1000000 1024 1024 1024
##./main ../datasets/Mars/P_3.csv 8233 512 512 1
./main ../datasets/A4Benchmark.csv 1680 128 128 2.8