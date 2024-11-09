#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export MASTER_ADDR=localhost
export MASTER_PORT=5622

taskset -c 60-80 python eval/evaluate.py