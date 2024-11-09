#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export MASTER_PORT=$(shuf -i 0-65535 -n 1)

export PYTHONPATH="$PYTHONPATH:open_flamingo"
taskset -c 150-170 python scorer.py