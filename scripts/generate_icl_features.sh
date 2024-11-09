export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export MASTER_ADDR=localhost
export MASTER_PORT=5344

taskset -c 50-70 python eval/cache_rices_features.py