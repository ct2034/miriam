# source miriam_env/bin/activate
export PYTHONPATH=${PYTHONPATH}:$(pwd)
export SCENARIO_STORAGE_PATH=.scenario_cache
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
ulimit -n 100000
