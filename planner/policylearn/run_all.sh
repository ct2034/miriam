#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../..
N_DATA_TO_GEN=1000000 ./generate_data.py generate_simulation data/data.pkl
# ./generate_data.py transfer_classification data/data_class.pkl data/data.pkl
./run_conversion.sh
# ./generate_data_demo.py plot_fovs data/data_class01.pkl
# ./train_model.py data/data_class.pkl my_model.h5
