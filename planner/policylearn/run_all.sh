#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/../..
N_DATA_TO_GEN=1000000 ./generate_data.py generate_simulation data.pkl
./generate_data.py transfer_classification data_class.pkl data.pkl
# ./generate_data_demo.py plot_fovs data_class.pkl
./train_model_classification.py data_class.pkl
