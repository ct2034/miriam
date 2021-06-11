# How to use
First, generate data by solving environments
```
N_DATA_TO_GEN=1000000 ./generate_data.py generate_simulation data/data.pkl
```

Merge the files
```
./generate_data.py merge_files -m data/data*.pkl
```

Then transfer to classification samples
```
./generate_data.py transfer_classification data_class.pkl data.pkl
```

Have a look at the data with ...
```
./generate_data_demo.py plot_fovs data_class.pkl
```

Train the model
```
./train_model.py data_class.pkl my_model.h5
```

Evaluate the model with ..
```
./evaluate_model_classification.py my_model.h5
```