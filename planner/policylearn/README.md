# How to use
First, generate data by solving environments
```
N_DATA_TO_GEN=1000000 ./generate_data.py generate_simulation data/data.pkl
```

<!-- Merge the files
```
./generate_data.py merge_files -m data/data*.pkl
``` -->

Then transfer to gcn samples
```
./run_conversion.sh
```

Have a look at the data with ...
```
./generate_data_demo.py plot_graph data/data_gcn02.pkl
```

Train the model
```
./train_model_gcn.py data/data_gcn*.pkl
```

<!-- Evaluate the model with ..
```
./evaluate_model_classification.py my_model.h5
``` -->