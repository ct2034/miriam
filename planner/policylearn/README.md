# How to use
First, generate data by solving environments

```
./generate_data.py generate_simulation data.pkl
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
./train_model_classification.py data_class.pkl
```