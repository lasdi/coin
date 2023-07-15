<img src=./coin.png width=300 />
COmbinational Intelligent Networks

# Usage

All COIN source codes are in `lib/` directory. Dataset specific codes are in `<dataset_dir>/` directory. 
This version comes with an implementation of MNIST dataset classification whose files are in `mnist/` directory.

## Single Model Example
From `mnist/` directory, run a simple COIN training for MNIST dataset as follows:
```
python ./run_single.py
```

This example trains a single COIN model using address size of 16 and thermometer resolution of 2 trained for 5 epoches. 
This humble configuration aims to cope with limited memory machines (8GB or less) and achives above 96% accuracy.

If you have enough memory, change the thermometer resolution and the number of epoches on the configuration script `mnist/config.py`. 
This script is where all COIN parameters are set. Check it out for detailed description of each parameter.


## Ensemble example
Train and evaluate an ensemble of 4 COIN models for MNIST dataset running the following command from `mnist/` folder:
```
python ./run_ensemble.py
```
This uses two models with address size of 14 and two models with address size of 16 to produce an ensemble model with 97.1% accuracy. 
Again, thermometer resolution is set to 2 in order to reduce memory requirement.

# Creating a new classifier

One can use the `mnist/` folder as a starting point for a new classifier implementation. All scripts in this folder should be 
adapted to the new dataset and classification problem.




