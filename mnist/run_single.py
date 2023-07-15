#!/usr/bin/env python3

"""
COIN (COmbinational Intelligent Networks) - Single Example

This examples runs a simple MNIST training using the standard 
configurations defined in mnist/config.py (see for details).
Models, results and logs are sent to mnist/out
"""

import sys
sys.path.insert(0, '../lib/')
from pathlib import Path
import os
from load_config import load_config
from load_data import load_data
from eval_imbalanced import eval_imbalanced
from gen_logicwisard import gen_logicwisard
from train_coin import train_coin
import numpy as np
import pickle

# Sets project name
project_name = 'mnist'   

# Loads all configurations from config.py file in project dir
config = load_config('./')
np.random.seed(config['SEED'])

out_dir = config['PROJ_DIR']+'/out/'
CLASSES = config['CLASSES']
DO_PLOTS = config['DO_PLOTS']

# cleans the output directory
os.system('rm -f ./out/*')
os.system('rm -rf ./out/rtl')

Path('./out/').mkdir(parents=True, exist_ok=True)

# Generates and trains 3 LogicWiSARD models, picking up 
# the most accurate.
config['N_THREADS'] = 1
gen_logicwisard(project_name, config)

# Convert LogicWiSARD to BNN, train it, and then convert to COIN
config['N_THREADS'] = 1
train_coin(project_name, config)

print(">>> COIN test set evaluation...")
cmd_lst = 'ls -1 '+out_dir+'coin*.pkl | sort > '+out_dir+'coin_lst.txt'
os.system(cmd_lst)
files_list = open(out_dir+'coin_lst.txt', 'r')
filenames = files_list.readlines()
X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test = load_data(config)


for filename in filenames:
    with open(filename.replace('\n',''), 'rb') as inp:
        coin_model = pickle.load(inp)
    Y_test_pred = coin_model.classify(X_test_lst, coin=True)

    sensitivities, specificities, accuracy, auc = eval_imbalanced(Y_test, Y_test_pred, CLASSES, do_plot=DO_PLOTS)
