#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 08:15:36 2022

@author: igor
"""

import sys
import pickle
import numpy as np
import time
import datetime
from wisard_tools import eval_predictions

def prune_model (project_name, config, filename):

    out_dir = config['PROJ_DIR']+'/out/'
    CLASSES = ['CLASSES']  
 
    start_time = time.time()
    full_log = "--- GENERATING COIN RTL: "+project_name+" ---\n"
    datetime_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
    full_log += "\nStarting at: "+datetime_string+"\n"
    print( full_log)
       
    coin_models = []    
    with open(filename, 'rb') as inp:
        # coin_model = pickle.load(inp)
        coin_models.append(pickle.load(inp))
        
    coin_model = coin_models[0]

    from load_data import load_data   
    config['N_TRAIN'] = 0
    config['N_VAL'] = 0
    n_export = config['N_TEST']
    X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test = load_data(config)
    Y_test_pred = coin_model.classify(X_test_lst, coin=True)
    acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=False)   
    print("Test set accuracy:", acc_test)
    print("Number of minterms:", coin_model.get_minterms_info())
   
    coin_model.pruning(X_test_lst, Y_test, 0.8)

    Y_test_pred = coin_model.classify(X_test_lst, coin=True)
    acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=False)   
    print("Test set accuracy:", acc_test)
    print("Number of minterms:", coin_model.get_minterms_info())
    
    print( "\n\n--- Ensembles evaluation executed in %.02f seconds ---" % (time.time() - start_time))   
       
    
if __name__ == "__main__":
    sys.path.insert(0, '../lib/')
    from load_config import load_config
    if len(sys.argv) > 2:
        project_name = sys.argv[1]
        filename = sys.argv[2]
    else:
        project_name = 'mnist'    
        filename = '../mnist/out/coin_2022-09-09_12-22-14_0.pkl'
    config = load_config('../'+project_name)  
    
    prune_model(project_name, config, filename)      
