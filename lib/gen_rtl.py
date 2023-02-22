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
from pathlib import Path

def gen_rtl (project_name, config, filename, coin):

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
    config['N_TRAIN'] = 2
    config['N_VAL'] = 1
    n_export = config['N_TEST']
    X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test = load_data(config)
    Y_test_pred = coin_model.classify(X_test_lst, coin=coin)
    acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=False)   
    print("Test set accuracy:", acc_test)
    print("Number of minterms:", coin_model.get_minterms_info())
    print('\n>>> Generating RTL...')
    
    Path(out_dir+'/rtl/').mkdir(parents=True, exist_ok=True)
    if coin:
        full_path = out_dir+'/rtl/coin/'
    else:
        full_path = out_dir+'/rtl/lw/'
    
    Path(full_path).mkdir(parents=True, exist_ok=True)
    
    coin_model.export2verilog(full_path, X_test_lst[0:n_export,:], Y_test_pred[0:n_export], coin=coin, export_data=False)
    coin_model.export2python(full_path, coin=coin)
    
    print( "\n\n--- Ensembles evaluation executed in %.02f seconds ---" % (time.time() - start_time))   
    
if __name__ == "__main__":
    sys.path.insert(0, '../lib/')
    from load_config import load_config
    if len(sys.argv) > 3:
        project_name = sys.argv[1]
        filename = sys.argv[2]
        coin = bool(sys.argv[3])
    else:
        project_name = 'mnist'    
        filename = '../mnist/out/lw_2023-02-22_09-09-26_0.pkl'
        coin = False
    config = load_config('../'+project_name)  
    
    gen_rtl(project_name, config, filename, coin)      
