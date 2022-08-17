#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 08:15:36 2022

@author: igor
"""


import sys
from wisard_tools import eval_predictions, write2file 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime

from multikernel import classify_mk

def eval_ensemble (project_name, config):

    # sys.path.insert(0, './'+project_name)
    SEED = config['SEED']
    DO_PLOTS = config['DO_PLOTS']
    DO_HAMMING = config['DO_HAMMING']
    CLASSES = config['CLASSES']

    np.random.seed(SEED)
    
    out_dir = config['PROJ_DIR']+'/out/'
    
    try:
        os.remove("./log.txt")
    except OSError:
        pass
    
    start_time = time.time()
    full_log = "--- RUNNING COIN TRAINING: "+project_name+" ---\n"
    full_log += "\n> seed = " + str(SEED) 
    datetime_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
    full_log += "\nStarting at: "+datetime_string+"\n"
    write2file( full_log)
    
    from load_data import load_data   
    X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test = load_data(config)
    
    # write2file("Train set input: "+str(X_train_lst.shape))
    # write2file("Train set output: "+str(Y_train.shape))
    # write2file("Val set input: "+str(X_val_lst.shape))
    # write2file("Val set output: "+str(Y_val.shape))
    write2file("Test set input: "+str(X_test_lst.shape))
    write2file("Test set output: "+str(Y_test.shape))

    coin_models = []
    
    cmd_lst = 'ls -1 '+out_dir+'coin*.pkl | sort > '+out_dir+'coin_lst.txt'
    os.system(cmd_lst)    
    files_list = open(out_dir+'coin_lst.txt', 'r')
    filenames = files_list.readlines()
    
    total_minterms = 0
    m=0
    for filename in filenames:
        filename = filename.replace('\n','')
        write2file('>>>> Loading model %d <<<<<' %(m))    
        
        with open(filename, 'rb') as inp:
            coin_models.append(pickle.load(inp))
    
        write2file("Address size: "+str(coin_models[m].address_size))
        Y_test_pred = coin_models[m].classify(X_test_lst, hamming=DO_HAMMING, bc=True)
        acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=False)    
        
        word_cnt, max_value = coin_models[m].get_mem_info()
        minterms_cnt = coin_models[m].get_minterms_info()
        total_minterms += minterms_cnt
        write2file('> Pre-ensemble test ACC: %f / Number of words: %d / Number of minterms: %d' % (acc_test, word_cnt, minterms_cnt))    
        m+=1
    
    write2file('\nTotal number of minterms: '+str(total_minterms))                    
    write2file('\n>>> Evaluating test set...')
    Y_test_pred = classify_mk(coin_models, X_test_lst, Y_test, hamming=DO_HAMMING, bc=True)
    
    acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=DO_PLOTS)    
    write2file(f'\n>>> Test set accuracy: {acc_test:.01%}') 

    write2file( "\n\n--- Ensembles evaluation executed in %.02f seconds ---" % (time.time() - start_time))   
    os.system("mv ./log.txt "+out_dir+"/log_ensemb_"+datetime_string+".txt") 

if __name__ == "__main__":
    sys.path.insert(0, '../lib/')
    from load_config import load_config
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = 'mnist'    
    config = load_config('../'+project_name)  
    
    eval_ensemble(project_name, config)      