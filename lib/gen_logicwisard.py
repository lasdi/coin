#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:22:17 2021

@author: igor
"""

import sys
# sys.path.insert(0, './lib/')
import logicwisard as lwsd

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import datetime
from wisard_tools import eval_predictions, write2file 

from load_config import load_config


def gen_logicwisard(project_name, config):

    # sys.path.insert(0, './'+project_name)
    
    SEED = config['SEED']
    ADDRESS_SIZE = config['ADDRESS_SIZE']
    THERMO_RESOLUTION = config['THERMO_RESOLUTION']
    MIN_THRESHOLD = config['MIN_THRESHOLD']
    MAX_THRESHOLD = config['MAX_THRESHOLD']
    ACC_DELTA = config['ACC_DELTA']
    ACC_PATIENCE = config['ACC_PATIENCE']
    N_GEN_MODELS = config['N_GEN_MODELS']
    N_SEL_MODELS = config['N_SEL_MODELS']
    DO_HAMMING = config['DO_HAMMING']
    SORT_MODELS_BY = config['SORT_MODELS_BY']
    CLASSES = config['CLASSES']
    DO_PLOTS = config['DO_PLOTS']
    
    np.random.seed(SEED)
    
    out_dir = config['PROJ_DIR']+'/out/'
    
    try:
        os.remove("./log.txt")
    except OSError:
        pass

    
    start_time = time.time()
    full_log = "--- RUNNING WISARD TRAINING FOR "+project_name+" ---\n"
    full_log += "\n> seed = " + str(SEED) 
    full_log += "\n> address size = " + str(ADDRESS_SIZE) 
    full_log += "\n> thermometer resolution = " + str(THERMO_RESOLUTION) 
    full_log += "\n> N_GEN_MODELS = " + str(N_GEN_MODELS) 
    datetime_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
    full_log += "\nStarting at: "+datetime_string+"\n"
    write2file( full_log)
    
    
    from load_data import load_data
    X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test = load_data(config)
    
    write2file("Train set input: "+str(X_train_lst.shape))
    write2file("Train set output: "+str(Y_train.shape))
    write2file("Val set input: "+str(X_val_lst.shape))
    write2file("Val set output: "+str(Y_val.shape))
    write2file("Test set input: "+str(X_test_lst.shape))
    write2file("Test set output: "+str(Y_test.shape))
    
    bin_acc = []
    bin_acc_test = []
    bin_mem = []
    bin_minterms = []
    bin_models = []
    
    for epoch in range(N_GEN_MODELS):
        write2file('>>>> EPOCH %d <<<<<' %(epoch))
        
           
        write2file('>>> Training Wisard...')
        mWisard = lwsd.logicwisard(CLASSES, ADDRESS_SIZE, MIN_THRESHOLD, MAX_THRESHOLD)
        mWisard.fit(X_train_lst, Y_train)
        
        write2file('>>> Evaluate model...')
        word_cnt, max_value = mWisard.get_mem_info()
        Y_test_pred = mWisard.classify(X_test_lst)    
        acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=False)       
        write2file('> Pre-bin Test ACC: %f / Number of words: %d ' % (acc_test, word_cnt))       
        
        ###### Binarization ######
        write2file('>>> Searching for binarization threshold...')
        best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt = mWisard.find_threshold(X_val_lst, Y_val, ACC_DELTA, ACC_PATIENCE)
        # write2file('Max results => acc: %f / threshold: %d / word_cnt: %d' % (max_acc, max_threshold, max_cnt))
        # write2file('best results => acc: %f / threshold: %d / word_cnt: %d' % (best_acc, best_threshold, best_cnt))
        
        mWisard.binarize_model()
        
        Y_test_pred = mWisard.classify(X_test_lst)
        acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=False)    
        
        word_cnt, max_value = mWisard.get_mem_info()
        minterms_cnt = mWisard.get_minterms_info()
        write2file('> Post-bin test ACC: %f / Number of words: %d / Number of minterms: %d' % (acc_test, word_cnt, minterms_cnt))
        
        bin_acc.append(max_acc)
        bin_acc_test.append(acc_test)
        bin_mem.append(word_cnt)
        bin_minterms.append(minterms_cnt)
        bin_models.append(mWisard)
     
    # Plot search Results
    if DO_PLOTS:
        plt.plot(bin_minterms, bin_acc_test,'g^')
        plt.xlabel('Number of minterms')
        plt.ylabel('Test set accuracy')
        plt.savefig(out_dir+'/model_search_minterms.pdf')
        plt.show()
       
    # Sorts by accuracy
    if SORT_MODELS_BY=='accuracy':
        sorted_i = np.argsort(bin_acc_test)
    else: #size
        sorted_i = np.argsort(bin_minterms)
    
    # Selecting the best models
    sel_i = []    
    sel_sizes = []
    for s in range(len(sorted_i)-N_SEL_MODELS,len(sorted_i)):
        i = sorted_i[s]
        sel_i.append(i)
        sel_sizes.append(bin_minterms[i])    
    
    # Save the selected models
    lw_models = []
    lw_accs = []
    lw_minterms = []
    total_minterms = 0
    write2file(">>> Selected models: " )
    for m in range(len(sel_i)):
        lw_models.append(bin_models[sel_i[m]])
        lw_accs.append(bin_acc_test[sel_i[m]])
        lw_minterms.append(bin_minterms[sel_i[m]])
        write2file("> Model %d (acc/minterms): %f / %d " % (m, lw_accs[m], lw_minterms[m]))
        total_minterms+=bin_minterms[sel_i[m]]
        with open(out_dir+'/lw_'+datetime_string+'_'+str(m)+'.pkl', 'wb') as outp:
            pickle.dump(lw_models[m], outp, pickle.HIGHEST_PROTOCOL)
    
    write2file("\n Total # minterms: %d" %(total_minterms))
      
    write2file( "\n\n--- Executed in %.02f seconds ---" % (time.time() - start_time))    
    os.system("mv ./log.txt "+out_dir+"/log_lwisard.txt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = 'mnist'    
    config = load_config('./')    
    
    gen_logicwisard(project_name, config)