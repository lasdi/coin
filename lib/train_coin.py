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
from bnn_mlp import bnn_mlp
from keras.utils import np_utils
# from sklearn.metrics import accuracy_score
import pandas as pd
# from load_config import load_config
# import shutil
import threading
import copy

def train_thread(m, config, filename,X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test):
       
    global lw_accs, lw_accs_float, lw_minterms, lw_filenames
    global g_weights
    DO_PLOTS = config['DO_PLOTS']
    VERBOSE = config['VERBOSE']
    DO_HAMMING = config['DO_HAMMING']
    CLASSES = config['CLASSES']
   
    
    write2file('> Started model %d ' %(m))    
    
    with open(filename, 'rb') as inp:
        mWisard = pickle.load(inp)
        
    # Y_test_pred = mWisard.classify(X_test_lst)
    # acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=False)    
    
    # word_cnt, max_value = mWisard.get_mem_info()
    # minterms_cnt = mWisard.get_minterms_info()
    # write2file('> Pre-BC test ACC: %f / Number of words: %d / Number of minterms: %d' % (acc_test, word_cnt, minterms_cnt))    

    # These lines below shouldn't be needed but every second loop
    # it stores information from previous loop even if
    # the object is deleted. I may not know python enough.
    mWisard.bc_encoded_rams = []
    mWisard.model_bc = {}
    mWisard.bc_total_minterms = 0
    mWisard.bc_weights = 0  
    minterms_cnt = mWisard.get_minterms_info()
    # write2file('\nNumber of minterms: '+str(minterms_cnt))        
    lw_minterms[m] = minterms_cnt
    
    if DO_HAMMING:
        # write2file('\n>>> Generate hamming model...')
        mWisard.gen_hamming_model()
        mWisard.model = mWisard.model_hamm

             
    X_train_bc = mWisard.gen_bc_encode(X_train_lst, hamming=DO_HAMMING)
    # X_val_bc = mWisard.gen_bc_encode(X_val_lst, hamming=DO_HAMMING)
    X_test_bc = mWisard.gen_bc_encode(X_test_lst, hamming=DO_HAMMING)
    
    # write2file(">>> Starting BNN training...")
    model_bc, history = bnn_mlp(config, X_train_bc, Y_train, X_test_bc, Y_test)

    # X_test_bc = mWisard.gen_bc_encode(X_test_lst, hamming=DO_HAMMING)
    Y_test_bc = np_utils.to_categorical(Y_test, len(CLASSES)) * 2 - 1
    
    # score = model_bc.evaluate(X_test_bc, Y_test_bc, verbose=0)
    # write2file('>>> BC Test score:', score[0])
    # write2file('>>> BC Test accuracy:', score[1])
    
    weights = model_bc.get_weights()
    # g_weights = copy.deepcopy(weights)
    w_thrd = 0.2*np.max(weights[0])
    for i in range (weights[0].shape[0]):
        for j in range (weights[0].shape[1]):
            weights[0][i,j] = 1 if weights[0][i,j] >= w_thrd else -1
    
    model_bc.set_weights(weights)
    score = model_bc.evaluate(X_test_bc, Y_test_bc, verbose=int(VERBOSE))
    # write2file('>>> BC clipped Test accuracy: ' +str(score[1]))
    lw_accs_float[m] = score[1]
    # Y_test_bc_pred = model_bc.predict_on_batch(X_test_bc)

    ################## BC-Wisard ############################
    
    # write2file('\n>>> Evaluating post-bc test set...')
    mWisard.create_model_from_bc (weights)

    Y_test_pred = mWisard.classify(X_test_lst, hamming=DO_HAMMING, bc=True)
    acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=DO_PLOTS)  
    # write2file('>>> Post-BNN Test set accuracy: ' +str(acc_test)) 
    # acc_test2 = accuracy_score(Y_test, Y_test_pred)
    # write2file('>>> Post-BNN Test set accuracy2: ' +str(acc_test2))
    del X_test_bc    
    lw_accs[m] = acc_test
    
    ################# Save results ##########################
    coin_filename = filename.replace('lw','coin')
    with open(coin_filename, 'wb') as outp:
        pickle.dump(mWisard, outp, pickle.HIGHEST_PROTOCOL)
    
    # import copy
    # mWisard.model = copy.deepcopy(mWisard.model_bc)
    # write2file("> COIN ones count: %d" % (mWisard.get_minterms_info(bits_on=True)))
    
    del mWisard
    
    lw_filenames[m] = coin_filename
    if DO_PLOTS:
        plt.figure(0)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'],'r')
        plt.show()
            
    write2file('> Finished model %d ' %(m))   
    
def train_coin(project_name, config):

    global lw_accs, lw_accs_float, lw_minterms, lw_filenames
    global g_weights
    SEED = config['SEED']
    N_THREADS = config['N_THREADS']
    ADDRESS_SIZE = config['ADDRESS_SIZE']
    N_TRAIN_MODELS = config['N_TRAIN_MODELS']

    np.random.seed(SEED)
    
    out_dir = config['PROJ_DIR']+'/out/'
    
    try:
        os.remove("./log.txt")
    except OSError:
        pass
    
    start_time = time.time()
    full_log = "--- RUNNING COIN TRAINING: "+project_name+" ---\n"
    full_log += "\n> seed = " + str(SEED) 
    full_log += "\n> Address Size = " + str(ADDRESS_SIZE) 
    datetime_string = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d_%H-%M-%S')
    full_log += "\nStarting at: "+datetime_string+"\n"
    write2file( full_log)
    
    
    from load_data import load_data
    config['N_VAL'] = 0
    X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test = load_data(config)
    
    write2file("Train set input: "+str(X_train_lst.shape))
    write2file("Train set output: "+str(Y_train.shape))
    write2file("Test set input: "+str(X_test_lst.shape))
    write2file("Test set output: "+str(Y_test.shape))
  
    cmd_lst = 'ls -1 '+out_dir+'lw*.pkl | sort > '+out_dir+'lw_lst.txt'
    os.system(cmd_lst)    
    files_list = open(out_dir+'lw_lst.txt', 'r')
    filenames = files_list.readlines()
    
  
    if N_TRAIN_MODELS!=-1:
        n_lw_models = min(N_TRAIN_MODELS,len(filenames))
    else:
        n_lw_models = len(filenames)

    lw_accs = [None]*n_lw_models
    lw_accs_float = [None]*n_lw_models
    lw_minterms = [None]*n_lw_models  
    lw_filenames = [None]*n_lw_models  
    
    for model_i in range(0,n_lw_models, N_THREADS):
      n_threads_r = min(N_THREADS, n_lw_models-model_i)
      print('>>>> model_i %d - %d / %d <<<<<' %(model_i+1,model_i+n_threads_r, n_lw_models))
      threads = []
      for t in range(n_threads_r):
          m = model_i+t
          filename = filenames[m].replace('\n','')
          new_thread = threading.Thread(target=train_thread, args=(m,config,filename,X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test))
          threads.append(new_thread)
          new_thread.start()
      
      for t in range(n_threads_r):
          threads[t].join()
        
    print('>>> Training completed.')    
    # print(g_weights[0].shape)
    
    df = pd.DataFrame()
    df['filename'] = lw_filenames
    df['n_minterms'] = lw_minterms
    df['acc_float'] = lw_accs_float
    df['acc_fixed'] = lw_accs
    df.to_csv(out_dir+"/res_coin_"+datetime_string+".csv")
    
    write2file( "\n\n--- COIN training Executed in %.02f seconds ---" % (time.time() - start_time))   
    os.system("mv ./log.txt "+out_dir+"/log_coin_"+datetime_string+".txt")
    os.system("cp "+config['PROJ_DIR']+"/config.py "+out_dir+"/config_coin_"+datetime_string+".py")
    
if __name__ == "__main__":
    sys.path.insert(0, '../lib/')
    from load_config import load_config
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = 'mnist'    
    config = load_config('../'+project_name)    
    np.random.seed(config['SEED'])
    
    train_coin(project_name, config)