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
from bnn_mlp import bnn_mlp, bnn_mlp_augment
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
import pandas as pd
# from load_config import load_config
import shutil
import threading

def train_thread(m, config, filename,X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test):
       
    from data_augment import gen_data, save_data
    global lw_accs, lw_accs_float, lw_minterms
    
    DO_PLOTS = config['DO_PLOTS']
    THERMO_RESOLUTION = config['THERMO_RESOLUTION']
    BATCH_SIZE = config['BATCH_SIZE']
    DO_HAMMING = config['DO_HAMMING']
    CLASSES = config['CLASSES']
    DO_AUGMENTATION = config['DO_AUGMENTATION']
    N_TRAIN = config['N_TRAIN']
    AUGMENT_RATIO = config['AUGMENT_RATIO']    
    
    write2file('>>>> MODEL %d <<<<<' %(m))    
    
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
    write2file('\nNumber of minterms: '+str(minterms_cnt))        
    lw_minterms[m] = minterms_cnt
    
    if DO_HAMMING:
        write2file('\n>>> Generate hamming model...')
        mWisard.gen_hamming_model()
        mWisard.model = mWisard.model_hamm

    
    if DO_AUGMENTATION==False:            
        X_train_bc = mWisard.gen_bc_encode(X_train_lst, hamming=DO_HAMMING)
        # X_val_bc = mWisard.gen_bc_encode(X_val_lst, hamming=DO_HAMMING)
        X_test_bc = mWisard.gen_bc_encode(X_test_lst, hamming=DO_HAMMING)
        
        write2file(">>> Starting BNN training...")
        model_bc, history = bnn_mlp(config, X_train_bc, Y_train, X_test_bc, Y_test)

    else: # AUGMENTATION
        X_test_bc = mWisard.gen_bc_encode(X_test_lst, hamming=DO_HAMMING)
        do_gen_augmented = True
        if do_gen_augmented:
            try:
                shutil.rmtree("./data")
            except OSError:
                pass        
            os.mkdir("./data")
            
            n_augmented_samples = int(AUGMENT_RATIO*N_TRAIN)
            print('> Generating and saving data...')
            train_ids, train_labels, n_total_samples, input_shape = gen_data (n_augmented_samples, BATCH_SIZE, THERMO_RESOLUTION, mWisard)
            print('> Saving val data...')
            # train_ids, train_labels = save_data (X_train_lst, Y_train_augm, 'train')
            val_ids, val_labels = save_data (X_test_bc, Y_test, 'val')
        else:
            print('> Loading generated data...')
            with open('data/train_ids.pkl', 'rb') as inp:
                train_ids = pickle.load(inp)        
            with open('data/train_labels.pkl', 'rb') as inp:
                train_labels = pickle.load(inp)               
            with open('data/val_ids.pkl', 'rb') as inp:
                val_ids = pickle.load(inp)        
            with open('data/val_labels.pkl', 'rb') as inp:
                val_labels = pickle.load(inp)     
                
            n_total_samples = len(train_ids)
            # xt = np.load('data/' + train_ids[0] + '.npy')
            # input_shape = len(xt)
    
        print ('Number of generated samples: '+str(n_total_samples))
        partition = {}
        partition['train'] = train_ids
        partition['validation'] = val_ids
        labels = train_labels
        labels.update(val_labels)
        
        print(">>> Starting BNN training...")
        model_bc, history = bnn_mlp_augment(config, n_total_samples, input_shape, partition, labels)

    
    # X_test_bc = mWisard.gen_bc_encode(X_test_lst, hamming=DO_HAMMING)
    Y_test_bc = np_utils.to_categorical(Y_test, len(CLASSES)) * 2 - 1
    
    # score = model_bc.evaluate(X_test_bc, Y_test_bc, verbose=0)
    # write2file('>>> BC Test score:', score[0])
    # write2file('>>> BC Test accuracy:', score[1])
    
    weights = model_bc.get_weights()
    for i in range (weights[0].shape[0]):
        for j in range (weights[0].shape[1]):
            weights[0][i,j] = 1 if weights[0][i,j] >= 0 else -1
    
    model_bc.set_weights(weights)
    score = model_bc.evaluate(X_test_bc, Y_test_bc, verbose=1)
    write2file('>>> BC clipped Test accuracy: ' +str(score[1]))
    lw_accs_float[m] = score[1]
    # Y_test_bc_pred = model_bc.predict_on_batch(X_test_bc)

    ################## BC-Wisard ############################
    
    write2file('\n>>> Evaluating post-bc test set...')
    mWisard.create_model_from_bc (weights)
    Y_test_pred = mWisard.classify(X_test_lst, hamming=DO_HAMMING, bc=True)
    acc_test = eval_predictions(Y_test, Y_test_pred, CLASSES, do_plot=DO_PLOTS)  
    write2file('>>> Post-BNN Test set accuracy: ' +str(acc_test)) 
    # acc_test2 = accuracy_score(Y_test, Y_test_pred)
    # write2file('>>> Post-BNN Test set accuracy2: ' +str(acc_test2))
    del X_test_bc    
    lw_accs[m] = acc_test
    
    ################# Save results ##########################
    coin_filename = filename.replace('lw','coin')
    with open(coin_filename, 'wb') as outp:
        pickle.dump(mWisard, outp, pickle.HIGHEST_PROTOCOL)
    del mWisard

    if DO_PLOTS:
        plt.figure(0)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'],'r')
        plt.show()
            

def train_coin(project_name, config):

    global lw_accs, lw_accs_float, lw_minterms
    
    SEED = config['SEED']
    N_THREADS = config['N_THREADS']
    ADDRESS_SIZE = config['ADDRESS_SIZE']
    DO_AUGMENTATION = config['DO_AUGMENTATION']
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
    
    if DO_AUGMENTATION == False:
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
  
    
    df = pd.DataFrame()
    df['n_minterms'] = lw_minterms
    df['acc_float'] = lw_accs_float
    df['acc_fixed'] = lw_accs
    df.to_csv(out_dir+"/res_coin_"+datetime_string+".csv")
    
    write2file( "\n\n--- COIN training Executed in %.02f seconds ---" % (time.time() - start_time))   
    os.system("mv ./log.txt "+out_dir+"/log_coin_"+datetime_string+".txt")
    
    
if __name__ == "__main__":
    sys.path.insert(0, '../lib/')
    from load_config import load_config
    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = 'mnist'    
    config = load_config('../'+project_name)    
    
    train_coin(project_name, config)