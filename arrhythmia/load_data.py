#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:11:08 2022

@author: igor
"""
from project_tools import wisard_data_encode, mnist_data_encode_b, mnist_data_encode_t, mnist_data_encode_z, mnist_data_noencode
from keras.datasets import mnist
import numpy as np
from data_augment import gen_data_raw
from binarization import exponential_thermometer
from exp2_encode import exp2_encode
from pandas import read_csv
from numpy import dstack
import pickle
from preprocess_mitdb import load_mitdb


    

def load_data (config):
    ADDRESS_SIZE = config['ADDRESS_SIZE']
    THERMO_RESOLUTION = config['THERMO_RESOLUTION']
    DO_HAMMING = config['DO_HAMMING']
    DO_AUGMENTATION = config['DO_AUGMENTATION']
    AUGMENT_RATIO = config['AUGMENT_RATIO']

    N_TRAIN = config['N_TRAIN']
    N_VAL = config['N_VAL']
    N_TEST = config['N_TEST']
    N_TRAIN -= N_VAL
    
    
    X_train, Y_train, X_test, Y_test = load_mitdb(intra_patients=True, n_max_class=int(N_TRAIN/5))        


    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    bits_in = 12
    X_train *= 2**bits_in - 1
    X_test *= 2**bits_in - 1
    X_train = X_train.astype(int)
    X_test = X_test.astype(int)
    X_train = np.expand_dims(X_train, 2)
    X_test = np.expand_dims(X_test, 2)    
        
    # Shuffle train set
    n_shuffled = np.arange(X_train.shape[0])
    np.random.shuffle(n_shuffled)
    X_train = X_train[n_shuffled,:,:]
    Y_train = Y_train[n_shuffled]

    if N_TRAIN>0:
        # Split data according to configuration
        X_val = X_train[0:N_VAL,:]
        Y_val = Y_train[0:N_VAL]
        n_train_a = len(Y_train)-N_VAL
        n_train_a = n_train_a if N_TRAIN==-1 else min(N_TRAIN, n_train_a)
        X_train = X_train[N_VAL:N_VAL+n_train_a,:]
        Y_train = Y_train[N_VAL:N_VAL+n_train_a]    
  
        print('>>> Encoding train set...')
        X_train_lst = mnist_data_encode_t(X_train, 0,2**bits_in - 1,THERMO_RESOLUTION)
        # X_train_lst = exp2_encode(X_train.reshape(X_train.shape[0],-1), bits_in, THERMO_RESOLUTION)
        
        X_train_lst = X_train_lst.astype(int)
        Y_train = Y_train.astype(int)
        
        if N_VAL>0:
            print('>>> Encoding val set...')
            X_val_lst = mnist_data_encode_t(X_val, 0,2**bits_in - 1,THERMO_RESOLUTION)
            # X_val_lst = exp2_encode(X_val.reshape(X_val.shape[0],-1), bits_in, THERMO_RESOLUTION)
            
            X_val_lst = X_val_lst.astype(int)
            Y_val = Y_val.astype(int)
        else:
            X_val_lst = []
            Y_val = []
            
    else:
        X_train_lst = []
        Y_train = []
        X_val_lst = []
        Y_val = []
    
    
    n_test_a = len(Y_test) if N_TEST==-1 else min(N_TEST, len(Y_test))
    X_test = X_test[0:n_test_a,:]
    Y_test = Y_test[0:n_test_a]    
    print('>>> Encoding test set...')
    X_test_lst = mnist_data_encode_t(X_test, 0,2**bits_in - 1,THERMO_RESOLUTION)
    # X_test_lst = exp2_encode(X_test.reshape(X_test.shape[0],-1), bits_in, THERMO_RESOLUTION)
    
    X_test_lst = X_test_lst.astype(int)
    Y_test = Y_test.astype(int)

    return X_train_lst, Y_train, X_val_lst, Y_val, X_test_lst, Y_test