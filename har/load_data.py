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

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values
 
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded
 
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y
 
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    # trainX = trainX + np.abs(np.min(trainX))
    # testX = testX + np.abs(np.min(testX))
    # trainX = trainX/(np.min(trainX) + np.max(trainX))
    # testX = testX/(np.min(testX) + np.max(testX))
    

    for i in range (9):
        trainX[:,i,:] = trainX[:,i,:]-(np.mean(trainX[:,i,:]))
        testX[:,i,:] = testX[:,i,:]-(np.mean(testX[:,i,:]))        
        trainX[:,i,:] = trainX[:,i,:]/(np.var(trainX[:,i,:]))
        testX[:,i,:] = testX[:,i,:]/(np.var(testX[:,i,:]))
    trainX = np.power(trainX,2)
    testX = np.power(testX,2)
    
    trainy = trainy - 1
    testy = testy - 1
    return trainX, trainy, testX, testy

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
    
    # Import from har data set
    X_train, Y_train, X_test, Y_test = load_dataset()
    Y_train = Y_train.reshape(-1)
    Y_test = Y_test.reshape(-1)
    bits_in = 12
    X_train *= 2**bits_in - 1
    X_test *= 2**bits_in - 1
    X_train = X_train.astype(int)
    X_test = X_test.astype(int)
    
    if DO_AUGMENTATION:
        batch_size = 32
        N_TRAIN = int(X_train.shape[0]*AUGMENT_RATIO)       
        X_train, Y_train = gen_data_raw (N_TRAIN, batch_size)
        
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