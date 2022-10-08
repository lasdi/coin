#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:11:08 2022

@author: igor
"""
from project_tools import wisard_data_encode, mnist_data_encode_b, mnist_data_encode_t, mnist_data_encode_z, mnist_data_noencode
# from keras.datasets import mnist
import numpy as np
# from data_augment import gen_data_raw
# from binarization import exponential_thermometer
# from exp2_encode import exp2_encode
# from pandas import read_csv
# from numpy import dstack
import pickle
# from preprocess_mitdb import load_mitdb
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
from feat_extract_chunk import feat_extract_chunk    
import librosa 
import speechpy
from exp2_encode import exp2_encode

def gen_list_files(config, listname):
    fp_files = open(listname, 'r')
    files = fp_files.readlines()
    Y_data = np.zeros(len(files))
    CLASSES = config['CLASSES']
    i = 0
    for file0 in files:
        file = './speech_commands_v0.02/'+file0.replace('\n','')
        file_class = file0[0:file0.find('/')]
        data, fs = librosa.load(file, sr=config['FEAT_FS'], mono=True)
        data /= (np.max(np.abs(data)))  
        if len(data)<config['FEAT_FS']:
            data = np.concatenate([data, np.zeros(config['FEAT_FS']-len(data))])
        X_data_tmp = feat_extract_chunk(config, data[0:config['FEAT_FS']])
        if i==0:
            X_data = np.zeros((len(files), X_data_tmp.shape[0], X_data_tmp.shape[1]))            
        X_data[i,:,:] = X_data_tmp
        Y_data[i] = CLASSES.index(file_class)
        i+=1
        
    with open(listname.replace('list.txt','x.pkl'), 'wb') as outp:
        pickle.dump(X_data, outp, pickle.HIGHEST_PROTOCOL)
    with open(listname.replace('list.txt','y.pkl'), 'wb') as outp:
        pickle.dump(Y_data, outp, pickle.HIGHEST_PROTOCOL)  
        
    print(X_data.shape)
    print(Y_data.shape)        


def gen_data ():
    feat_config = {}
    # Standard sample rate for segmentation and feature extraction. All signals are resampled to this rate.
    feat_config["FEAT_FS"] = 16000
    # Selected features to be used. The line below contain all features implemented so far
    #ALL_FEAT = ('mfb', 'raw_mfcc', 'delta0_mfcc')
    feat_config["SELECTED_FEAT"] = ('raw_mfcc')
    # Frame window in seconds
    feat_config["T_FRAME"] = 0.050
    # number of overlap samples in power estimator
    feat_config["T_FRAME_OVERLAP"] = feat_config["T_FRAME"] * 0.5
    # Number of triangular bandpass filters to compute the mel frequency spectrum.
    feat_config["N_MELS_BPF"] = 40
    # Number of MFCC coefficients
    feat_config["N_MFCC"] = 13
  
    fp_classes = open('./speech_commands_v0.02/classes.txt', 'r')
    classes = fp_classes.readlines()
    for i in range(len(classes)):
        classes[i] = classes[i].replace('\n','')
    feat_config["CLASSES"] = classes
    
    # gen_list_files(feat_config, './speech_commands_v0.02/testing_mini_list.txt')
    gen_list_files(feat_config, './speech_commands_v0.02/testing_list.txt')
    gen_list_files(feat_config, './speech_commands_v0.02/validation_list.txt')
    gen_list_files(feat_config, './speech_commands_v0.02/training_list.txt')

def sel_classes (ALL_CLASSES, CLASSES, X_data, Y_data):
    Y_sel = []
    Y_sel_n = []
    ind_t = []
    ind_n = []
    
    for i in range(X_data.shape[0]):
        if ALL_CLASSES[int(Y_data[i])] in CLASSES:
            ind_t.append(i)
            Y_sel.append(CLASSES.index(ALL_CLASSES[int(Y_data[i])]))
        else:
            ind_n.append(i)
            Y_sel_n.append(len(CLASSES)-1)
    
   
    X_sel = X_data[ind_t]
    X_sel_n = X_data[ind_n]
    
    Y_sel = np.array(Y_sel)
    Y_sel_n = np.array(Y_sel_n)
    
    # Shuffle train set
    n_shuffled = np.arange(X_sel_n.shape[0])
    np.random.shuffle(n_shuffled)
    X_sel_n = X_sel_n[n_shuffled,:,:]
    Y_sel_n = Y_sel_n[n_shuffled] 
    # remove excess
    X_sel_n = X_sel_n[0:int(X_sel.shape[0]/(len(CLASSES)-1)), :, :]
    Y_sel_n = Y_sel_n[0:X_sel_n.shape[0]]
    # concatenate
    X_sel = np.vstack([X_sel, X_sel_n])
    Y_sel = np.concatenate([Y_sel, Y_sel_n])
    return X_sel, Y_sel       
    
def load_data (config, do_encoding=True):
    THERMO_RESOLUTION = config['THERMO_RESOLUTION']
    CLASSES = config['CLASSES']
    
    fp_classes = open('./speech_commands_v0.02/classes.txt', 'r')
    ALL_CLASSES = fp_classes.readlines()
    for i in range(len(ALL_CLASSES)):
        ALL_CLASSES[i] = ALL_CLASSES[i].replace('\n','')
    
        
    with open('./speech_commands_v0.02/testing_x.pkl', 'rb') as inp:
        X_test = pickle.load(inp)            
    with open('./speech_commands_v0.02/testing_y.pkl', 'rb') as inp:
        Y_test = pickle.load(inp)  
    X_test, Y_test = sel_classes (ALL_CLASSES, CLASSES, X_test, Y_test)
    
    with open('./speech_commands_v0.02/validation_x.pkl', 'rb') as inp:
        X_val = pickle.load(inp)    
    with open('./speech_commands_v0.02/validation_y.pkl', 'rb') as inp:
        Y_val = pickle.load(inp)         
    X_val, Y_val = sel_classes (ALL_CLASSES, CLASSES, X_val, Y_val)        
    
    with open('./speech_commands_v0.02/training_x.pkl', 'rb') as inp:
        X_train = pickle.load(inp)    
    with open('./speech_commands_v0.02/training_y.pkl', 'rb') as inp:
        Y_train = pickle.load(inp)     
    X_train, Y_train = sel_classes (ALL_CLASSES, CLASSES, X_train, Y_train)        
    
    
    if do_encoding:
        nbits = 12
        all_min = 0
        all_max = 2**nbits - 1
        
        # X_train -= np.min(X_train); X_train *= all_max/np.max(X_train);
        # X_val -= np.min(X_val); X_val *= all_max/np.max(X_val);
        # X_test -= np.min(X_test); X_test *= all_max/np.max(X_test);
        
        # X_train = 10**X_train; X_train *= all_max/np.max(X_train);
        # X_val = 10**X_val; X_val *= all_max/np.max(X_val);
        # X_test = 10**X_test; X_test *= all_max/np.max(X_test);
        
        for i in range(X_val.shape[0]):
            x_tmp = speechpy.processing.cmvn(X_val[i,:,:].T).T
            x_tmp -= np.min(x_tmp, axis=0).reshape(1,-1)
            x_tmp *= all_max/np.max(x_tmp, axis=0).reshape(1,-1)
            X_val[i,:,:] = x_tmp
        for i in range(X_test.shape[0]):
            x_tmp = speechpy.processing.cmvn(X_test[i,:,:].T).T
            x_tmp -= np.min(x_tmp, axis=0).reshape(1,-1)
            x_tmp *= all_max/np.max(x_tmp, axis=0).reshape(1,-1)
            X_test[i,:,:] = x_tmp
        for i in range(X_train.shape[0]):
            x_tmp = speechpy.processing.cmvn(X_train[i,:,:].T).T
            x_tmp -= np.min(x_tmp, axis=0).reshape(1,-1)
            x_tmp *= all_max/np.max(x_tmp, axis=0).reshape(1,-1)
            X_train[i,:,:] = x_tmp
            
        X_train = exp2_encode(X_train.reshape(X_train.shape[0],-1), nbits, THERMO_RESOLUTION)
        X_val = exp2_encode(X_val.reshape(X_val.shape[0],-1), nbits, THERMO_RESOLUTION)
        X_test = exp2_encode(X_test.reshape(X_test.shape[0],-1), nbits, THERMO_RESOLUTION)
        
        # X_train = mnist_data_encode_t(X_train.astype(int), all_min,all_max,THERMO_RESOLUTION)
        # X_val = mnist_data_encode_t(X_val.astype(int), all_min,all_max,THERMO_RESOLUTION)
        # X_test = mnist_data_encode_t(X_test.astype(int), all_min,all_max,THERMO_RESOLUTION)
        X_train = X_train.astype(int)
        Y_train = Y_train.astype(int)      
        X_val = X_val.astype(int)
        Y_val = Y_val.astype(int)    
        X_test = X_test.astype(int)
        Y_test = Y_test.astype(int)        
        
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


if __name__ == "__main__":
    # gen_data()
    config = {}
    config['THERMO_RESOLUTION'] = 2
    config['CLASSES'] = ["down", "go", "left", "no", "off", "on", "right",
                     "stop", "up", "yes", "unknown"]
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(config)