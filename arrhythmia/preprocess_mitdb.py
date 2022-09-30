#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:20:18 2022

@author: Anamika, Igor
"""
from matplotlib import pyplot as plt
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def label_encoding(label):
    if label == '+' or label == 'N' or label == 'L' or label == 'R' or label == 'e' or label == 'j':
        # non_ectopic[j] = signal_points
        enc_label = 0
    elif label == 'A' or label == 'a' or label == 'J' or label == 'S':
        # supravent[j] = signal_points
        enc_label = 1
    elif label == 'V' or label == '[' or label == '!' or label == ']' or label == 'E':
        # ventric[j] = signal_points
        enc_label = 2
    elif label == 'F':
        # fusion[j] = signal_points
        enc_label = 3
    elif label == '/' or label == 'f' or label == 'Q':
        # unknown[j] = signal_points
        enc_label = 4
    else:
        # oops[j] = signal_points    
        enc_label = 5
    return enc_label

def count_classes (Y, name):
    print("\n> "+name+" class count: ", end='')
    for i in range(5):
        print(str(np.count_nonzero(Y==i))+', ',end='')    
    print(' ')
    
def preprocess_patient (record):
    ecg, fields = wfdb.rdsamp(record, channels=[0])
    annotation = wfdb.rdann(record, 'atr')
    # Store all the physician labels for each detected R peak in labels[] (use later for classifying each beat as
    # one of the five AAMI classes based on which label the R peak is associated with)
    labels = annotation.__dict__['symbol']
    qrs_peaks = annotation.sample
    
    # preprocessed_ecg = np.squeeze(ecg)
    
    # Filtering 
    fs = 360
    cutoff = 2
    nyq = 0.5*fs
    cutoff_norm = cutoff/nyq
    b, a = butter(5, cutoff_norm, 'highpass')
    preprocessed_ecg = filtfilt(b, a, np.squeeze(ecg))
    
    # Normalization
    preprocessed_ecg /= np.max(preprocessed_ecg[qrs_peaks[1]-5:qrs_peaks[-1]-93])
    
    # # Ploting
    # plt.plot(preprocessed_ecg)
    # l = np.ones((len(qrs_peaks)))
    # plt.plot(qrs_peaks, l, 'o')
    # plt.show()
    
    # Split into segments
    X_data = np.zeros((0, 187))
    Y_data = []
    for i in range (2,len(labels)-1):
       label = label_encoding(labels[i])
       if label<=4: # To remove unlabeled cases
           chunk =  preprocessed_ecg[qrs_peaks[i]-93:qrs_peaks[i]+94]  
           X_data = np.vstack([X_data, chunk.reshape(1,-1)])
           Y_data.append(label)
    Y_data = np.array(Y_data, dtype=int)
    
    return X_data, Y_data

def gen_mitdb(train_ratio = 0.7):
 
    patients = open('mitdb_list.txt').readlines()
    shuffled_i = np.arange(len(patients))
    np.random.shuffle(shuffled_i)    
    n_patients_train = int(train_ratio*len(patients))
       
    # Train selection
    X_train = np.zeros((0, 187))
    Y_train = np.zeros((0, 1))
    for i in range(n_patients_train):
        patient = patients[shuffled_i[i]]
        record = 'mitdb/'+patient.replace("\n", "")
        X_data, Y_data = preprocess_patient (record)
        X_train = np.vstack([X_train, X_data])
        Y_train = np.vstack([Y_train, Y_data. reshape(-1,1)])
    Y_train = np.squeeze(Y_train)
    
    with open("./mitdb/x_train.pkl", 'wb') as outp:
        pickle.dump(X_train, outp, pickle.HIGHEST_PROTOCOL)
    with open("./mitdb/y_train.pkl", 'wb') as outp:
        pickle.dump(Y_train, outp, pickle.HIGHEST_PROTOCOL)        

    count_classes (Y_train, "Train")

    # test selection
    X_test = np.zeros((0, 187))
    Y_test = np.zeros((0, 1))
    for i in range(n_patients_train, len(patients)):
        patient = patients[shuffled_i[i]]
        record = 'mitdb/'+patient.replace("\n", "")
        X_data, Y_data = preprocess_patient (record)
        X_test = np.vstack([X_test, X_data])
        Y_test = np.vstack([Y_test, Y_data. reshape(-1,1)])
    Y_test = np.squeeze(Y_test)
            
    with open("./mitdb/x_test.pkl", 'wb') as outp:
        pickle.dump(X_test, outp, pickle.HIGHEST_PROTOCOL)
    with open("./mitdb/y_test.pkl", 'wb') as outp:
        pickle.dump(Y_test, outp, pickle.HIGHEST_PROTOCOL)        
         
    count_classes (Y_test, "Test")    

def load_mitdb(intra_patients = False, n_max_class=60000):
    with open('mitdb/x_train.pkl', 'rb') as inp:
        X_train = pickle.load(inp)    
    with open('mitdb/y_train.pkl', 'rb') as inp:
        Y_train = pickle.load(inp)    
    with open('mitdb/x_test.pkl', 'rb') as inp:
        X_test = pickle.load(inp)    
    with open('mitdb/y_test.pkl', 'rb') as inp:
        Y_test = pickle.load(inp)        
    
    # Normalization
    X_train -= np.min(X_train,axis=1).reshape(-1,1)
    X_train /= np.max(X_train,axis=1).reshape(-1,1)
    X_test -= np.min(X_test,axis=1).reshape(-1,1)
    X_test /= np.max(X_test,axis=1).reshape(-1,1)       

    # Intra-patient analisys shuffles everything
    if intra_patients:
        X_data = np.vstack([X_train, X_test])
        Y_data = np.vstack([Y_train.reshape(-1,1), Y_test.reshape(-1,1)])        
        X_train, X_test, Y_train, Y_test = train_test_split(X_data,Y_data,test_size=0.2)       

    count_classes (Y_train, "Train")

    # Removind excess of normal class
    normals = np.where(Y_train==0)[0]
    X_train = np.delete(X_train, normals[n_max_class:], axis=0)
    Y_train = np.delete(Y_train, normals[n_max_class:])
    
    ## Augmentation
    #oversample = SMOTE()
    #X_train, Y_train = oversample.fit_resample(X_train, Y_train)

    count_classes (Y_train, "Train-Balanced")
    count_classes (Y_test, "Test")
    
    return X_train[:,0:186], Y_train, X_test[:,0:186], Y_test


if __name__ == '__main__':
    np.random.seed(56)
    gen_mitdb()        
