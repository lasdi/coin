#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 06:30:12 2021

@author: igor
"""
import numpy as np
from discriminator import discriminator_train, discriminator_eval, discriminator_eval_bc
from wisard_tools import separate_classes, eval_predictions
from hamming import hamming_correction
from keras import backend as K
import math

def square_group_mapping (W,H, w,h):
    N = W*H
    
    inner_mapping = np.arange(w*h)
    np.random.shuffle(inner_mapping)
    
    o_mapping = np.arange(N)
    map_mat = np.reshape(o_mapping, (W,H))
    mapping = np.zeros((0))
    for i in range(int(H/h)):
        for j in range(int(W/w)):
            m_tmp = map_mat[i*h:(i+1)*h,j*w:(j+1)*w]
            m_tmp = np.reshape(m_tmp,(-1,1))
            m_tmp = m_tmp[inner_mapping,0]
            mapping = np.hstack([mapping, m_tmp])
    mapping = np.squeeze(mapping)
    mapping = mapping.astype(int)
    return mapping   

def linear_group_mapping (N, p):
    
    # inner_mapping = np.arange(p)
    # np.random.shuffle(inner_mapping)
    
    mapping = np.arange(N)

    for i in range(0,N,p):
        m_tmp = mapping[i:i+p]
        inner_mapping = np.arange(p)
        np.random.shuffle(inner_mapping)
        m_tmp = m_tmp[inner_mapping]
        mapping[i:i+p] = m_tmp 

    return mapping      

def block_mapping (N, thermo_resolution, block_width):
    
    inner_mapping = np.arange(thermo_resolution * block_width)
    inner_mapping = inner_mapping.reshape(block_width, thermo_resolution).T
    for i in range (inner_mapping.shape[0]):
        np.random.shuffle(inner_mapping[i,:])
    inner_mapping = inner_mapping.reshape(-1)
    # print (inner_mapping)
    
    mapping = np.arange(N)
    mapping = mapping.reshape(-1,len(inner_mapping))
    
    for i in range(mapping.shape[0]):
        mapping[i,:] = mapping[i,inner_mapping]
    
    return mapping.reshape(-1)    
    
def wisard_train (X, Y, classes, address_size):
    
    # Totally random mapping
    # mapping = np.arange(X.shape[1])
    # np.random.shuffle(mapping)
    
    # mapping = square_group_mapping(28,28,4,4)
    # mapping = linear_group_mapping(X.shape[1],address_size)
    
    mapping = block_mapping(X.shape[1],8, address_size)
    
    X_mapped = X[:,mapping]
    # X_mapped = hamming_correction(X_mapped, address_size)
    
    X_class = separate_classes (X_mapped, Y, classes, address_size)
    
    model = {}
    
    for c in range (len(classes)):
        model[classes[c]] = discriminator_train(X_class[classes[c]])
    
    return model, mapping

def wisard_eval_bin (X, model, mapping, classes, address_size, thresholds=[1], hamming=False):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    
    if hamming:
        X_mapped = hamming_correction(X_mapped, address_size)
    
    Y_pred = []
    for b in range(len(thresholds)):
        Y_pred.append([])
    
    # Eval for each sample
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        scores = np.zeros((len(classes)))
        
        # For multiple threshold
        for b in range(len(thresholds)):
            ####### Binarized model ####################
                
            for c in range (len(classes)):
                scores[c] = discriminator_eval(xti.astype(int), model[classes[c]], thresholds[b])
                
            ############################################        
            
            best_class = np.argmax(scores)    
            Y_pred[b].append(best_class)
    
    if len(thresholds)==1:
        Y_pred = Y_pred[0]
    return np.array(Y_pred)

def wisard_eval_bc (X, model, mapping, classes, address_size, threshold=1, hamming=False, bc_weights='',n_minterms=0):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    epsilon = 1e-6
    # For Glorot correction
    bc_h = np.float32(np.sqrt(1.5 / (n_minterms + len(classes))))
    
    # Tau values for thresholding as in FINN
    tau = np.zeros((len(classes)))
    tau_inv = np.ones((len(classes)))
    for c in range (len(classes)):
        gamma = bc_weights[1][c]; mov_mean = bc_weights[3][c]; mov_var = bc_weights[4][c]; beta = bc_weights[2][c];
        tau[c] = mov_mean - (beta/(gamma/np.sqrt(mov_var)))
        tau[c] = math.ceil(tau[c]/bc_h) # Glorot correction
        # This correction is not needed. Here just to show the diff from original paper
        # tau[c] = int((tau[c]+n_minterms)/2) 
        if (gamma/np.sqrt(mov_var+epsilon))<0:
            tau_inv[c] = -1
            
        # print("Tau[%d]: %d - Inv[%d]: %d" %(c, tau[c], c, tau_inv[c]))
        
        
    if hamming:
        X_mapped = hamming_correction(X_mapped, address_size)
    
    Y_pred = []
    
    # Eval for each sample
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        scores = np.zeros((len(classes)))
        
        ####### Binarized model ####################
            
        for c in range (len(classes)):
            scores[c] = discriminator_eval_bc(xti.astype(int), model[classes[c]], threshold)
                       
            # Batch normalization correction 
            # scores[c] *= bc_h # Glorot correction
            # gamma = bc_weights[1][c]; mov_mean = bc_weights[3][c]; mov_var = bc_weights[4][c]; beta = bc_weights[2][c];
            # scores[c] = gamma*(scores[c] - mov_mean)/np.sqrt(mov_var + epsilon) + beta
            
            # Batch normalization correction (thresholding as in FINN)
            scores[c] -= tau[c]
            scores[c] *= tau_inv[c]
        ############################################        
        
        best_class = np.argmax(scores)    
        Y_pred.append(best_class)
    
    
    
    return np.array(Y_pred)


def wisard_eval (X, model, mapping, classes, address_size, min_threshold=1, max_threshold=100, hamming=False):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    
    if hamming:
        X_mapped = hamming_correction(X_mapped, address_size)
    
    Y_pred = []
    
    # Eval for each sample
    blc_cnt = 0
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        scores = np.zeros((len(classes)))    
        
        # Increase threshold until the max or there is no tie 
        for threshold in range(min_threshold,max_threshold+1):
            if(threshold==max_threshold):
                # print('# WARNING max threshold reached: '+str(threshold))
                blc_cnt+=1
            tie = False
            for c in range (len(classes)):
                scores[c] = discriminator_eval(xti.astype(int), model[classes[c]], threshold)
                
                # Compare with the previous                
                for c2 in range (c):
                    if scores[c]==scores[c2]:
                        tie = True
                        break
                if tie==True and threshold<max_threshold:
                    break                    
            # Stops if there is no more ties
            if tie==False:
                break
      
        
        best_class = np.argmax(scores)    
        Y_pred.append(best_class)
    
    # print('%d samples of %d reached the max threshold' % (blc_cnt,n_samples))
    
    return np.array(Y_pred)

def get_above_threshold_count (model, classes, threshold):
    cnt = 0 
    for c in range (len(classes)):  
        for r in range(len(model[classes[c]])):                
            dict_tmp = model[classes[c]][r]
            for a in dict_tmp:
                if dict_tmp[a]>=threshold:
                    cnt += 1
    return cnt

def wisard_find_threshold (X, Y, model, mapping, classes, address_size, min_threshold=1, max_threshold=100, acc_delta=0.001, acc_patience=2, hamming=False):
    max_acc = 0
    maxacc_threshold = 0
    maxacc_cnt = 0
    best_acc = 0
    best_threshold = 0
    best_cnt = 0
    fall_cnt = 0
    
    Y_pred = wisard_eval_bin (X, model, mapping, classes, address_size, thresholds=np.arange(min_threshold,max_threshold+1), hamming=hamming)   
    
    accuracies = []
    word_counts = []
    
    # Compute all accuracies and word counts
    b = 0
    for threshold in range(min_threshold,max_threshold+1):
        # Y_pred = wisard_eval_bin (X, model, mapping, classes, address_size, thresholds=[threshold], hamming=hamming)        
        acc_t = eval_predictions(Y, Y_pred[b], classes, do_plot=False)   
        cnt_t = get_above_threshold_count (model, classes, threshold)
        accuracies.append(acc_t)
        word_counts.append(cnt_t)
        b+=1
    
    # Search for the desired threshold
    b = 0
    for threshold in range(min_threshold,max_threshold+1):    
        if accuracies[b]>=max_acc:
            max_acc = accuracies[b]
            maxacc_threshold = threshold
            maxacc_cnt = word_counts[b]
            fall_cnt = 0
        else:
            fall_cnt += 1
        # print('<b: %d, acc: %.04f> '%(threshold, accuracies[b]))
        if fall_cnt>=acc_patience:
            break
        if accuracies[b]>max_acc-acc_delta:
            best_acc = accuracies[b]
            best_threshold = threshold                
            best_cnt = word_counts[b]
        
        b+=1
        
    return best_acc, best_threshold, best_cnt, max_acc, maxacc_threshold, maxacc_cnt
            
