#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 06:30:12 2021

@author: igor
"""
import numpy as np
from discriminator import discriminator_train, discriminator_eval, discriminator_eval_coin
from wisard_tools import separate_classes, eval_predictions
from keras import backend as K
import math
from scipy.optimize import minimize

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
    mapping = np.arange(X.shape[1])
    np.random.shuffle(mapping)
    
    # mapping = square_group_mapping(28,28,4,4)
    # mapping = linear_group_mapping(X.shape[1],address_size)
    
    # thermo_resolution = 8
    #mapping = block_mapping(X.shape[1],thermo_resolution , 28*28)
    
    X_mapped = X[:,mapping]

    
    X_class = separate_classes (X_mapped, Y, classes, address_size)
    
    model = {}
    
    for c in range (len(classes)):
        model[classes[c]] = discriminator_train(X_class[classes[c]])
    
    return model, mapping


# Slow evaluation. Use wisard_eval_bin_array
def wisard_eval_bin (X, model, mapping, classes, address_size, thresholds=[1]):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]

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

# Faster evaluation function
def wisard_eval_bin_array (X, model, mapping, classes, address_size, thresholds=[1]):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]

    Y_pred = []
    for b in range(len(thresholds)):
        Y_pred.append([])
    
    # Eval for each sample
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        
        # Vector is prepared to be used across classes
        X_single = xti.astype(int)
        # Matrix for scores across classes and thresholds
        scores_classes_thresholds = np.zeros((len(classes),len(thresholds)))
        for c in range (len(classes)):
            class_model = model[classes[c]]
            sel_scores = []
            # Find the RAMs values accessed by the input vector X_single            
            for r in range(len(X_single)):
                if X_single[r] in class_model[r]:      
                    sel_scores.append(class_model[r][X_single[r]])
            sel_scores = np.array(sel_scores)
            # Evaluate for all thresholds at once
            above_thresholds_mtx = sel_scores.reshape(-1,1) >= thresholds.reshape(1,-1)
            above_thresholds_mtx = above_thresholds_mtx.astype(int)
            scores_t = np.sum(above_thresholds_mtx, axis=0)
            scores_classes_thresholds[c,:] = scores_t.reshape(1,-1)
            
        # Find the best classes across thresholds
        scores_thresholds = np.argmax(scores_classes_thresholds, axis=0)
            
        # For multiple threshold
        for b in range(len(thresholds)):            
            Y_pred[b].append(scores_thresholds[b])
    
    if len(thresholds)==1:
        Y_pred = Y_pred[0]
        
    return np.array(Y_pred)

def wisard_eval_coin (X, model, mapping, classes, address_size, threshold=1, coin_weights='',n_minterms=0):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]

    from tau_gen import tau_gen
    tau = tau_gen (coin_weights, n_minterms, len(classes))    


    # epsilon = 1e-6
    # # For Glorot correction
    # coin_h = np.float32(np.sqrt(1.5 / (n_minterms + len(classes))))
    
    # # Tau values for thresholding as in FINN
    # tau = np.zeros((len(classes)))
    # tau_inv = np.ones((len(classes)))
    # for c in range (len(classes)):
    #     gamma = coin_weights[1][c]; mov_mean = coin_weights[3][c]; mov_var = coin_weights[4][c]; beta = coin_weights[2][c];
    #     tau[c] = mov_mean - (beta/(gamma/np.sqrt(mov_var)))
    #     tau[c] = math.ceil(tau[c]/coin_h) # Glorot correction
    #     # This correction is not needed. Here just to show the diff from original paper
    #     # tau[c] = int((tau[c]+n_minterms)/2) 
    #     if (gamma/np.sqrt(mov_var+epsilon))<0:
    #         tau_inv[c] = -1
            
        
    print('TAU VALUES:')
    print(classes)
    print(tau)
    # print(tau_inv)
    Y_pred = []
    
    # Eval for each sample
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        scores = np.zeros((len(classes)))
        
        ####### Binarized model ####################
            
        for c in range (len(classes)):
            scores[c] = discriminator_eval_coin(xti.astype(int), model[classes[c]], threshold)
                       
            # Batch normalization correction 
            # scores[c] *= coin_h # Glorot correction
            # gamma = coin_weights[1][c]; mov_mean = coin_weights[3][c]; mov_var = coin_weights[4][c]; beta = coin_weights[2][c];
            # scores[c] = gamma*(scores[c] - mov_mean)/np.sqrt(mov_var + epsilon) + beta
            
            # Batch normalization correction (thresholding as in FINN)
            scores[c] -= tau[c]
            # scores[c] *= tau_inv[c]
        ############################################        
        
        best_class = np.argmax(scores)    
        Y_pred.append(best_class)
    
    
    
    return np.array(Y_pred)


def wisard_eval (X, model, mapping, classes, address_size, min_threshold=1, max_threshold=100):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    
    
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

def get_num_minterms_raw_model (model, classes, thresholds):
    """
    Gets the number of words used for throughout recognizers after the
    minterms fusion.
    """        
    total_min = 0
    for r in range(len(model[classes[0]])):
        unified_ram = {}
        for c in range (len(classes)):
            dict_tmp = model[classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                bit = int(dict_tmp[a]>=thresholds[c,r])
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (bit<<c)
                else:
                    unified_ram[ai] = bit<<c

        for u in unified_ram:
            total_min += 1
 
    return total_min



def wisard_find_threshold (X, Y, model, mapping, classes, address_size, min_threshold=1, max_threshold=100, opt_acc_first=True, acc_delta=0.001, minterms_max=100000):
    max_acc = 0
    maxacc_threshold = 0
    maxacc_cnt = 0
    best_acc = 0
    best_threshold = 0
    best_cnt = 0
    
    Y_pred = wisard_eval_bin_array (X, model, mapping, classes, address_size, thresholds=np.arange(min_threshold,max_threshold+1))
    # Y_pred = wisard_eval_bin (X, model, mapping, classes, address_size, thresholds=np.arange(min_threshold,max_threshold+1))
       
    accuracies = []
    minterms_counts = []
    # Compute all accuracies and word counts
    b = 0
    for threshold in range(min_threshold,max_threshold+1):
        # Y_pred = wisard_eval_bin (X, model, mapping, classes, address_size, thresholds=[threshold])
        acc_t = eval_predictions(Y, Y_pred[b], classes, do_plot=False)   
        # cnt_t = get_above_threshold_count (model, classes, threshold)
        thresholds = threshold*np.ones((len(classes), len(model[classes[0]])))  
        cnt_t = get_num_minterms_raw_model (model, classes, thresholds)
        accuracies.append(acc_t)
        minterms_counts.append(cnt_t)
        b+=1
    
    ## Search for the desired threshold
    thresholds = np.arange(min_threshold,max_threshold+1)
    
    
    
    # Find maximum accuracy
    b_max = np.argmax(accuracies)
    b_max_v = np.where (np.array(accuracies)==accuracies[b_max])[0]
    b = b_max_v[-1]
    max_acc = accuracies[b]
    maxacc_threshold = thresholds[b]
    maxacc_cnt = minterms_counts[b]

    if opt_acc_first==False:         
        # Find sufficient small modules
        smalls_v = np.where (np.array(minterms_counts)<=minterms_max)[0]
        if len(smalls_v)>0:
            smalls_thresholds = thresholds[smalls_v]
            smalls_acc = accuracies[smalls_v]
            smalls_cnt = minterms_counts[smalls_v]
            b = np.argmax(smalls_acc)
            best_acc = smalls_acc[b]
            best_threshold = smalls_thresholds[b]*np.ones((len(classes), len(model[classes[0]])))  
            best_cnt = smalls_cnt[b] 
        else:
            opt_acc_first = True
    
    if opt_acc_first:      
        # Find minimum size
        sel_accs = np.where(accuracies>= max_acc-acc_delta)[0]
        b = sel_accs[-1]
        best_acc = accuracies[b]
        best_threshold = thresholds[b]*np.ones((len(classes), len(model[classes[0]])))                
        best_cnt = minterms_counts[b]   
 
    
        
    return best_acc, best_threshold, best_cnt, max_acc, maxacc_threshold, maxacc_cnt


def multi_threshold_loss (thresholds, Y, Y_counts):
    
    thresholds = thresholds.reshape(Y_counts.shape[1],-1)
    
    # Y_pred = np.zeros((Y_counts.shape[0]), dtype=int)
    
    # for n in range (Y_counts.shape[0]):
    #     above_thresholds_mtx = Y_counts[n,:,:] >= thresholds
    #     above_thresholds_mtx = above_thresholds_mtx.astype(int)        
    #     scores_t = np.sum(above_thresholds_mtx, axis=1)            
    #     # Find the best classes across thresholds
    #     Y_pred[n] = np.argmax(scores_t)

    above_thresholds_mtx = Y_counts >= thresholds
    above_thresholds_mtx = above_thresholds_mtx.astype(int)        
    scores_t = np.sum(above_thresholds_mtx, axis=2)
    Y_pred = np.argmax(scores_t, axis=1)

    error = 1 - (sum(Y_pred == Y) / len(Y))
    
    loss = error + 15/np.mean(thresholds)
    return loss
        
def wisard_find_thresholds (X, Y, model, mapping, classes, address_size, min_threshold=1, max_threshold=100, opt_acc_first=True, acc_delta=0.001, minterms_max=100000):
    max_acc = 0
    maxacc_threshold = 0
    maxacc_cnt = 0
    best_acc = 0
    best_threshold = 0
    best_cnt = 0
    fall_cnt = 0
    thresholds=np.arange(min_threshold,max_threshold+1)
    
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    
    # 3D vector with #samples, #classes, #RAMs
    Y_counts = np.zeros((n_samples,len(classes),len(model[classes[0]])), dtype=int)    
    
    # Eval for each sample
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        
        # Vector is prepared to be used across classes
        X_single = xti.astype(int)
        
        # Filling matrix for scores across classes and rams
        for c in range (len(classes)):
            class_model = model[classes[c]]
            # Find the RAMs values accessed by the input vector X_single            
            for r in range(len(X_single)):
                if X_single[r] in class_model[r]:      
                    Y_counts[n,c,r] = class_model[r][X_single[r]]

    thresholds_0 = np.random.randint(5, min(20, max_threshold), (Y_counts.shape[1]*Y_counts.shape[2]))        
    
    print ("Optimization...",end='')
    res = minimize(multi_threshold_loss, thresholds_0, method='BFGS',
                          args=(Y, Y_counts), options={'disp': True})
    
    print (" done!")
    
    thresholds = np.array(res.x).reshape(Y_counts.shape[1],-1)
    Y_pred = np.zeros((Y_counts.shape[0]), dtype=int)
    
    for n in range (Y_counts.shape[0]):
        above_thresholds_mtx = Y_counts[n,:,:] >= thresholds
        above_thresholds_mtx = above_thresholds_mtx.astype(int)        
        scores_t = np.sum(above_thresholds_mtx, axis=1)            
        # Find the best classes across thresholds
        Y_pred[n] = np.argmax(scores_t)

    best_acc = sum(Y_pred == Y) / len(Y)
    best_threshold = thresholds
    best_cnt = 0
    
    max_acc = best_acc
    maxacc_threshold = best_threshold
    maxacc_cnt = best_cnt
    
    return best_acc, best_threshold, best_cnt, max_acc, maxacc_threshold, maxacc_cnt
            
