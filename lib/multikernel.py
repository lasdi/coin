#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:47:12 2022

@author: igor
"""
import numpy as np
from discriminator import discriminator_train, discriminator_eval, discriminator_eval_coin
from wisard_tools import separate_classes, eval_predictions
import matplotlib.pyplot as plt
from statistics import mode
import math

def wisard_eval_bin_mk (X, model, mapping, classes, address_size, threshold=1, coin_weights='',n_minterms=0):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    epsilon = 1e-6
    # Glorot correction
    coin_h = np.float32(np.sqrt(1.5 / (n_minterms + len(classes))))

    # Tau values for thresholding as in FINN
    tau = np.zeros((len(classes)))
    tau_inv = np.ones((len(classes)))
    for c in range (len(classes)):
        gamma = coin_weights[1][c]; mov_mean = coin_weights[3][c]; mov_var = coin_weights[4][c]; beta = coin_weights[2][c];
        tau[c] = mov_mean - (beta/(gamma/np.sqrt(mov_var)))
        tau[c] = math.ceil(tau[c]/coin_h) # Glorot correction
        # This correction is not needed. Here just to show the diff from original paper
        # tau[c] = int((tau[c]+n_minterms)/2) 
        if (gamma/np.sqrt(mov_var+epsilon))<0:
            tau_inv[c] = -1
    
        
    # Eval for each sample
    scores = np.zeros((n_samples, len(classes)))
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        
        
        ####### Binarized model ####################
            
        for c in range (len(classes)):
            scores[n,c] = discriminator_eval_coin(xti.astype(int), model[classes[c]], threshold)
            
            # Batch normalization correction
            # scores[n,c] *= coin_h
            # gamma = coin_weights[1][c]; mov_mean = coin_weights[3][c]; mov_var = coin_weights[4][c]; beta = coin_weights[2][c];
            # scores[n,c] = gamma*(scores[n,c] - mov_mean)/np.sqrt(mov_var + epsilon) + beta
            
            # Batch normalization correction (thresholding as in FINN)
            scores[n,c] -= tau[c]
            scores[n,c] *= tau_inv[c]            
            
        ############################################        

    
    
    return scores



def classify_mk (models, X, Y_test, coin = False):
    """
    Runs the trained classifier on X data.

    Parameters
    ----------
    X : numpy array
        The same format as the fit input.

    Returns
    -------
    Y_pred : numpy array
        An int array whose values correspond to the classes indexes.
    """
    n_samples = X.shape[0]
    n_classes = len(models[0].classes)
    n_models = len(models)
    scores = np.zeros((n_models, n_samples, n_classes))
    
    for m in range(n_models):
        lw = models[m]
        
        if coin:
            model_arg = lw.model_coin
        else:
            model_arg = lw.model
             
    
        t_scores = wisard_eval_bin_mk(X, model_arg, lw.mapping, lw.classes, 
                             lw.address_size, threshold=lw.min_threshold, 
                             coin_weights=lw.coin_weights,n_minterms=lw.get_minterms_info())
        scores[m,:,:] = t_scores
    
    
    ### Confidence based
    # Y_pred = []
    # for n in range (n_samples):
    #     ratios = []
    #     inds = []
    #     for m in range(len(models)):
    #         scores_m_n = scores[m,n,:]
    #         imax = np.argmax(scores_m_n)
    #         # vmax = scores_m_n[imax]
    #         # scores_m_n[imax] = 0             
    #         # imax2 = np.argmax(scores_m_n)
    #         # vmax2 = scores_m_n[imax]            
    #         # ratios.append(vmax/(vmax2+1e-6))
    #         inds.append(imax)        
        
    #     # Better ratio based
    #     # Y_pred.append(inds[np.argmax(ratios)])
    #     # # Mode based   
    #     Y_pred.append(mode(inds))

    # ### Naive Summation
    scores_2d = np.sum(scores,axis=0)
    print(scores.shape)
    print(scores_2d.shape)
    Y_pred = []
    for n in range (n_samples):
        Y_pred.append(np.argmax(scores_2d[n,:]))
        
        

    # # Debug plots
    # color_v = ['b','g','r','c','m','y','k','b','g','r']
    # barWidth = 0.15
    # fig = plt.subplots(figsize =(12, 8))    
    # for n in range (n_samples):
    #     classes_v = np.arange(n_classes)
    #     for m in range (n_models):
    #         plt.bar(classes_v, scores[m,n,:], color =color_v[m], width = barWidth)
    #         classes_v = classes_v + barWidth
    #     print("Predicted/Ref: %d / %d" % (np.argmax(scores_2d[n,:]), Y_test[n]))
    #     plt.show()
    #     dummy = input('Press enter...')
        
    
    

        
    return np.array(Y_pred)
