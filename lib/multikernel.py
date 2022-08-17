#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:47:12 2022

@author: igor
"""
import numpy as np
from discriminator import discriminator_train, discriminator_eval, discriminator_eval_bc
from wisard_tools import separate_classes, eval_predictions
from hamming import hamming_correction
import matplotlib.pyplot as plt
from statistics import mode

def wisard_eval_bin_mk (X, model, mapping, classes, address_size, threshold=1, hamming=False, bc_weights='',n_minterms=0):
    n_samples = X.shape[0]
    X_mapped = X[:,mapping]
    epsilon = 1e-6
    bc_h = np.float32(np.sqrt(1.5 / (n_minterms + len(classes))))
    
    if hamming:
        X_mapped = hamming_correction(X_mapped, address_size)
    
    
    # Eval for each sample
    scores = np.zeros((n_samples, len(classes)))
    
    for n in range (n_samples):
        xt = X_mapped[n,:].reshape(-1, address_size)
        xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
        
        
        ####### Binarized model ####################
            
        for c in range (len(classes)):
            scores[n,c] = discriminator_eval_bc(xti.astype(int), model[classes[c]], threshold)
            # Batch normalization correction
            scores[n,c] = bc_weights[1][c]*(bc_h*scores[n,c] - bc_weights[3][c])/np.sqrt(bc_weights[4][c] + epsilon) + bc_weights[2][c]
                
        ############################################        

    
    
    return scores



def classify_mk (models, X, Y_test, hamming=False, bc = False):
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
        
        if bc:
            model_arg = lw.model_bc   
        elif hamming:
            model_arg = lw.model_hamm         
        else:
            model_arg = lw.model
             
    
        t_scores = wisard_eval_bin_mk(X, model_arg, lw.mapping, lw.classes, 
                             lw.address_size, threshold=lw.min_threshold, 
                             hamming=hamming, bc_weights=lw.bc_weights,n_minterms=lw.get_minterms_info())
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