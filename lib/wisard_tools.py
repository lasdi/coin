#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 08:45:23 2021

@author: igor
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns




def separate_classes (X, Y, classes, address_size):
    n_rams = int(X.shape[1]/address_size)
    
    X_class = {}
    
    for c in range (len(classes)):
        X_class_t = np.empty((0,n_rams),dtype=int)
        for i in range (X.shape[0]):
            if int(Y[i])==c:
                xt = X[i,:].reshape(-1, address_size)
                xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
                X_class_t = np.vstack([X_class_t, xti.reshape(1,-1)])
    
        X_class[classes[c]] = X_class_t
    return X_class


def eval_predictions(y_true, y_pred, classes, do_plot):

    test_acc = sum(y_pred == y_true) / len(y_true)

    
    # Display a confusion matrix
    # A confusion matrix is helpful to see how well the model did on each of 
    # the classes in the test set.
    if do_plot==True:
        confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mtx, xticklabels=classes, yticklabels=classes, 
                    annot=True, fmt='g')
        plt.xlabel('Prediction')
        plt.ylabel('Label')
        plt.show()
    return test_acc


# This function writes content to a file. 
def write2file(to_write, file_name='./log.txt'):
    # Append-adds at last 
    print(to_write+'\n')
    file = open(file_name,"a")#append mode 
    file.write(to_write) 
    file.close() 