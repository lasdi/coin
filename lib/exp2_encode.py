#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:22:38 2022

@author: igor
"""
import numpy as np 


# Used during training
def exp2_encode (X, ibits, obits):
    ratio = int(ibits/obits)
    
    e = 2**np.arange(ibits)
    thresholds = []
    for i in range(ratio-1,ibits,ratio):
        thresholds.append(e[i])
    # print(thresholds)
    terms = [X >= threshold for threshold in thresholds]
    X_enc = np.stack(terms, axis=-1)
    return X_enc.reshape(X_enc.shape[0],-1).astype(int)

# To debug previous function    
def exp2_encode_int (X, ibits, obits):
    ratio = int(ibits/obits)
    # step = 2**ibits / obits
    e = 2**np.arange(ibits)
    thresholds = []
    for i in range(ratio-1,ibits,ratio):
        thresholds.append(e[i])
        # thresholds.append((i+1)*step)
    
    terms = [X >= threshold for threshold in thresholds]
    X_enc = np.stack(terms, axis=-1).astype(int)
    X_enc = np.flip(X_enc, axis=2)
    
    X_enc_int = X_enc.dot(1 << np.arange(X_enc.shape[-1] - 1, -1, -1))
    
    return X_enc_int

# To debug hardware implementation
def exp2_encode_hw (X, ibits, obits):
    ratio = int(ibits/obits)
    Y = np.zeros(X.shape, dtype=int)
    for i in range (X.shape[0]):
        for j in range (X.shape[1]):
            xt = X[i,j]
            xt_b = np.array([int(k) for k in f'{xt:08b}'])
            yt_b = np.zeros(obits, dtype=int)
            yt_b[0] = xt_b[0]
            for l in range(1,obits):
               yt_b[l] = xt_b[l] | yt_b[l-1]
            Y[i,j] = yt_b.dot(1 << np.arange(yt_b.shape[-1] - 1, -1, -1))
            
    return Y


if __name__ == "__main__":
    x_arr = np.array([[1, 5, 20, 128, 96, 255, 0, 34]])
    Y = exp2_encode(x_arr, 8, 8)
    print (Y)
    Y2 = exp2_encode_int(x_arr, 8, 8)
    print (Y2)    
    Y3 = exp2_encode_hw(x_arr, 8, 8)
    print (Y3)    