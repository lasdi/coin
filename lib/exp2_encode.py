#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:22:38 2022

@author: igor
"""
import numpy as np 

def exp2_encode (X, ibits, obits):
    ratio = int(ibits/obits)
    
    e = 2**np.arange(ibits)
    thresholds = []
    for i in range(ratio-1,ibits,ratio):
        thresholds.append(e[i])
    
    terms = [X >= threshold for threshold in thresholds]
    X_enc = np.stack(terms, axis=-1)
    return X_enc.reshape(X_enc.shape[0],-1).astype(int)
    

if __name__ == "__main__":
    
    Y = exp2_encode(np.array([[1,2],[8, 128]]), 8, 8)
    print (Y)