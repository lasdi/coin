#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:22:38 2022

@author: igor
"""
import numpy as np 

def get_msb (x):
    number = x
    bitpos = 0
    while number != 0:
        bitpos+=1             # increment the bit position
        number = number >> 1 # shift the whole thing to the right once
    return bitpos

def exp2_encode_vec (X, encods, ratio):
    
                    
    Y = np.zeros((1,0), dtype=int)
    # Find the most significant bits and replace
    for i in range (len(X)):
        msb = int(get_msb (X[i])/ratio)
        # print('MSB', msb)
        Y = np.hstack([Y, encods[msb]])
        
    return Y[0]

def exp2_encode (X, ibits, obits):
    ratio = int(ibits/obits)
    # Build the encoded sequences    
    encods = []
    for i in range (obits+1):
        v = np.hstack([np.ones((1,i)), np.zeros((1,obits-i))])
        encods.append(v.astype(int)) 
    
    # Encode sample by sample
    Y = np.zeros((X.shape[0], obits*X.shape[1]), dtype=int)
    for i in range(X.shape[0]):
        vec = np.squeeze(X[i,:])
        Y[i,:] = exp2_encode_vec (vec, encods, ratio)
        
    return Y


def exp2_encode_new (X, ibits, obits):
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
