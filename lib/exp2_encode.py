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

def exp2_encode_vec (X, ibits, obits):
    ratio = int(ibits/obits)

    # Build the encoded sequences    
    encods = []
    for i in range (obits+1):
        v = np.hstack([np.ones((1,i)), np.zeros((1,obits-i))])
        encods.append(v.astype(int))  
                    
    Y = np.zeros((1,0), dtype=int)
    # Find the most significant bits and replace
    for i in range (len(X)):
        msb = int(get_msb (X[i])/ratio)
        # print('MSB', msb)
        Y = np.hstack([Y, encods[msb]])
        
    return Y[0]

def exp2_encode (X, ibits, obits):

    Y = np.zeros((X.shape[0], obits*X.shape[1]), dtype=int)
    for i in range(X.shape[0]):
        vec = np.squeeze(X[i,:])
        Y[i,:] = exp2_encode_vec (vec, ibits, obits)
        
    return Y
if __name__ == "__main__":
    
    Y = exp2_encode(np.array([[1,2],[8, 33]]), 8, 8)
    print (Y)