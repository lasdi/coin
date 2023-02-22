#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:30:33 2022

@author: igor
"""

import numpy as np 
import math

def tau_gen (coin_weights, n_minterms, n_classes):
    epsilon = 1e-6
    coin_h = np.float32(np.sqrt(1.5 / (n_minterms + n_classes)))
    tau = np.zeros((n_classes))
    tau_inv = np.ones((n_classes))
    for c in range (n_classes):
        gamma = coin_weights[1][c]; mov_mean = coin_weights[3][c]; mov_var = coin_weights[4][c]; beta = coin_weights[2][c];
        tau[c] = mov_mean - (beta/(gamma/np.sqrt(mov_var)))
        tau[c] = math.ceil(tau[c]/coin_h) # Glorot correction
        # This correction is not needed. Here just to show the diff from original paper
        # tau[c] = int((tau[c]+n_minterms)/2) 
        if (gamma/np.sqrt(mov_var+epsilon))<0:
            tau_inv[c] = -1
            print("### WARNING: INVERTION NEEDED.")
            
    return tau
