#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:30:33 2022

@author: igor
"""

import numpy as np 
import math

def tau_gen (bc_weights, n_minterms, n_classes):
    epsilon = 1e-6
    bc_h = np.float32(np.sqrt(1.5 / (n_minterms + n_classes)))
    tau = np.zeros((n_classes))
    tau_inv = np.ones((n_classes))
    for c in range (n_classes):
        gamma = bc_weights[1][c]; mov_mean = bc_weights[3][c]; mov_var = bc_weights[4][c]; beta = bc_weights[2][c];
        tau[c] = mov_mean - (beta/(gamma/np.sqrt(mov_var)))
        tau[c] = math.ceil(tau[c]/bc_h) # Glorot correction
        # This correction is not needed. Here just to show the diff from original paper
        # tau[c] = int((tau[c]+n_minterms)/2) 
        if (gamma/np.sqrt(mov_var+epsilon))<0:
            tau_inv[c] = -1
            
    return tau