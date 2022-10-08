#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:00:17 2022

@author: igor
"""
import numpy as np
from functools import reduce

def gen_h3(ibits, obits):
    return np.random.randint(0,2**obits, (ibits,1))
    
def eval_h3 (h3, addresses):
    ibits = h3.shape[0]
    # obits = h3.shape[1]
    keys = []
    for address in addresses:
        addr_v = [(address >> i)&1 for i in range(ibits)]
        addr_v = np.array(addr_v).reshape(-1,1)        
        mult_addr_h3 = np.multiply(addr_v, h3)
        keys.append(reduce(lambda x, y: x ^ y, mult_addr_h3)[0])
    
    return keys