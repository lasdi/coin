#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 06:14:05 2021

@author: igor
"""


# Faster but not so much
def build_dict_ram(addresses):
    dict_ram = {}
    
    for a in addresses:
        dict_ram[a] = 0
    for a in addresses:
        dict_ram[a] += 1
        
    return dict_ram


def build_dict_ram_slow(addresses):
    dict_ram = {}
    
    for a in range (len(addresses)):
        # if addresses[a] != 0:
        if addresses[a] in dict_ram:
            dict_ram[addresses[a]] += 1
        else:
            dict_ram[addresses[a]] = 1
    
    return dict_ram
