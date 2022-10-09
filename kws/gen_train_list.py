#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:13:06 2022

@author: igor
"""


fp_files_full = open('./full_list.txt', 'r')
files_full = fp_files_full.readlines()
fp_files_test = open('./testing_list.txt', 'r')
files_test = fp_files_test.readlines()
fp_files_val = open('./validation_list.txt', 'r')
files_val = fp_files_val.readlines()

training = []

for filename in files_full:
    
    if (filename not in files_test) and (filename not in files_val):    
        training.append(filename.replace('\n',''))
    
print(len(training))    

with open('training_list.txt', 'w') as f:
    for line in training:
        f.write(f"{line}\n")