#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 19:28:13 2022

@author: igor
"""

import sys

def load_config (data_path): 
    sys.path.insert(0, data_path)
    import config
    parameters = { "PROJ_DIR": data_path, 
                   "SEED": config.SEED,
                   "DO_PLOTS": config.DO_PLOTS,
                   "ADDRESS_SIZE": config.ADDRESS_SIZE,
                   "THERMO_RESOLUTION": config.THERMO_RESOLUTION,
                   "MIN_THRESHOLD": config.MIN_THRESHOLD,
                   "MAX_THRESHOLD": config.MAX_THRESHOLD,
                   "ACC_DELTA": config.ACC_DELTA,
                   "ACC_PATIENCE": config.ACC_PATIENCE,
                   "N_SEL_MODELS": config.N_SEL_MODELS,
                   "N_GEN_MODELS": config.N_GEN_MODELS,
                   "DO_HAMMING": config.DO_HAMMING,                   
                   "SORT_MODELS_BY": config.SORT_MODELS_BY,                   
                   "CLASSES": config.CLASSES,                   
                   "N_TRAIN": config.N_TRAIN,
                   "N_VAL": config.N_VAL,                   
                   "N_TEST": config.N_TEST,
                   "BATCH_SIZE": config.BATCH_SIZE,
                   "LR_START": config.LR_START,
                   "LR_END": config.LR_END,
                   "LR_DECAY": config.LR_DECAY,
                   "EPSILON": config.EPSILON,
                   "MOMENTUM": config.MOMENTUM,
                   "DROP_IN": config.DROP_IN,                   
                   "DO_AUGMENTATION": config.DO_AUGMENTATION,
                   "AUGMENT_RATIO": config.AUGMENT_RATIO,
                   "N_TRAIN_EPOCHS": config.N_TRAIN_EPOCHS,
                   "N_TRAIN_MODELS": config.N_TRAIN_MODELS,             
                   }
    return parameters 
