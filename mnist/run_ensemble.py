#!/usr/bin/env python3

"""
COIN (COmbinational Intelligent Networks)
Ensemble Example

This examples shows how to build an ensemble model for MNIST 
classification. For this we generate 4 LogicWiSARD models with
different configurations, as detailed below.
Most configurations are set in mnist/config.py (see for details).
Models, results and logs are sent to mnist/out
"""

import sys
sys.path.insert(0, './lib/')
import os
from load_config import load_config
from gen_logicwisard import gen_logicwisard
from train_coin import train_coin
from eval_ensemble import eval_ensemble

# Sets project name
project_name = 'mnist'   

# Loads all configurations from config.py file in project dir
config = load_config('./'+project_name)

# Cleans the output directory
os.system('rm -f ./'+project_name+'/out/*')

# For each of the 2 configurations, generates 3 LogicWiSARD models, 
# picking up the 2 the most accurate. 4 models in total
config['N_SEL_MODELS'] = 2
config['ADDRESS_SIZE'] = 14
gen_logicwisard(project_name, config)
config['ADDRESS_SIZE'] = 16
gen_logicwisard(project_name, config)

# Convert all LogicWiSARD models to BNN, train them, 
# and then convert to COINs
config['N_TRAIN_MODELS'] = 4
train_coin(project_name, config)

# Evaluate the for models as an ensemble
eval_ensemble(project_name, config) 