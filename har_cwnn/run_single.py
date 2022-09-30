#!/usr/bin/env python3

"""
COIN (COmbinational Intelligent Networks) - Single Example

This examples runs a simple MNIST training using the standard 
configurations defined in mnist/config.py (see for details).
Models, results and logs are sent to mnist/out
"""

import sys
sys.path.insert(0, '../lib/')
import os
from load_config import load_config
from gen_logicwisard import gen_logicwisard
from train_coin import train_coin
import numpy as np

# Sets project name
project_name = 'har_cwnn'   

# Loads all configurations from config.py file in project dir
config = load_config('./')
np.random.seed(config['SEED'])

# cleans the output directory
os.system('rm -f ./out/*')

# Generates and trains 3 LogicWiSARD models, picking up 
# the most accurate.
gen_logicwisard(project_name, config)

# Convert LogicWiSARD to BNN, train it, and then convert to COIN
train_coin(project_name, config)
