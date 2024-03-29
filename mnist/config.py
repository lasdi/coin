"""
COIN (COmbinational Intelligent Networks)
Configuration  File
These parameters are used throughout the scripts. 
They can be modified during execution as in the following 
example:
    config['DO_PLOTS'] = True    
"""
#################### General ####################

# For reproducibility
SEED = 32
# Number of threads requested simultaneously. Used for both
# LogicWiSARD and COIN training.
N_THREADS = 1
# Enables plots throughout trainings. Doing this for multiple
# may be messy
DO_PLOTS = False
# Enables verbose throughout scripts
VERBOSE = True
# Application classes. Didn't test anything other than numbers
CLASSES = ['0','1','2','3','4','5','6','7','8','9']
# Number of samples available for training. It can be reduced
# for speed. 
N_TRAIN = 60000
# Number of samples to be used in validation during LogicWiSARD
# generation only. This will be taken from the train set defined
# above. For ex., if N_TRAIN=60000 and N_VAL=5000, then 55000
# samples will be used to train LogicWiSARD and 5000 for threshold
# search. In this case, BNN training will used the whole 60000 set
# for training.
N_VAL = 2000
# Number of samples available for test.
N_TEST = 10000
# Address size of LogicWiSARD models.
ADDRESS_SIZE = 16
# Thermometer resolution, if this encoding is used
THERMO_RESOLUTION = 2


#################### LogicWiSARD ####################

# The LogicWiSARD generate a bunch of models and selects a few 
# among them. This parameters set how many should be generated
N_GEN_MODELS = 1
# This is to set how many should be selected.
N_SEL_MODELS = 1
# Defines the selection type: 'accuracy' or 'size'
SORT_MODELS_BY = 'accuracy'
# Minimum value for the threshold search
MIN_THRESHOLD = 1
# Maximum value for the threshold search
MAX_THRESHOLD = 100

# If True, the threshold search will find the smallest models whose accuracy is
# above maximum accuracy minus ACC_DELTA. Otherwise, it will search for the best
# accuracy models whose number of minterms is below MINTERMS_MAX.
OPT_ACC_FIRST = True
# Sets the accuracy tolerance below the maximum accuracy 
# while searching for the threshold. In other words,
# thresholds that produces models within this range are
# considered to be chosen
ACC_DELTA = 0.005
# Maximum number of minterms
MINTERMS_MAX = 100000
# If selected, it will search for a multidimensional threshold, across classes and RAMs
MULTIDIM_THRESHOLD = False
#################### BNN ####################

# Sets the maximum number of models to train. -1 to train all models
N_TRAIN_MODELS = -1
# Number of epochs
N_TRAIN_EPOCHS = 25
# Batch size
BATCH_SIZE = 100
# Learning rate start
LR_START = 3E-3
# Learning rate final value
LR_END = 3E-5
# Learning rate decaying schedule
LR_DECAY = (LR_END / LR_START)**(1. / N_TRAIN_EPOCHS)
# Epsilon on batch normalization
EPSILON = 1E-6
# Momentum on batch normalization
MOMENTUM = 0.9
# Dropout ratio of inputs
DROP_IN = 0.0
# Enables data augmentation
DO_AUGMENTATION = False
# Data augmentation ratio. It must be greater than 1.0
AUGMENT_RATIO = 1.1
