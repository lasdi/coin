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
N_THREADS = 5
# Enables plots throughout trainings. Doing this for multiple
# may be messy
DO_PLOTS = True
# Enables verbose throughout scripts
VERBOSE = True
# Application classes. Didn't test anything other than numbers
CLASSES = ["down", "go", "left", "no", "off", "on", "right",
                     "stop", "up", "yes", "unknown"]
# Number of samples available for training. It can be reduced
# for speed. 
N_TRAIN = 20000*5
# Number of samples to be used in validation during LogicWiSARD
# generation only. This will be taken from the train set defined
# above. For ex., if N_TRAIN=60000 and N_VAL=5000, then 55000
# samples will be used to train LogicWiSARD and 5000 for threshold
# search. In this case, BNN training will used the whole 60000 set
# for training.
N_VAL = int(N_TRAIN*0.2)
# Number of samples available for test.
N_TEST = 100000
# Address size of LogicWiSARD models.
ADDRESS_SIZE = 13
# Thermometer resolution, if this encoding is used
THERMO_RESOLUTION = 8 
# Enables the Hamming reduction method
DO_HAMMING = False

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
MAX_THRESHOLD = 10
# Sets the accuracy tolerance below the maximum accuracy 
# while searching for the threshold. In other words,
# thresholds that produces models within this range are
# considered to be chosen
ACC_DELTA = 0.001
# Sets how many threshold search attempts it tries before stopping
ACC_PATIENCE = 2


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
DROP_IN = 0.2
# Enables data augmentation
DO_AUGMENTATION = False
# Data augmentation ratio. It must be greater than 1.0
AUGMENT_RATIO = 1.1
