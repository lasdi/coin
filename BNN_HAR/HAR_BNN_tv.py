#  v3 - Cross validation + different accuracy evaluations; combined train and test data

# Trains a simple binarize CNN on the HAR dataset

from __future__ import print_function
import numpy as np
import keras.backend as K
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
np.random.seed(1337)  # for reproducibility

from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv1D

def binary_tanh(x):
    return binary_tanh_op(x)

# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values
 
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded
 
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y
 
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    #trainy = trainy - 1
    #testy = testy - 1
    # one hot encode y
    #trainy = to_categorical(trainy)
    #testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy
 
# standardize data
def scale_data(trainX):
    # remove overlap
    cut = int(trainX.shape[1] / 2)
    longX = trainX[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
    #flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
    # standardize
    s = StandardScaler()
    # fit on training data
    s.fit(longX)
    # apply to training and test data
    longX = s.transform(longX)
    flatTrainX = s.transform(flatTrainX)
    #flatTestX = s.transform(flatTestX)
    # reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
    #flatTestX = flatTestX.reshape((testX.shape))
    return flatTrainX

H = 1.
kernel_lr_multiplier = 'Glorot'

# BNN
batch_size = 50
epochs = 1
kernel_size = 3
pool_size = 2
classes = 6
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-6
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# fit and evaluate a model
def evaluate_model():
    # load data
    trainX, trainy, testX, testy = load_dataset()

    # Merge inputs and targets
    trainX = np.concatenate((trainX, testX), axis=0)
    trainy = np.concatenate((trainy, testy), axis=0)

    # scale data
    trainX = scale_data(trainX)

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cvscores = []
    
    for train, val in kfold.split(trainX, trainy):

        # zero-offset class values
        trainy_int = trainy - 1
        # one hot encode y
        trainy_int = to_categorical(trainy_int)

        n_timesteps, n_features, n_outputs = trainX[train].shape[1], trainX[train].shape[2], trainy_int[train].shape[1]

        model = Sequential()
    
        # conv1
        model.add(BinaryConv1D(filters=128, kernel_size=kernel_size, input_shape=(n_timesteps, n_features),
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv1'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
        model.add(Activation(binary_tanh, name='act1'))
    
        # conv2
        model.add(BinaryConv1D(filters=128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv2'))
        model.add(MaxPooling1D(pool_size=pool_size, name='pool2'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
        model.add(Activation(binary_tanh, name='act2'))

        model.add(Flatten())
        model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
        model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

        opt = Adam(lr=lr_start) 
        model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
        model.summary()

        lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
        history = model.fit(trainX[train], trainy_int[train], batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(trainX[val], trainy_int[val]), callbacks=[lr_scheduler])
        score = model.evaluate(trainX[val], trainy_int[val], verbose=0)
        print('Accuracy:', score[1])
        cvscores.append(score[1])
        
        # scikit learn
        y_pred = model.predict(trainX[val])
        y_pred = np.argmax(y_pred, axis=1) + 1
        scikit_learn_score = accuracy_score(trainy[val], y_pred)
        print('Scikit learn accuracy on val data:', scikit_learn_score)
 
    cm = confusion_matrix(trainy[val], y_pred)  
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show() 

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores)*100, np.std(cvscores)*100)) 

evaluate_model()

#import pickle
#with open('./conv_bnn.pkl', 'wb') as outp:
#    pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)