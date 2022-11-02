'''Trains a simple binarize fully connected NN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 97.9% test accuracy after 20 epochs using theano backend
'''


from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import keras
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

from bnn_ops import binary_tanh as binary_tanh_op
from bnn_layers import BinaryDense, Clip

from keras.models import load_model
from sklearn.utils import class_weight

from brevitas_train_bnn import create_model
import torch

class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
    '''
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs

def binary_tanh(x):
    return binary_tanh_op(x)

def bnn_mlp (config, X_train, y_train, X_test, y_test, lw_model):

    nb_classes = len(config['CLASSES'])
    
    H = 'Glorot'
    kernel_lr_multiplier = 'Glorot'
    USE_BIAS = False

    VERBOSE = config['VERBOSE']
    N_TRAIN_EPOCHS = config['N_TRAIN_EPOCHS']    
    BATCH_SIZE = config['BATCH_SIZE']    
    LR_START = config['LR_START']
    LR_DECAY = config['LR_DECAY']
    EPSILON = config['EPSILON']
    MOMENTUM = config['MOMENTUM']
    DROP_IN = config['DROP_IN']
    
    
    # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
    # Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1
    print(X_train.reshape(X_train.shape[0],-1).shape)
    print(y_train.shape)
    # train_dataset = torch.utils.data.TensorDataset(X_train.reshape(X_train.shape[0],-1), y_train)
    # test_dataset = torch.utils.data.TensorDataset(X_test.reshape(X_test.shape[0],-1), y_test)

    train_data = torch.stack([torch.from_numpy(X_train[i,:]) for i in range(len(X_train))]).float()
    train_labels = torch.LongTensor([x for x in y_train]).long()
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_data = torch.stack([torch.from_numpy(X_test[i]) for i in range(len(X_test))]).float()
    test_labels = torch.LongTensor([x for x in y_test]).long()
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    model_fname = "model.pt"
    model = create_model(train_dataset, test_dataset, [], model_fname)
    
    history = []
    return lw_model, history
