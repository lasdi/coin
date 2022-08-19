'''Trains a simple binarize fully connected NN on the MNIST dataset.
Modified from keras' examples/mnist_mlp.py
Gets to 97.9% test accuracy after 20 epochs using theano backend
'''


from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(896)  

import tensorflow as tf
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

def bnn_mlp (config, X_train, y_train, X_test, y_test):

    nb_classes = len(config['CLASSES'])
    
    H = 'Glorot'
    kernel_lr_multiplier = 'Glorot'
    USE_BIAS = False


    N_TRAIN_EPOCHS = config['N_TRAIN_EPOCHS']    
    BATCH_SIZE = config['BATCH_SIZE']    
    LR_START = config['LR_START']
    LR_DECAY = config['LR_DECAY']
    EPSILON = config['EPSILON']
    MOMENTUM = config['MOMENTUM']
    DROP_IN = config['DROP_IN']
    
    
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
    Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1
    
    model = Sequential()
    model.add(DropoutNoScale(DROP_IN, input_shape=(X_train.shape[1],), name='drop0'))

    model.add(BinaryDense(nb_classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=USE_BIAS,
              name='dense'))

    model.add(BatchNormalization(epsilon=EPSILON, momentum=MOMENTUM, name='bn'))
    
    # model.summary()
    
    opt = Adam(lr=LR_START) 
    model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])

    lr_scheduler = LearningRateScheduler(lambda e: LR_START * LR_DECAY ** e)
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE, epochs=N_TRAIN_EPOCHS,
                        verbose=0, validation_data=(X_test, Y_test),
                        callbacks=[lr_scheduler])

    return model, history

def bnn_mlp_augment (config, n_samples, input_shape, partition, labels):
    from RealTimeLoad import DataGenerator
    
    nb_classes = len(config['CLASSES'])
    
    H = 'Glorot'
    kernel_lr_multiplier = 'Glorot'
    
    USE_BIAS = False


    N_TRAIN_EPOCHS = config['N_TRAIN_EPOCHS']    
    BATCH_SIZE = config['BATCH_SIZE']    
    LR_START = config['LR_START']
    LR_DECAY = config['LR_DECAY']
    EPSILON = config['EPSILON']
    MOMENTUM = config['MOMENTUM']
    DROP_IN = config['DROP_IN']
    
    # Parameters
    params = {'dim': (input_shape,),
              'batch_size': BATCH_SIZE,
              'n_classes': nb_classes,
              'n_channels': 1,
              'shuffle': True}
    
    
    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)   
    
    # convert class vectors to binary class matrices
    # Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
    # Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1
    
    model = Sequential()
    model.add(DropoutNoScale(DROP_IN, input_shape=(input_shape,), name='drop0'))

    model.add(BinaryDense(nb_classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=USE_BIAS,
              name='dense'))

    model.add(BatchNormalization(epsilon=EPSILON, momentum=MOMENTUM, name='bn'))
    
    model.summary()
    
    opt = Adam(lr=LR_START) 
    model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
    
    lr_scheduler = LearningRateScheduler(lambda e: LR_START * LR_DECAY ** e)
    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        # use_multiprocessing=True,
                        # workers=16,
                        epochs=N_TRAIN_EPOCHS,
                        verbose=0, 
                        callbacks=[lr_scheduler])

    return model, history
