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
    Y_train = np_utils.to_categorical(y_train, nb_classes) * 2 - 1 # -1 or 1 for hinge loss
    Y_test = np_utils.to_categorical(y_test, nb_classes) * 2 - 1
    
    model = Sequential()
    model.add(DropoutNoScale(DROP_IN, input_shape=(X_train.shape[1],), name='drop0'))

    model.add(BinaryDense(nb_classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=USE_BIAS,
              name='dense'))

    model.add(BatchNormalization(epsilon=EPSILON, momentum=MOMENTUM, name='bn'))
    

    # model.summary()
    METRICS = ['accuracy']
    #METRICS = [keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'),]
    #METRICS = [keras.metrics.Recall(name='recall')]
    #METRICS = [tf.keras.metrics.MeanSquaredError()]
    opt = Adam(learning_rate=LR_START) 
    model.compile(loss='squared_hinge', optimizer=opt, metrics=METRICS)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=METRICS)

    #class_weights_v = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(Y_test),y=Y_test)
    #class_weights_v = {0: 0.77934463, 1: 0.00910069, 2: 0.09746169, 3: 0.00104672, 4: 0.11304626}
    class_weights_v = None

    model.set_weights(lw_model.create_coin_from_model(model.get_weights()))

    lr_scheduler = LearningRateScheduler(lambda e: LR_START * LR_DECAY ** e)
    history = model.fit(X_train, Y_train,
                        batch_size=BATCH_SIZE, epochs=N_TRAIN_EPOCHS,
                        verbose=int(VERBOSE), validation_data=(X_test, Y_test),
                        callbacks=[lr_scheduler],
                        class_weight=class_weights_v)

    return model, history

def bnn_mlp_augment (config, n_samples, input_shape, partition, labels):
    from RealTimeLoad import DataGenerator
    
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
    
    opt = Adam(learning_rate=LR_START) 
    model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
    
    lr_scheduler = LearningRateScheduler(lambda e: LR_START * LR_DECAY ** e)
    history = model.fit(training_generator,
                        validation_data=validation_generator,
                        # use_multiprocessing=True,
                        # workers=16,
                        epochs=N_TRAIN_EPOCHS,
                        verbose=int(VERBOSE), 
                        callbacks=[lr_scheduler])

    return model, history
