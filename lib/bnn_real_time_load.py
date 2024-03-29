#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:46:47 2022

@author: igor
"""

import numpy as np
import keras

class DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(4704,), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # print(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def __data_generation(self, list_IDs_temp):
    #     'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     # Initialization

    #     i = np.random.randint(1,len(list_IDs_temp))
    #     ID = list_IDs_temp[i]
        
    #     # Generate data
        
    #     # Store sample
    #     X = np.load('data/' + ID + '.npy')

    #     # Store class
    #     y = np.array([self.labels[ID]])        
    #     Yt = keras.utils.all_utils.to_categorical(y.astype('int'), num_classes=self.n_classes)*2-1
    #     print('X shape: ',X.shape)
    #     print('Y shape: ',Yt.shape)
    #     return X.T, Yt
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            xt = np.load('data/' + ID + '.npy')
            X[i,:] = np.squeeze(xt.astype('int'))

            # Store class
            y[i] = self.labels[ID]
        Yt = keras.utils.all_utils.to_categorical(y.astype('int'), num_classes=self.n_classes)*2-1
        # print('X shape: ',X.shape)
        # print('Y shape: ',Yt.shape)
        return X, Yt    