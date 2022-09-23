#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:59:25 2022

@author: igor
"""

import tensorflow as tf
import numpy as np
# from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.layers import Input, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Embedding, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, MaxPool1D, ZeroPadding1D, GlobalMaxPooling2D, GlobalAveragePooling2D, LSTM, SpatialDropout1D
from keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from preprocess_mitdb import load_mitdb
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


from eval_imbalanced import eval_imbalanced
import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(56)

# Types: mlp, cnn_sathvik, lenet5
nn_type = 'cnn_sathvik'

X_train, Y_train, X_test, Y_test = load_mitdb(intra_patients=False, n_max_class=40000)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.2)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_val=to_categorical(Y_val)

if nn_type=='mlp':
    hidden_neurons = 50
    ann_model = Sequential()
    ann_model.add(Dense(hidden_neurons, activation='relu', input_shape=(187,)))
    ann_model.add(Dense(hidden_neurons, activation='relu'))
    ann_model.add(Dense(hidden_neurons, activation='relu'))
    ann_model.add(Dense(hidden_neurons, activation='relu'))
    ann_model.add(Dense(5, activation='softmax'))
elif nn_type=='cnn_sathvik':
    X_train = np.expand_dims(X_train, 2)
    X_val = np.expand_dims(X_val, 2)
    X_test = np.expand_dims(X_test, 2)
    
    # LeNet5
    ann_model=Sequential()    
    ann_model.add(Conv1D(filters=50, kernel_size=35, strides=1, padding='same', activation='relu', input_shape=X_train.shape[1:3]))    
    ann_model.add(MaxPool1D(pool_size=2))    
    ann_model.add(Flatten())    
    ann_model.add(Dense(300, activation='relu'))
    ann_model.add(Dense(150, activation='relu'))    
    ann_model.add(Dense(5, activation = 'softmax'))    
elif nn_type=='lenet5':
    X_train = np.expand_dims(X_train, 2)
    X_val = np.expand_dims(X_val, 2)
    X_test = np.expand_dims(X_test, 2)
    
    # LeNet5
    ann_model=Sequential()    
    ann_model.add(Conv1D(filters=6, kernel_size=3, padding='same', activation='relu', input_shape=X_train.shape[1:3]))
    ann_model.add(BatchNormalization())
    ann_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))    
    ann_model.add(Conv1D(filters=16, strides=1, kernel_size=5, activation='relu'))
    ann_model.add(BatchNormalization())
    ann_model.add(MaxPool1D(pool_size=2, strides=2, padding='same'))    
    ann_model.add(Flatten())    
    ann_model.add(Dense(64, activation='relu'))
    ann_model.add(Dense(32, activation='relu'))    
    ann_model.add(Dense(5, activation = 'softmax'))
 
    
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

ann_model_history = ann_model.fit(X_train, Y_train,validation_data=(X_val, Y_val), epochs=10, batch_size = 100)

print(">>> Test set evaluation...")
y_true=[]
for element in Y_test:
    y_true.append(np.argmax(element))
prediction_proba=ann_model.predict(X_test,verbose=0)
y_prediction=np.argmax(prediction_proba,axis=1)

CLASSES = ['N','S', 'V', 'F', 'Q']
sensitivities, specificities, accuracy = eval_imbalanced(y_true, y_prediction, CLASSES, do_plot=True)
