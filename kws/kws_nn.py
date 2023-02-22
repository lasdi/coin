#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:59:25 2022

@author: igor
"""
import sys
sys.path.insert(0, '../lib/')
import tensorflow as tf
import numpy as np
# from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.layers import Input, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Embedding, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, AveragePooling2D, MaxPooling2D, MaxPool1D, MaxPool2D, ZeroPadding1D, GlobalMaxPooling2D, GlobalAveragePooling2D, LSTM, SpatialDropout1D
from keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from eval_imbalanced import eval_imbalanced
import os 
from load_data import load_data

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(56)

# Types: mlp, cnn
nn_type = 'mlp'

# fp_classes = open('./speech_commands_v0.02/classes.txt', 'r')
# CLASSES = fp_classes.readlines()
# for i in range(len(CLASSES)):
#     CLASSES[i] = CLASSES[i].replace('\n','')



config = {}
config['THERMO_RESOLUTION'] = 4
config['CLASSES'] = ["down", "go", "left", "no", "off", "on", "right",
                     "stop", "up", "yes", "unknown"]  #, "silence"
CLASSES = config['CLASSES']

X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(config, do_encoding=True)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
Y_val=to_categorical(Y_val)

if nn_type=='mlp':
    X_train = X_train.reshape((X_train.shape[0],-1))
    X_val = X_val.reshape((X_val.shape[0],-1))
    X_test = X_test.reshape((X_test.shape[0],-1))
    hidden_neurons = 128
    ann_model = Sequential()
    ann_model.add(Dense(hidden_neurons, activation='relu', input_shape=(X_train.shape[1],)))
    ann_model.add(Dense(hidden_neurons, activation='relu'))
    ann_model.add(Dense(hidden_neurons, activation='relu'))
    ann_model.add(Dense(hidden_neurons, activation='relu'))
    ann_model.add(Dense(len(CLASSES), activation='softmax'))
elif nn_type=='cnn':
    X_train = np.expand_dims(X_train, 3)
    X_val = np.expand_dims(X_val, 3)
    X_test = np.expand_dims(X_test, 3)
    
    ann_model=Sequential()    
    ann_model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=X_train.shape[1:4]))    
    ann_model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))    
    ann_model.add(MaxPool2D())  
    ann_model.add(Dropout(0.25)),
    ann_model.add(Flatten())    
    ann_model.add(Dense(128, activation='relu'))
    ann_model.add(Dropout(0.5)),
    ann_model.add(Dense(len(CLASSES), activation = 'softmax'))    

ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

ann_model.summary()

ann_model_history = ann_model.fit(X_train, Y_train,validation_data=(X_val, Y_val), epochs=10, batch_size = 100)

print(">>> Test set evaluation...")
y_true=[]
for element in Y_test:
    y_true.append(np.argmax(element))
prediction_proba=ann_model.predict(X_test,verbose=0)
y_prediction=np.argmax(prediction_proba,axis=1)

sensitivities, specificities, accuracy = eval_imbalanced(y_true, y_prediction, CLASSES, do_plot=True)
