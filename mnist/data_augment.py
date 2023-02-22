#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:12:12 2022

@author: igor
"""
import sys
sys.path.insert(0, '../lib/')
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.datasets import mnist
import numpy as np
import pickle
from project_tools import wisard_data_encode, mnist_data_encode_b, mnist_data_encode_t, mnist_data_encode_z

def gen_data_raw (n_samples, batch_size):

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    X = X_train.reshape((X_train.shape[0], 28, 28, 1))    
    X = X.astype('float32')    
    X /= 255
    
    shift = 0.075
    rotation = 15
    datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift, rotation_range=rotation)
    datagen.fit(X)
    
    X_train_augm = np.empty((0,X.shape[1],X.shape[2]))
    Y_train_augm = np.empty((0,1))
    n_gen_samples = 0
    if n_samples>0:
        for X_batch, y_batch in datagen.flow(X, Y_train, batch_size=batch_size):
            # create a grid of 3x3 images
            X_train_augm = np.vstack([X_train_augm, X_batch.reshape(X_batch.shape[0],X_batch.shape[1],X_batch.shape[2])])
            Y_train_augm = np.vstack([Y_train_augm, y_batch.reshape(y_batch.shape[0],1)])
            n_gen_samples += batch_size
            if n_gen_samples>=n_samples:
                break
            
    # Convert and concatenate augmented data
    X_train_augm *= 255
    X_train_augm = X_train_augm.astype('uint8')    
    X_train_augm = np.vstack([X_train, X_train_augm])
    Y_train_augm = np.concatenate((Y_train, np.squeeze(Y_train_augm)))
    
    # shuffling
    n_shuffled = np.arange(X_train_augm.shape[0])
    np.random.shuffle(n_shuffled)
    X_train_augm = X_train_augm[n_shuffled,:,:]
    Y_train_augm = Y_train_augm[n_shuffled]

    return X_train_augm, Y_train_augm
    
    
def gen_data (n_samples, batch_size, thermo_resolution, mWisard=0, bnn=True):

    X_train_augm, Y_train_augm = gen_data_raw (n_samples, batch_size)


    # Y_train_augm = np.squeeze(Y_train_augm)
    
    X_train_lst = mnist_data_encode_t(X_train_augm, 0,255,thermo_resolution)
    
    if bnn:
        ids = []
        labels = {}
        n_total_samples = 0
        for n in range(0,X_train_lst.shape[0],batch_size):
            X_train_coin_tmp = mWisard.gen_coin_encode(X_train_lst[n:n+batch_size,:])
            prefix = 'train_b'+str(int(n/batch_size))
            ids_tmp, labels_tmp = save_data(X_train_coin_tmp,Y_train_augm[n:n+batch_size], prefix, save_idlabel=False)
            ids = [*ids, *ids_tmp]
            labels.update(labels_tmp)
            n_total_samples += batch_size
        
        out_dir = './data/'
        with open(out_dir+'train_ids.pkl', 'wb') as outp:
            pickle.dump(ids, outp, pickle.HIGHEST_PROTOCOL)
        with open(out_dir+'train_labels.pkl', 'wb') as outp:
            pickle.dump(labels, outp, pickle.HIGHEST_PROTOCOL)    
        return ids, labels, n_total_samples, X_train_coin_tmp.shape[1]
    
    else:
        X_train_lst = X_train_lst.astype(int)
        Y_train_augm = Y_train_augm.astype(int)
        out_dir = '../mnist/out/'
        with open(out_dir+'x_mnist_augmented.pkl', 'wb') as outp:
            pickle.dump(X_train_lst, outp, pickle.HIGHEST_PROTOCOL)
        with open(out_dir+'y_mnist_augmented.pkl', 'wb') as outp:
            pickle.dump(Y_train_augm, outp, pickle.HIGHEST_PROTOCOL)


def save_data(X_train_lst,Y_train_augm, prefix, save_idlabel=True):
    out_dir = './data/'    
    labels = {}
    ids = []
    for n in range(X_train_lst.shape[0]):
        id_str = prefix+'_'+str(n)
        with open(out_dir+id_str+'.npy', 'wb') as f:
            np.save(f, X_train_lst[n,:].reshape(-1,1))
            
        ids.append(id_str)
        labels[id_str] = Y_train_augm[n]
    
    
    if save_idlabel:
        with open(out_dir+prefix+'_ids.pkl', 'wb') as outp:
            pickle.dump(ids, outp, pickle.HIGHEST_PROTOCOL)
        with open(out_dir+prefix+'_labels.pkl', 'wb') as outp:
            pickle.dump(labels, outp, pickle.HIGHEST_PROTOCOL)

    return ids, labels



def load_batch (n_samples):
    out_dir = '../mnist/out/'
    with open(out_dir+'x_mnist_augmented.pkl', 'rb') as inp:
        X_train_augm = pickle.load(inp)
    with open(out_dir+'y_mnist_augmented.pkl', 'rb') as inp:
        Y_train_augm = pickle.load(inp)        

    # shuffling
    n_shuffled = np.arange(X_train_augm.shape[0])
    np.random.shuffle(n_shuffled)
    X_train_augm = X_train_augm[n_shuffled,:]
    Y_train_augm = Y_train_augm[n_shuffled]
        
    if n_samples==-1:
        n_samples = X_train_augm.shape[0]
    
    return X_train_augm[0:n_samples,:], Y_train_augm[0:n_samples]

if __name__ == '__main__':
    
    thermo_resolution = 8
    n_samples = 0 #10000
    batch_size = 32
    gen_data (n_samples, batch_size, thermo_resolution, mWisard=0,bnn=False)
    X_train_augm, Y_train_augm = load_batch(-1)
    
  
