#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 08:45:23 2021

@author: igor
"""
import numpy as np
from encoders import ThermometerEncoder
import ctypes as c
from scipy.stats import norm
from scipy.ndimage import interpolation

def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    img = interpolation.affine_transform(image,affine,offset=offset)
    return (img - img.min()) / (img.max() - img.min())


def bitfield(n, res): 
    # print (n)
    return [int(digit) for digit in bin(n)[2:].zfill(res)]



def mnist_data_noencode (X, minimum, maximum, resolution):
    o,m,n = X.shape
    f = m*n
    
    X_lst = np.zeros((o,resolution*f))

    for i in range(o):
        img = X[i,:,:]
        
        x_lst_t = img.reshape(-1).tolist()
        
        for j in range (f):
            val = ((2**resolution) - 1)*(int(x_lst_t[j]) - minimum)/(maximum-minimum)
            xt = bitfield(int(val), resolution)
            X_lst[i,j*resolution:(j+1)*resolution] = np.array(xt).reshape(1,-1)

    return X_lst    

def wisard_data_encode(X, classes, resolution=1, minimum=0, maximum=1):
    n_dim = len(X.shape)
    if n_dim==2:
        o,f = X.shape
    elif (n_dim==3):
        o,m,n = X.shape
        f = m*n
    else:
        return 0, 0
    
    X_lst = np.zeros((o,f*resolution))
    
    if resolution>1:
        thermometer = ThermometerEncoder(minimum=minimum, maximum=maximum, resolution=resolution)
    for i in range(o):
        if n_dim==2:
            x_lst_t = X[i,:].reshape(-1).tolist()
        else:
            x_lst_t = X[i,:,:].reshape(-1).tolist()
        if resolution>1:
            x_lst_t = thermometer.encode(x_lst_t)
        else:
            x_lst_t = np.array(x_lst_t).reshape(1,-1)
            
        flat_x_lst = [item for sublist in x_lst_t for item in sublist]
        X_lst[i,:] = np.array(flat_x_lst)
    
    return X_lst

def mnist_data_encode_b(X):

    o,m,n = X.shape
    f = m*n
    
    X_lst = np.zeros((o,f))

    for i in range(o):

        # x_lst_t = X[i,:,:].reshape(-1).tolist()
        X_lst[i,:] = X[i,:,:].reshape(1,-1)
        
        # img = deskew(X[i,:,:]/255)*255
        # img = img.astype(int)
        # X_lst[i,:] = img.reshape(1,-1)
        
        xi_mean = np.mean(X_lst[i,:])
        
        X_lst[i,:] = np.where(X_lst[i,:] > xi_mean, 1, 0)
    
    return X_lst

def mnist_data_encode_t(X, minimum, maximum, resolution):
    # minimum = 0
    # maximum = 255
    # resolution = 8
    o,m,n = X.shape
    # m = 28
    # n = 28
    f = m*n
    
    X_lst = np.zeros((o,resolution*f))

    thermometer = ThermometerEncoder(minimum=minimum, maximum=maximum, resolution=resolution)
    
    for i in range(o):
        img = X[i,:,:]
        
        #img = deskew(img/255)*255
        #img = img.astype(int)
        #msk = (img > 25 ).astype(int)
        #img = msk*img
        
        # img = cv2.resize(X[i,:,:], (m,n), interpolation = cv2.INTER_CUBIC)
        
        x_lst_t = img.reshape(-1).tolist()
        
        # for j in range (len(x_lst_t)):            
        #     X_lst[i,j*resolution:(j+1)*resolution] = np.array(thermometer.encode(x_lst_t[j])).reshape(1,-1)

        X_lst[i,:] = np.array(thermometer.encode(x_lst_t).T).reshape(1,-1)

    
    return X_lst

def mnist_data_encode_z(X,x_mean_a, x_std_a):
    bits_per_input = 8
    o,m,n = X.shape
    f = m*n
    
    X_lst = np.zeros((o,bits_per_input*f))
    X_flat = X.reshape(o,-1)
    if len(x_mean_a)==0:
        x_mean = np.mean(X_flat, axis=0,keepdims=True)
        x_std = np.std(X_flat, axis=0,keepdims=True)
    else:
        x_mean = x_mean_a
        x_std = x_std_a
    std_skews = [norm.ppf((i+1)/(bits_per_input+1)) for i in range(bits_per_input)]
    
    x_tmp = np.zeros((f,bits_per_input))
    for i in range(o):
        for j in range(bits_per_input):
           x_tmp[:,j] = (X_flat[i,:] >= x_mean + (std_skews[j]*x_std)).astype(c.c_ubyte)
        X_lst[i,:] = x_tmp.reshape(1,-1)
    
    return X_lst, x_mean, x_std

