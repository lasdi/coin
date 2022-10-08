#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:42:44 2018

@author: igor
"""


import numpy as np
import librosa
import librosa.display

# import sys
# sys.path.insert(0, '../clibs/')
# import triang_spectrogram as ts
# from rcepstrum import rcepstrum
import python_speech_features as psf

def feat_extract_chunk (configurations, chunk):
    """
    Extract features for a chunk of data

    :type configurations: dictionary
    :param configurations: Set of configurations defined in dsprep_cfg.py and class_cfg.py

    :type chunk: np.array
    :param chunk: Chunk with acoustic signal
    """      
    FEAT_FS = configurations["FEAT_FS"]
    SELECTED_FEAT = configurations["SELECTED_FEAT"]
    T_FRAME = configurations["T_FRAME"]
    T_FRAME_OVERLAP = configurations["T_FRAME_OVERLAP"]
    N_MELS_BPF = configurations["N_MELS_BPF"]
    N_MFCC = configurations["N_MFCC"]

    n_frame = int(T_FRAME*FEAT_FS) 
    n_frame_overlap = int(T_FRAME_OVERLAP*FEAT_FS)     
    n_hop = n_frame - n_frame_overlap 
    

    ### Using librosa
    # Spectrogram in mel scale
    S = librosa.feature.melspectrogram(chunk, sr=FEAT_FS, n_fft=n_frame, hop_length=n_hop, n_mels=N_MELS_BPF, power=2)        
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    features = librosa.power_to_db(S, ref=np.max, top_db=None)
        

    # Next, we extract the top 13 Mel-frequency cepstral coefficients (MFCCs)                
    if(('delta0_mfcc' in SELECTED_FEAT) or ('raw_mfcc' in SELECTED_FEAT)):
        
        ### Using Librosa from MEL FB computed previously
        mfcc = librosa.feature.mfcc(S=features, n_mfcc=N_MFCC)  #, norm='ortho')
        
        if('delta0_mfcc' in SELECTED_FEAT):
            mfcc = psf.lifter(mfcc.T,23)
            mfcc = mfcc.T
        
        features = mfcc

    
    return features
