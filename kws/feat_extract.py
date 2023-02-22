#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:42:44 2018

@author: igor
"""


import numpy as np
import librosa
import librosa.display
import re
import sys
sys.path.insert(0, '../clibs/')
import elan
from load_dataset import load_vector_list
from feat_extract_chunk import feat_extract_chunk
import tdsplib as td

def feat_extract (configurations, LST_FILE, setname="kfold"):
    """
    Extracts the features of wav files segments specified in *.eaf files. 
    The features extracted are defined in the variable list SELECTED_FEAT in the configurations dictionary. 
    The implemented features are ZCR, log of mel scaled frequencies, MFCC and MFCC 1st and 
    2nd derivatives. If more features are desired, implement it in this function. 
    The maximum segment length is defined by the configuration FEAT_DURATION_MS.
    If the segment length defined in the eaf files is smaller the FEAT_DURATION_MS, it is filled with zeros.
    

    :type configurations: dictionary
    :param configurations: Set of configurations defined in dsprep_cfg.py and class_cfg.py

    :type LST_FILE: string
    :param LST_FILE: A list of eaf files (*.glst), with relative path from the project directory
    """
    
    # Directory where the wav files are
    DATADIR_PATH = configurations["DATADIR_PATH"]
    CLASSES = configurations["CLASSES"]
    FEAT_DURATION_MS = configurations["FEAT_DURATION_MS"]    
    THRESHOLD_FEAT_EXTRACT_DB = configurations["THRESHOLD_FEAT_EXTRACT_DB"]
    FEAT_FS = configurations["FEAT_FS"]
    T_FRAME = configurations["T_FRAME"]
    T_MF_SMOOTH = configurations["T_MF_SMOOTH"]
    
    if(FEAT_DURATION_MS==160):
        FEAT_N_INNER = 7
    elif (FEAT_DURATION_MS==320):
        FEAT_N_INNER = 3
    elif (FEAT_DURATION_MS==480):
        FEAT_N_INNER = 2
    else:
        FEAT_N_INNER = 1
        
    FEAT_N_INNER *= np.ones((len(CLASSES)), dtype=int)
    FEAT_OVLP_INNER = 0.75 * np.ones((len(CLASSES)))

    n_frame = int(0.025*FEAT_FS)   # 25 ms used as standard for power calc  
    n_mf_smooth=int(T_MF_SMOOTH*FEAT_FS/n_frame)
    if (n_mf_smooth%2==0):
        n_mf_smooth+=1   
        
    MAX_DURATION_SAMPLES = int(FEAT_DURATION_MS*FEAT_FS/1000)
    
    
    if(setname=='global'):
        with open(DATADIR_PATH+LST_FILE) as f:
            content = f.readlines()
        my_list = [x.strip() for x in content] 
    else:
        my_list = load_vector_list(configurations, setname, "lst")
    
    print("fe > Extracting features...") 
    g_cnt = 0
    
    for file in my_list:
        
        wav_pathname = DATADIR_PATH+ re.sub(r'(?is)__.+', '.wav', file) #  file.replace("__.*",".wav")
        res_tmp = re.search(r'__[a-zA-Z]', file)
        file_class = re.sub('_','', res_tmp[0])
        file_class_ind = CLASSES.index(file_class)
        
        data, fs = librosa.load(wav_pathname, sr=FEAT_FS, mono=True)
        data /= (np.max(np.abs(data)))
        
        if(THRESHOLD_FEAT_EXTRACT_DB > 0):            
            pdata = td.signal2power (data, n_frame, 0) 
            noise_mean, _ = td.smooth_mean (pdata, n_mf_smooth) 
            threshold_ratio = 10**(THRESHOLD_FEAT_EXTRACT_DB/10)
            threshold = threshold_ratio * noise_mean
        
        # Load events from the file
        ini_ms, end_ms, strs = elan.elan_read (DATADIR_PATH+file)
    
        for k in range (0,len(ini_ms)):
            ini_samples = int(max (ini_ms[k] * fs/1000, 0))
            end_samples_prev = int(min (end_ms[k] * fs/1000, len(data)-1))
            
            if(THRESHOLD_FEAT_EXTRACT_DB > 0):    
                ini_samples_pwr = int(max(np.round(ini_samples/n_frame),0))
                end_samples_prev_pwr = int(min(np.round(end_samples_prev/n_frame), len(pdata)-1))
                pdata_tmp = pdata[ini_samples_pwr:end_samples_prev_pwr]
                if(np.max(pdata_tmp)<threshold):
                    continue
            
            for inner_cnt in range(FEAT_N_INNER[file_class_ind]):                                
                if (inner_cnt==0):                    
                    end_samples = int(min(ini_samples+MAX_DURATION_SAMPLES-1, end_samples_prev))
                    if (end_samples >= len(data)):
                        end_samples = len(data)-1
                else:
                    ini_samples += int(FEAT_OVLP_INNER[file_class_ind]*MAX_DURATION_SAMPLES)
                    end_samples = int(ini_samples+MAX_DURATION_SAMPLES-1)
                    if(end_samples>=end_samples_prev):
                        continue
                
                chunk = np.zeros ((MAX_DURATION_SAMPLES))
                chunk[0:end_samples-ini_samples] = data[ini_samples:end_samples]        
                
                db_col = feat_extract_chunk (configurations, chunk)    

                chunk_n = '_{:03d}-{:02d}.feat.npy'.format(k, inner_cnt)
                chunk_name = file.replace(".eaf", chunk_n)
                np.save(DATADIR_PATH+chunk_name, db_col)
                #print("\t"+DATADIR_PATH+chunk_name)
                g_cnt += 1
    print("%d eaf files analyzed. %d events found." % (len(my_list), g_cnt))

