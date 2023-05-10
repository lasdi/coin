#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 19:37:48 2023

@author: igor
"""
import sys
import os

def wisard_lut_out_gen (out_dir):
    
    sys.path.insert(0, out_dir)
    
    import wisard_lut
    
    cmd_lst = 'ls -1 '+out_dir+'data/in*.txt | sort > '+out_dir+'in_lst.txt'
    os.system(cmd_lst)    
    files_list = open(out_dir+'in_lst.txt', 'r')
    filenames = files_list.readlines()
    
    
    for filename in filenames:
        # print(filename)
        lut_out_str = ''
        with open(filename.replace('\n','')) as file:
            index = 0            
            for line in file:
                addr = int(line.rstrip(),16)
                if addr in wisard_lut.tables[index]:
                    val = wisard_lut.tables[index][addr]
                else:
                    val = 0
                lut_out_str += str(val)+'\n'
                index += 1
        lo_filename = filename.replace('\n','')  
        lo_filename = lo_filename.replace('/in','/lo')
        text_file = open(lo_filename, "w")
        text_file.write(lut_out_str)
        text_file.close()


if __name__ == "__main__":        
    sys.path.insert(0, '../lib/')
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = '../arrhythmia/out/rtl/coin/'
    wisard_lut_out_gen (path)