#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:07:17 2022

@author: igor
"""
import numpy as np
# from h3_encode import gen_h3, eval_h3

def get_rightmost (x):
    number = x
    bitpos = 0
    while number != 0:
        if number&1==1:
            break
        bitpos+=1             # increment the bit position
        number = number >> 1 # shift the whole thing to the right once
    if number == 0:
        return 0   
    return bitpos + 1


def gen_lut_grouped (wsrd, INDEX_WIDTH, O_WIDTH, path, coin=True):
    if coin==True:
        model = wsrd.model_coin
    else:
        model = wsrd.model

    # LUT v2 #############################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nreg [%d:0] out_v [0:%d];\n\n' % (O_WIDTH-1,len(model[wsrd.classes[0]])-1)
    code += '\nassign out = out_v[index];\n\n'      
    
    # ibits= 16; obits = 12;
    # h3 = gen_h3(ibits, obits)    
    # addr_size = obits
    
    addr_size = wsrd.address_size
    
    for r in range(len(model[wsrd.classes[0]])):
        # code += '\nreg [%d:0] out%d;\n' % (N_CLASSES-1, r)
        code += 'always @(*) begin\n  case (addr)\n'
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp =model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                bit = int((dict_tmp[a]+1)/2)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (bit<<c)
                else:
                    unified_ram[ai] = bit<<c

        for u in unified_ram:
            u_addr = u
        

            if unified_ram[u]==0 or unified_ram[u]==2**O_WIDTH-1:
                print("### WARNING - Unified RAM position with all zeros or all ones (will be treated as absent case)")
            else:
                # Decimal printing
                code += '    %d\'d%d: out_v[%d] = %d\'d%d;\n' % (addr_size,u_addr,r, O_WIDTH,unified_ram[u])
            

            
        code += '    default: out_v[%d] = %d\'d0;\n  endcase\nend\n\n' % (r,O_WIDTH)
    


    code += '\nendmodule'
    
    if coin==True:
        text_file = open(path+"wisard_lut.v", "w")
    else:
        text_file = open(path+"wisard_lut_lw.v", "w")
    text_file.write(code)
    text_file.close()      
    

def gen_lut_grouped_python (wsrd, O_WIDTH, path, coin=True):
    if coin==True:
        model = wsrd.model_coin
    else:
        model = wsrd.model
    
    code = 'tables = []\n\n'
    
    for r in range(len(model[wsrd.classes[0]])):
        # code += '\nreg [%d:0] out%d;\n' % (N_CLASSES-1, r)
        code += 'table_tmp = {}\n'
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp =model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                bit = int((dict_tmp[a]+1)/2)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (bit<<c)
                else:
                    unified_ram[ai] = bit<<c
  

        for u in unified_ram:
            u_addr = u


            if unified_ram[u]==0 or unified_ram[u]==2**O_WIDTH-1:
                print("### WARNING - Unified RAM position with all zeros or all ones  (will be treated as absent case)")
            else: 
                ## Decimal printing
                code += 'table_tmp[%d] = %d\n' % (u_addr,unified_ram[u])
            
            
        code += 'tables.append(table_tmp)\n\n' 
    


    # code += '\nendmodule'
    
    if coin==True:
        text_file = open(path+"wisard_lut.py", "w")
    else:
        text_file = open(path+"wisard_lut_lw.py", "w")
    text_file.write(code)
    text_file.close() 
