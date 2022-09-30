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
        model = wsrd.model_bc
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


        # # Sort in reverse order to make it similar to logicwisard
        # if coin==True:
        #     uv = []
        #     dv = []
        #     dvi = []
        #     for u in unified_ram:
        #         uv.append(u)
        #         dv.append(unified_ram[u])
        #         bin_data = "{0:010b}".format(unified_ram[u])
        #         dec_data = int(bin_data[::-1],2)
        #         dvi.append(dec_data)
        #     i_sort = np.argsort(dvi)
        #     unified_ram = {}
        #     for i in reversed(i_sort):
        #         unified_ram[uv[i]] = dv[i]

        # ##### Sort ####    
        # uv = []
        # dv = []
        # for u in unified_ram:
        #     uv.append(u)
        #     dv.append(unified_ram[u])    
        # i_sort = np.argsort(uv)
        # unified_ram = {}   
        # cnt = 0
        # for i in i_sort:
        #     unified_ram[uv[i]] = dv[i]  
        # ################

        # # Convert using H3
        # uv = []
        # dv = []
        # for u in unified_ram:
        #     uv.append(u)
        #     dv.append(unified_ram[u])    
        # uv_keys = eval_h3(h3, uv)
        # i_sort = np.argsort(uv_keys)
        # unified_ram = {}   
        # for i in i_sort:
        #     unified_ram[uv_keys[i]] = dv[i]   
        # ################    

        for u in unified_ram:
            u_addr = u
            
            ## Binary printing - addr and data (debug)
            u_addr = "{0:016b}".format(u_addr)
            bin_data = "{0:010b}".format(unified_ram[u])            
            
            # # only most significant
            # shift = get_rightmost(unified_ram[u])               
            # if shift>2:
            #     bin_data = "{0:010b}".format( 1 << shift-1 )
            # elif shift==0:
            #     bin_data = '0000000000'
            # ########################
            
            code += '    %d\'b%s: out_v[%d] = %d\'b%s;\n' % (addr_size,u_addr,r, O_WIDTH,bin_data)
            
            ## Decimal printing
            #code += '    %d\'d%d: out_v[%d] = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, O_WIDTH,unified_ram[u])
            
            if unified_ram[u]==0:
                print("### WARNING - Unified RAM position with all zeros (will be treated as absent case)")
            
        code += '    default: out_v[%d] = %d\'d0;\n  endcase\nend\n\n' % (r,O_WIDTH)
    


    code += '\nendmodule'
    
    if coin==True:
        text_file = open(path+"wisard_lut.v", "w")
    else:
        text_file = open(path+"wisard_lut_lw.v", "w")
    text_file.write(code)
    text_file.close()      
    
def gen_lut_modules (wsrd, INDEX_WIDTH, O_WIDTH, path, coin = True): 

    if coin==True:
        model = wsrd.model_bc
    else:
        model = wsrd.model
    # LUT v4 ##############################################################
    
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nwire [%d:0] out_v [0:%d];\n\n' % (O_WIDTH-1,len(model[wsrd.classes[0]])-1)
    code += '\nassign out = out_v[index];\n\n'     
    local_code = ''
    for r in range(len(model[wsrd.classes[0]])):
        
        
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp = model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                bit = int((dict_tmp[a]+1)/2)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (bit<<c)
                else:
                    unified_ram[ai] = bit<<c
        
        n_chunks = 4 
        chunk_size =  int(np.ceil(float(len(unified_ram))/n_chunks))
        declarations = 'wire [%d:0] ' % (O_WIDTH-1)
        or_reduction = 'assign out_v[%d] = ' % (r)
        instances = ''
        i_chk = 0         
        chk_cnt = -1
        for u in unified_ram:
            
            if i_chk%chunk_size==0:
                chk_cnt+=1
                local_code += '\nmodule ram%d_%d (input [%d:0] in, output reg [%d:0] out_l);\n' % (r,chk_cnt,wsrd.address_size-1,O_WIDTH-1)
                local_code += 'always @(*) begin\n  case (in[%d:0])\n' % (wsrd.address_size-1)
                if i_chk>0:
                    declarations += ', '
                    or_reduction += ' | '
                declarations += 'out_v_r%d_%d' % (r, chk_cnt)
                or_reduction += 'out_v_r%d_%d' % (r, chk_cnt)
                #instances += 'ram%d_%d ram%d_%d_u0 (.in(addr_%d), .out_l(out_v_r%d_%d));\n' % (r, chk_cnt,r, chk_cnt,r,r, chk_cnt)
                instances += 'ram%d_%d ram%d_%d_u0 (.in(addr[%d:0]), .out_l(out_v_r%d_%d));\n' % (r, chk_cnt,r, chk_cnt,wsrd.address_size-1,r, chk_cnt)
                
            u_addr = u    
            local_code += '    %d\'d%d: out_l = %d\'d%d;\n' % (wsrd.address_size,u_addr, O_WIDTH,unified_ram[u])
            # local_code += '    %d\'d%d: out_v_r%d_%d = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, chk_cnt, O_WIDTH,unified_ram[u])
            
            i_chk+=1
            
            if i_chk%chunk_size==0 or i_chk==len(unified_ram):
                local_code += '    default: out_l = %d\'d0;\n  endcase\nend\n' % (O_WIDTH)
                local_code += 'endmodule\n'
                
            
    
        code += '// RAM %d\n\n' % (r)
        code += declarations+';\n'
        #code += 'wire [%d:0] addr_%d;\n' % (wsrd.address_size-1, r)
        #code += 'assign addr_%d = index==%d ? addr[%d:0] :  %d\'d0;\n' % (r,r,wsrd.address_size-1, O_WIDTH)
        code += instances+'\n'
        code += or_reduction+';\n'
        
    code += '\nendmodule'
    code += local_code
    
    if coin==True:
        text_file = open(path+"wisard_lut_modules.v", "w")
    else:
        text_file = open(path+"wisard_lut_modules_lw.v", "w")
    text_file.write(code)
    text_file.close()        

    

def gen_lut_overgrouped (wsrd, INDEX_WIDTH, O_WIDTH, path, coin=True):
    if coin==True:
        model = wsrd.model_bc
    else:
        model = wsrd.model
        
    # LUT v1 ##############################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output reg [O_WIDTH-1:0] out);\n\n'
    code += 'wire [%d:0] in;\n\n' % (wsrd.address_size+INDEX_WIDTH-1)
    code += 'assign in = {index, addr};\n\n'
    
    code += 'always @(in) begin\n  case (in)\n'
    
    super_unified_ram = {}
    
    for r in range(len(model[wsrd.classes[0]])):
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp = model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                bit = int((dict_tmp[a]+1)/2)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (bit<<c)
                else:
                    unified_ram[ai] = bit<<c
                    
              
        for u in unified_ram:
            u_addr = (r<<wsrd.address_size) | u
            super_unified_ram[u_addr] = unified_ram[u]
            
            # ## Binary printing - addr and data (debug)
            # u_addr = "{0:016b}".format(u_addr)
            # bin_data = "{0:010b}".format(unified_ram[u])               
            # code += '    %d\'b%s: out = %d\'b%s;\n' % (wsrd.address_size+INDEX_WIDTH,u_addr,O_WIDTH,bin_data)
            
            
            # Decimal
            # code += '    %d\'d%d: out = %d\'d%d;\n' % (wsrd.address_size+INDEX_WIDTH,u_addr,O_WIDTH,unified_ram[u])
    
    
    ##### Sort ####    
    uv = []
    dv = []
    for u in super_unified_ram:
        uv.append(u)
        dv.append(super_unified_ram[u])    
    i_sort = np.argsort(dv)
    super_unified_ram = {}    
    for i in reversed(i_sort):
        super_unified_ram[uv[i]] = dv[i]  
    ################
    
    for u in super_unified_ram:    
        code += '    %d\'d%d: out = %d\'d%d;\n' % (wsrd.address_size+INDEX_WIDTH,u,O_WIDTH,super_unified_ram[u])
    code += '    default: out = %d\'d0;\n  endcase\nend\n\nendmodule' % (O_WIDTH)
    
    if coin==True:
        text_file = open(path+"wisard_lut_overgrouped.v", "w")
    else:
        text_file = open(path+"wisard_lut_overgrouped_lw.v", "w")    
    text_file.write(code)
    text_file.close()
    


def gen_lut_gates (wsrd, INDEX_WIDTH, O_WIDTH, path):
    # LUT v3 redesign #####################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nwire [%d:0] out_v [0:%d];\n' % (O_WIDTH-1,len(wsrd.model[wsrd.classes[0]])-1)
    # code += '\nreg [%d:0] out_v [0:%d];\n' % (O_WIDTH-1,len(wsrd.model[wsrd.classes[0]])-1)
    code += 'assign out = out_v[index];\n\n'      
    
    
    for r in range(len(wsrd.model[wsrd.classes[0]])):
        code += '\n// RAM %d\n\n' % (r)
        
        # code += 'always @(*) begin\n  case (in[%d:0])\n' % (wsrd.address_size-1)
        
        unified_ram = {}
        for c in range (len(wsrd.classes)):
            dict_tmp = wsrd.model[wsrd.classes[c]][r]
            for a in dict_tmp:
                ai = int(a)
                if ai in unified_ram:
                    unified_ram[ai] = unified_ram[ai] | (1<<c)
                else:
                    unified_ram[ai] = 1<<c

        included = np.zeros((len(unified_ram)), dtype=int)
                
        i=0
        declarations='wire [%d:0] ' % (O_WIDTH-1)
        all_selected = ''
        # mux_struct = 'always @(*) begin\n'
        or_struct = 'assign out_v[%d] = ' % (r)
        for u in unified_ram:
            # u_addr = u
            # u_addr = (u + np.random.randint(0,2**1,1)[0]) % (2**14)
            # u_addr = np.random.randint(0,2**14,1)[0]
            
            # u_addr = "{0:014b}".format(u_addr)
            # code += '    %d\'b%s: out_v[%d] = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, O_WIDTH,unified_ram[u])
            
            if included[i]==0:
                included[i] = 1
                if i>0:
                    declarations += ','
                    # mux_struct += 'else '
                    or_struct += '|'
                declarations += 'hit_r%d_%d' % (r,unified_ram[u])
                selected = 'assign hit_r%d_%d = addr==%d\'d%d' % (r,unified_ram[u],wsrd.address_size,u)
                # mux_struct += 'if (hit_r%d_%d) out_v[%d] = %d\'d%d;\n' % (r,unified_ram[u], r, O_WIDTH, unified_ram[u])
                or_struct += 'hit_r%d_%d' % (r,unified_ram[u])
                
                j = 0
                for u2 in unified_ram:
                    if unified_ram[u2] == unified_ram[u] and included[j]==0:
                        selected += ' || addr==%d\'d%d' % (wsrd.address_size,u2)                    
                        included[j] = 1
                    j+=1
                selected += ' ? %d\'d%d : %d\'d0;\n' % (O_WIDTH, unified_ram[u], O_WIDTH)

                all_selected += selected
            # code += '    %d\'d%d: out_v[%d] = %d\'d%d;\n' % (wsrd.address_size,u_addr,r, O_WIDTH,unified_ram[u])
            
            i+=1
            
        code+= declarations
        code += ';\n\n'                
        code += all_selected
        code += '\n\n'   
        # code += mux_struct + 'else out_v[%d] = %d\'d0;\nend\n' % (r,O_WIDTH)            
        code += or_struct + ';\n\n'            


    code += '\nendmodule'
    
    text_file = open(path+"wisard_lut_gates.v", "w")
    text_file.write(code)
    text_file.close()       
    

    
def gen_lut_ungrouped (wsrd, INDEX_WIDTH, O_WIDTH, path, coin=True):
    if coin==True:
        model = wsrd.model_bc
    else:
        model = wsrd.model
        
    # LUT v5 Non-grouped ##################################################
    code = 'module wisard_lut\n#(parameter ADDR_WIDTH=%d, INDEX_WIDTH=%d, O_WIDTH=%d)\n' % (wsrd.address_size,INDEX_WIDTH,O_WIDTH)
    code += '(input [ADDR_WIDTH-1:0] addr, input [INDEX_WIDTH-1:0] index, output [O_WIDTH-1:0] out);\n\n'
    code += '\nreg [%d:0] out_v [0:%d];\n' % (len(model[wsrd.classes[0]])-1,len(wsrd.classes)-1)                
    
    for c in range (len(wsrd.classes)):
        code += 'assign out[%d] = out_v[%d][index];\n' % (c,c)
        
    code += '\n'
    
    for c in range (len(wsrd.classes)):
        for r in range(len(model[wsrd.classes[0]])):            
            code += 'always @(*) begin\n  case (addr)\n'             
            dict_tmp = model[wsrd.classes[c]][r]
            
            for a in dict_tmp:
                ai = int(a)
                bit = int((dict_tmp[a]+1)/2)
                if bit==1:
                    code += '    %d\'d%d: out_v[%d][%d] = 1\'b1;\n' % (wsrd.address_size,ai,c,r)
            code += '    default: out_v[%d][%d] = 1\'b0;\n  endcase\nend\n' % (c,r)
            
    code += '\nendmodule'

    if coin==True:
        text_file = open(path+"wisard_lut_ungrouped.v", "w")
    else:
        text_file = open(path+"wisard_lut_ungrouped_lw.v", "w")        
    text_file.write(code)
    text_file.close()    
