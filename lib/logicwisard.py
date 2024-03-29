#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 06:30:30 2021

@author: Igor D. S. Miranda (igordantas@ufrb.edu.br)
"""
from wisard_base import wisard_train, wisard_eval, wisard_find_threshold, wisard_find_thresholds, wisard_eval_bin, wisard_eval_coin
import math
import os
import numpy as np
import wisard_lut_gen as wsd_lut
import copy
class logicwisard:
    """
    logicwisard is a LogicWiSARD/COIN classifier library
    This implementation can fit data to a Wisard model using dictionary or 
    cache approaches for RAM modeling. Random mapping and Sequential threshold
    are adopted.
    
    Attributes
    ----------
    classes: a list of classes string
    ram_type: 'dict' for dictionary-based RAMs or 'cache' for cache-based RAMs
    address_size: Address bit width
    mapping: Array of shuffled indexes that represents the mapping
    model: A dictionary containing the discriminator for each class
    """
    classes = []
    ram_type = 'dict'
    address_size = 4 
    min_threshold = 1
    max_threshold = 100
    best_threshold = 1
    binarized = False
    mapping = []
    model = {}
    model_hamm = {}
    model_conv = {}
    model_coin = {}
    coin_encoded_rams = []
    coin_total_minterms = 0
    coin_weights = 0
    prune_scores = {}
    
    def __init__(self, classes, address_size, min_threshold=1, max_threshold=100, ram_type = 'dict'):
        self.address_size = address_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.classes = classes
        self.ram_type = ram_type
    
    
    def fit (self, X,Y):
        """
        Fits data to a Wisard model.

        Parameters
        ----------
        X : numpy array
        A 2-dimensional array with samples in the first dimension
        and input features in the second. For example, to run a training with
        1000 vector of 16 bits each X should have (1000,16) shape. The input 
        data must be binary and the array type can have int type. 
        
        Y : numpy array
            An int array whose values correspond to the classes indexes.
        Returns
        -------
        None.
        """

        self.model, self.mapping = wisard_train (X, Y, self.classes, 
                                                 self.address_size)
        
        
    def classify (self, X, coin = False):
        """
        Runs the trained classifier on X data.

        Parameters
        ----------
        X : numpy array
            The same format as the fit input.

        Returns
        -------
        Y_pred : numpy array
            An int array whose values correspond to the classes indexes.
        """
        if coin:
            model_arg = self.model_coin
        else:
            model_arg = self.model
        
        if coin:
            Y_pred = wisard_eval_coin(X, model_arg, self.mapping, self.classes,
                                 self.address_size, threshold=self.min_threshold, 
                                 coin_weights=self.coin_weights, n_minterms=self.get_minterms_info())
        elif self.binarized:
            Y_pred = wisard_eval_bin(X, model_arg, self.mapping, self.classes, 
                                 self.address_size, thresholds=[self.min_threshold])
        else:
            Y_pred = wisard_eval(X, model_arg, self.mapping, self.classes, 
                                 self.address_size, min_threshold=self.min_threshold, 
                                 max_threshold=self.max_threshold)
        return Y_pred
    

    def find_threshold (self, X, Y, OPT_ACC_FIRST, ACC_DELTA, MINTERMS_MAX, MULTIDIM_THRESHOLD):
        """
        Find the threshold that provides the best accuracy. It starts with 
        threshold equal to 1 and stops after 5 consecutive falls.

        Parameters
        ----------
        X : numpy array
            The same format as the fit input.

        Returns
        -------
        Y_pred : numpy array
            An int array whose values correspond to the classes indexes.
        """
        if MULTIDIM_THRESHOLD:
            best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt = wisard_find_thresholds(X, Y, self.model, self.mapping, self.classes, 
                                     self.address_size, min_threshold=self.min_threshold, 
                                     max_threshold=self.max_threshold, opt_acc_first=OPT_ACC_FIRST, acc_delta=ACC_DELTA, minterms_max=MINTERMS_MAX)
            self.best_threshold = best_threshold
        else:
            best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt = wisard_find_threshold(X, Y, self.model, self.mapping, self.classes, 
                                     self.address_size, min_threshold=self.min_threshold, 
                                     max_threshold=self.max_threshold, opt_acc_first=OPT_ACC_FIRST, acc_delta=ACC_DELTA, minterms_max=MINTERMS_MAX)
            self.best_threshold = best_threshold            
        return best_acc, best_threshold, best_cnt, max_acc, max_threshold, max_cnt
    
    
    def binarize_model(self):
        """
        This function can be used to apply the threshold to all RAM values.
        The RAM positions with 0 in it is deleted from the dictionaries. 
        The final data could be coverted to vector, but it was left as dictionaries
        for compatility with the threshold models.
        """

        for c in range (len(self.classes)):  
            for r in range(len(self.model[self.classes[c]])):                
                dict_tmp = self.model[self.classes[c]][r]
                for a in dict_tmp:
                    if dict_tmp[a]>=self.best_threshold[c,r]:
                        dict_tmp[a] = 1
                    else:
                        dict_tmp[a] = 0
                self.model[self.classes[c]][r]  = {key:val for key, val in dict_tmp.items() if val != 0}
        self.min_threshold = 1
        self.max_threshold = 1
        self.binarized = True
        
        
    def get_mem_info(self):
        """
        Gets the number of words used for RAMs in the recognizers.
        It also gets the maximum value stored in the RAMs to help on memory
        word decision.
        """
        word_cnt = 0
        max_value = 0
        for c in range (len(self.classes)):        
            for r in range(len(self.model[self.classes[c]])):
                word_cnt += len(self.model[self.classes[c]][r])
                vals = self.model[self.classes[c]][r].values()
                if len(vals)>0:
                    max_t = max(vals)
                else:
                    max_t = 0
                if max_t>max_value:
                    max_value = max_t
        return word_cnt, max_value
    
    def get_minterms_info (self, bits_on=False):
        """
        Gets the number of words used for throughout recognizers after the
        minterms fusion.
        """        
        total_min = 0
        for r in range(len(self.model[self.classes[0]])):
            unified_ram = {}
            for c in range (len(self.classes)):
                dict_tmp = self.model[self.classes[c]][r]
                for a in dict_tmp:
                    ai = int(a)
                    bit = int((dict_tmp[a]+1)/2)
                    if ai in unified_ram:
                        unified_ram[ai] = unified_ram[ai] | (bit<<c)
                    else:
                        unified_ram[ai] = bit<<c
            if bits_on:
                counts = bytes(bin(x).count("1") for x in range(2**16))
            for u in unified_ram:
                if bits_on==False:
                    total_min += 1
                else:
                    total_min += counts[unified_ram[u]]

        return total_min


    def gen_coin_encode (self, X):
        """
        Gets the number of words used for throughout recognizers after the
        minterms fusion.
        """        
               
        if len(self.coin_encoded_rams)==0:
            self.coin_total_minterms = 0
            for r in range(len(self.model[self.classes[0]])):
                unified_ram = {}
                for c in range (len(self.classes)):
                    dict_tmp = self.model[self.classes[c]][r]
                    for a in dict_tmp:
                        ai = int(a)
                        if ai in unified_ram:
                            unified_ram[ai] = unified_ram[ai]
                        else:
                            unified_ram[ai] = self.coin_total_minterms
                            self.coin_total_minterms += 1
                self.coin_encoded_rams.append(unified_ram)
       
        n_samples = X.shape[0]
        X_mapped = X[:,self.mapping]                       
            
        X_coin = np.zeros((n_samples, self.coin_total_minterms), dtype=np.bool_)
        # Eval for each sample        
        for n in range (n_samples):
            xt = X_mapped[n,:].reshape(-1, self.address_size)
            xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))
            for r in range(len(self.coin_encoded_rams)):
                tuple_v = int(xti[r])
                if tuple_v in self.coin_encoded_rams[r]:
                    ind = self.coin_encoded_rams[r][tuple_v]
                    X_coin[n, ind] = 1

        return X_coin
    
    def create_model_from_coin (self, weights):
        self.coin_weights = weights
        
        self.model_coin = copy.deepcopy(self.model)
        
        for r in range(len(self.model_coin[self.classes[0]])):
            for a in self.coin_encoded_rams[r]:
                ai = int(a)
                ind_w = self.coin_encoded_rams[r][ai]
                for c in range (len(self.classes)):
                    self.model_coin[self.classes[c]][r][ai] = weights[0][ind_w, c]

    def create_coin_from_model (self, weights):
        delta = 0.001
        self.coin_weights = copy.deepcopy(weights)
        
        for r in range(len(self.model[self.classes[0]])):
            for a in self.coin_encoded_rams[r]:
                ai = int(a)
                ind_w = self.coin_encoded_rams[r][ai]
                for c in range (len(self.classes)):
                    if ai in self.model[self.classes[c]][r]:
                        #self.coin_weights[0][ind_w, c] = 2*(self.model[self.classes[c]][r][ai] - 0.5)
                        self.coin_weights[0][ind_w, c] = 1*(self.model[self.classes[c]][r][ai] - 0)
                    else:
                        self.coin_weights[0][ind_w, c] = 0

        return self.coin_weights
    
    def update_pruning_scores(self, xti, label_class):

        for c in range (len(self.classes)):   # Classes/discriminators     
            for r in range(len(self.prune_scores[self.classes[c]])):  # RAMs in a class
                pos = xti[r]
                if pos in self.model_coin[self.classes[c]][r]:
                    if c==label_class and self.model_coin[self.classes[c]][r][pos]==1:
                        self.prune_scores[self.classes[c]][r][pos] -= 1
                    elif c!=label_class and self.model_coin[self.classes[c]][r][pos]==-1:
                        self.prune_scores[self.classes[c]][r][pos] -= 1
                    else:
                        self.prune_scores[self.classes[c]][r][pos] += 1
                # else:
                #     self.prune_scores[self.classes[c]][r][pos] +=1

                
    def pruning (self, X, Y, keep_ratio):
        """
        Gets the score of each RAM position on each discriminator.
        """
        self.prune_scores = copy.deepcopy(self.model_coin)
        
        # Zeroing all positions
        for c in range (len(self.classes)):   # Classes/discriminators     
            for r in range(len(self.prune_scores[self.classes[c]])):  # RAMs in a class
                for pos in self.prune_scores[self.classes[c]][r]:   # Positions in a RAM 
                    self.prune_scores[self.classes[c]][r][pos] = 0
        
        # Assign scores
        for n in range (X.shape[0]):
            xt = X[n,:].reshape(-1, self.address_size)
            xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))            
            self.update_pruning_scores(xti, Y[n])
            
        # Zeroing all positions
        for c in range (len(self.classes)):   # Classes/discriminators     
            for r in range(len(self.prune_scores[self.classes[c]])):  # RAMs in a class    
                vals = self.prune_scores[self.classes[c]][r].values()
                sorted_i = np.argsort(vals)
                last = int(keep_ratio*len(vals))
                threshold = vals[sorted_i[last]]
                dict_model = self.model_coin[self.classes[c]][r]
                dict_scores = self.prune_scores[self.classes[c]][r]
                
                for it in dict_scores:
                    if dict_scores[it]>=threshold:
                        dict_model[it] = 0
                    
                self.model_coin[self.classes[c]][r]  = {key:val for key, val in dict_model.items() if val != 0}
  
    def export2verilog(self, path, X, Y, coin=True, export_data=False):
        """
        Exports model to verilog RTL, creates a testbech and exports the data.

        Parameters
        ----------
        path : String
            Directory where the verilog and data files will be placed.
        X : TYPE
            Input data.
        Y : TYPE
            Output data.            

        Returns
        -------
        code : String
            Verilog code.

        """
        
        ## Exporting verilog code
        x_dim = X.shape[1]
        N_INDEXES = int(x_dim/self.address_size)
        INDEX_WIDTH = math.ceil(math.log2(N_INDEXES))
        I_WIDTH = self.address_size + INDEX_WIDTH
        
        O_WIDTH = len(self.classes)
        N_RAMS = int(math.ceil(x_dim/self.address_size))
        # Mapping
        
        code = 'module wisard_mapping\n'
        code += '#(parameter ADDRESS_WIDTH = %d, INDEX_WIDTH=%d)\n' % (self.address_size, INDEX_WIDTH)
        code += '(input clk,\ninput rst_n,\ninput sink_sop,\ninput sink_valid,\ninput sink_eop,\n'
        code += 'input [ADDRESS_WIDTH-1:0] addr,\ninput [INDEX_WIDTH-1:0] index,\n'
        code += 'output reg source_sop,\noutput source_valid, \noutput reg source_eop,\n'
        code += 'output reg [ADDRESS_WIDTH-1:0] source_addr, \noutput [INDEX_WIDTH-1:0] source_index);\n\n'        
        code += 'localparam N_INDEXES = %d;\n\n' % (N_INDEXES)
        
        text_file = open('../lib/templates/mapping_v_fragment.txt')
        code += text_file.read()
        text_file.close()        
        
        for m in range (len(self.mapping)):
            code += 'assign out_mem_flat[%d] = in_mem_flat[%d];\n' % (m, self.mapping[m])
                
        code += '\nendmodule'
        
        text_file = open(path+"wisard_mapping.v", "w")
        text_file.write(code)
        text_file.close()        
        

        # Generate verilog LUTs
        wsd_lut.gen_lut_grouped (self, INDEX_WIDTH, O_WIDTH, path, coin=coin)
        
        ## Transfering template files
        os.system("cp ../lib/templates/*.v "+path)
        os.system("cp ../lib/templates/*.s* "+path)
        if coin:
            os.system("rm "+path+"/lw*")
        else:
            os.system("mv "+path+"/lw_wisard.v "+path+"/wisard.v")
        
        ## Set testbench parameters
        tb_params = 'localparam ADDRESS_WIDTH = %d;\n' % (int(self.address_size))
        tb_params += 'localparam N_RAMS = %d;\n' % (N_RAMS)
        tb_params += 'localparam INDEX_WIDTH = %d;\n' % (int(math.ceil(math.log2(x_dim/self.address_size))))
        tb_params += 'localparam N_CLASSES = %d;\n' % (len(self.classes))
        tb_params += 'localparam CLASS_WIDTH = %d;\n' % (int(math.ceil(math.log2(len(self.classes)))))
        tb_params += 'localparam N_INPUTS = %d;\n' % (X.shape[0])
                
        # For parallel implementation
        text_file = open(path+'/tb_wisard_parallel.v')
        tb_param_v = text_file.read()
        text_file.close()
        tb_param_v = tb_param_v.replace('//__AUTO_PARAMETERS__', tb_params)
        text_file = open(path+'/tb_wisard_parallel.v', "w")
        text_file.write(tb_param_v)
        text_file.close()
        
        # For serial implementation
        text_file = open(path+'/tb_wisard_serial.v')
        tb_param_v = text_file.read()
        text_file.close()
        tb_param_v = tb_param_v.replace('//__AUTO_PARAMETERS__', tb_params)
        text_file = open(path+'/tb_wisard_serial.v', "w")
        text_file.write(tb_param_v)
        text_file.close()        
        
        ## Set wisard parameters
        if coin:        
            text_file = open(path+'/wisard.v')
            wsd_param_v = text_file.read()
            text_file.close()
            wsd_param_v = wsd_param_v.replace('__ADDRESS_WIDTH__', str(int(self.address_size)))
            index_width_t = int(math.ceil(math.log2(x_dim/self.address_size)))
            wsd_param_v = wsd_param_v.replace('__INDEX_WIDTH__', str(index_width_t))
            wsd_param_v = wsd_param_v.replace('__N_RAMS__', str(N_RAMS)) 
            wsd_param_v = wsd_param_v.replace('__N_CLASSES__', str(len(self.classes)))
            wsd_param_v = wsd_param_v.replace('__CLASS_WIDTH__', str(int(math.ceil(math.log2(len(self.classes))))))
            
            text_file = open(path+'/wisard.v', "w")
            text_file.write(wsd_param_v)
            text_file.close()


            # Tau insertion in wisard_crtl
            from tau_gen import tau_gen
            text_file = open(path+'/wisard_ctrl.v')
            wsd_param_ctrl_v = text_file.read()
            text_file.close()
            tau = tau_gen (self.coin_weights, self.get_minterms_info(), len(self.classes))
            tau_width = index_width_t+1
            tau_str = "wire signed [%d:0] TAU [0:%d];\n" % (tau_width-1, len(self.classes)-1)
            tau_str += "wire signed [%d:0] TAU_PLUS1 [0:%d];\n" % (tau_width-1, len(self.classes)-1)
            tau_str += "wire signed [%d:0] TAU_MINUS1 [0:%d];\n" % (tau_width-1, len(self.classes)-1)
            
            for i in range(len(tau)):
               sign = '-' if tau[i] >0 else ''
               tau_str += "assign TAU[%d] = %s%d'd%d;\n" % (i,sign,tau_width, abs(tau[i]))
               sign = '-' if tau[i]-1 >0 else ''
               tau_str += "assign TAU_PLUS1[%d] = %s%d'd%d;\n" % (i,sign, tau_width, abs(tau[i] - 1))
               sign = '-' if tau[i]+1 >0 else ''
               tau_str += "assign TAU_MINUS1[%d] = %s%d'd%d;\n" % (i,sign, tau_width, abs(tau[i] + 1))
            wsd_param_ctrl_v = wsd_param_ctrl_v.replace('__TAU_PARAMETERS__', tau_str)
            text_file = open(path+'/wisard_ctrl.v', "w")
            text_file.write(wsd_param_ctrl_v)
            text_file.close()            
        
        ## Exporting data
        if export_data:
            os.system("rm -rf "+path+"data")
            os.system("mkdir "+path+"data")
            n_samples = X.shape[0]
            X_mapped = X[:,self.mapping]
            # X_mapped = X
            txt_o = ''
            for n in range (n_samples):
                xt = X_mapped[n,:].reshape(-1, self.address_size)
                # xt = np.flip(xt, axis=1) # flip to correct endianess
                xti = xt.dot(1 << np.arange(xt.shape[-1] - 1, -1, -1))         
                txt_i = ''            
                for m in range (len(xti)):
                    txt_i += '%08x\n' % (int(xti[m]))
    
                txt_o += str(int(Y[n]))+'\n'
                fname = "data/in%d.txt" % (n)    
                text_file = open(path+fname, "w")
                text_file.write(txt_i)
                text_file.close()

            
            fname = "data/y_pred_sw.txt"
            text_file = open(path+fname, "w")
            text_file.write(txt_o)
            text_file.close()


    def export2python(self, path, coin):
        """
        Exports model to verilog RTL, creates a testbech and exports the data.

        Parameters
        ----------
        path : String
            Directory where the verilog and data files will be placed.
        X : TYPE
            Input data.
        Y : TYPE
            Output data.            

        Returns
        -------
        code : String
            Verilog code.

        """
        O_WIDTH = len(self.classes)
        
        ## Exporting verilog code
        wsd_lut.gen_lut_grouped_python (self, O_WIDTH, path, coin=coin)
        
