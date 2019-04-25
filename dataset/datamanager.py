import numpy as np
import json
import pickle
import json
from lib.lib import *

PAD = '<PAD>' # index 0
BOS = '<BOS>' # index 1
EOS = '<EOS>' # index 2
UNK = '<UNK>' # index 3

        
class DataManager():
        
        ##1.get train file##
        ##2.build dictionary##
        ##3.use dictionary to buld trainable data##
        
    def __init__(self, file_name_do, file_name_ans, max_len, batch_size):
        self.max_len = max_len
        
        self.train_x_file_do, self.train_y_file_do = self.get_train_file(file_name_do)
        self.train_x_file_ans, self.train_y_file_ans = self.get_train_file(file_name_ans)
        
        self.dic_x_do = self.build_language_dic(self.train_x_file_do)
        self.dic_y_do = self.build_command_dic(self.train_y_file_do)
        self.dic_x_ans = self.build_language_dic(self.train_x_file_ans)
        self.dic_y_ans = self.build_language_dic(self.train_y_file_ans)
        
    def get_train_data_do(self):
        train_x = []
        train_y = []
        for i in self.train_x_file_do:
            d = []
            for ii in range(self.max_len):
                if ii < len(i):
                    if i[ii] in self.dic_x_do:
                        d.append(self.dic_x_do.index(i[ii]))
                    else:
                        d.append(3)
                elif ii == len(i):
                    d.append(2)
                else:
                    d.append(0)
            train_x.append(d)
        for i in self.train_y_file_do:
            d = []
            i = i.split()
            for ii in range(self.max_len):
                if ii < len(i):
                    if i[ii] in self.dic_y_do:
                        d.append(self.dic_y_do.index(i[ii]))
                    else:
                        d.append(3)
                elif ii == len(i):
                    d.append(2)
                else:
                    d.append(0)
                    
            train_y.append(d)
        
        #change to array
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        
        return train_x, train_y, self.dic_x_do, self.dic_y_do
    
    def get_train_data_ans(self):
        train_x = []
        train_y = []
        for i in self.train_x_file_ans:
            d = []
            for ii in range(self.max_len):
                if ii < len(i):
                    if i[ii] in self.dic_x_ans:
                        d.append(self.dic_x_ans.index(i[ii]))
                    else:
                        d.append(3)
                elif ii == len(i):
                    d.append(2)
                else:
                    d.append(0)
            train_x.append(d)
        for i in self.train_y_file_ans:
            d = []
            for ii in range(self.max_len):
                if ii < len(i):
                    if i[ii] in self.dic_y_ans:
                        d.append(self.dic_y_ans.index(i[ii]))
                    else:
                        d.append(3)
                elif ii == len(i):
                    d.append(2)
                else:
                    d.append(0)
            train_y.append(d)
        
        #change to array
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        
        return train_x, train_y, self.dic_x_ans, self.dic_y_ans
    
    def get_test_data(self):
        return self.dic_x_do, self.dic_y_do, self.dic_x_ans, self.dic_y_ans
        
    def build_language_dic(self, train_x_file):
        word_list = [PAD, BOS, EOS, UNK]
        for i in train_x_file:
            for ii in i:
                if ii not in word_list:
                    word_list.append(ii)
        
        return word_list
    
    def build_command_dic(self, train_y_file):
        word_list = [PAD, BOS, EOS, UNK]
        for i in train_y_file:
            i = i.split()
            for ii in i:
                if ii not in word_list:
                    word_list.append(ii)
        
        return word_list
    
    def get_train_file(self, file_name):
        file = read_file(file_name)
        
        train_x_file = []
        train_y_file = []
        
        for i in range(len(file)):
            if i%2 == 0:
                train_x_file.append(file[i][0])
            else:
                train_y_file.append(file[i][0])
                
        return train_x_file, train_y_file

  
  