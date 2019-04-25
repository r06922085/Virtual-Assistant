import argparse
import numpy as np
import random
import sys

sys.path.append("lib/")

from lib import *
from data_format import *


class Create_data():
    def __init__(self):
        #set parameters
        self.set_parameters()
        
    def set_parameters(self):
    
        self.task_data_num = 1000
        self.task_name = data_format('task_name')
        self.task_num = len(self.task_name)
     
    def create(self):
        print('creating training data...')
        
        
        ##creating do data
        f = write_file('dataset/train_data_do.txt')
        
        for train_data_amount in range(self.task_data_num):
            for i in range(self.task_num):
                consecutive_task_num = random.randint(1,2)
                input = ''
                output = ''
                for c_t_n in range(consecutive_task_num):
                    if c_t_n == 0:
                        s, d, a = data_format(self.task_name[i])
                    else:
                        s, d, a = data_format(self.task_name[random.randint(0, len(self.task_name)-1)])
                    
                    random.shuffle(s)
                    for ii in range(len(s)):
                        random.shuffle(s[ii])
                        input += s[ii][0];
               
                    for ii in range(len(d)):
                        output += str(d[ii]);
                        output += " ";
                       
                f.write(input)
                f.write('\n')
                f.write(output)
                if(train_data_amount != self.task_data_num-1 
                    or i != self.task_num-1):
                    f.write('\n') 
        
        f.truncate()
        
        ##creating ans data
        f = write_file('dataset/train_data_ans.txt')
        
        for train_data_amount in range(self.task_data_num):
            for i in range(self.task_num):
                consecutive_task_num = random.randint(1,2)
                input = ''
                output = ''
                for c_t_n in range(consecutive_task_num):
                    if c_t_n == 0:
                        s, d, a= data_format(self.task_name[i])
                    else:
                        s, d, a= data_format(self.task_name[random.randint(0, len(self.task_name)-1)])
                
                    random.shuffle(s)
                    for ii in range(len(s)):
                        random.shuffle(s[ii])
                        input += s[ii][0];
               
                    for ii in range(len(a)):
                        rand = random.randint(0, len(a[ii])-1)
                        output += str(a[ii][rand]);
                   
                f.write(input)
                f.write('\n')
                f.write(output)
                if(train_data_amount != self.task_data_num-1 
                    or i != self.task_num-1):
                    f.write('\n') 
        
        f.truncate()
    

if __name__ == '__main__':
    main = Create_data()
    main.create()
        