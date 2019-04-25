#"mode con cp select=950"
from lib.data_format import *
import tensorflow as tf
import numpy as np
import os
import argparse
import random
import math
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'#只有一個gpu

class main(object):
    def parse(self): 
        parser = argparse.ArgumentParser(description="chatbot")
        parser.add_argument('--model', default=None, help='model type')
        parser.add_argument('--train', action='store_true', help='whether train')
        parser.add_argument('--test', action='store_true', help='whether test')
        parser.add_argument('--keep', action='store_true', help='whether use the trained model to train')
      
        try:
            from argument import add_arguments
            parser = add_arguments(parser)
        except:
            pass
        args = parser.parse_args()
        
        return args
        
    def initialization(self, args):
        print('setting parameters...')
        
        self.file_name_do = 'dataset/train_data_do.txt'
        self.file_name_ans = 'dataset/train_data_ans.txt'
        self.max_len = 20
        
        if args.train:
            self.batch_size = 100
            
        elif args.test:
            self.batch_size = 1
        
    def train(self, args):
        print('start training...')
        
        #get train data
        train_data_do, train_data_ans = self.get_train_data(args)
        train_x_do, train_y_do, dic_x_do, dic_y_do = train_data_do
        train_x_ans, train_y_ans, dic_x_ans, dic_y_ans = train_data_ans
       
        #Graph
        g1 = tf.Graph()
        g2 = tf.Graph()
       
        #get model
        model_do = self.get_model_do(args, dic_x_do, dic_y_do)
        model_ans = self.get_model_ans(args, dic_x_do, dic_y_do, dic_y_ans)
        
   
        #start train
        epoch = 0
        min_loss_do = math.inf
        min_loss_ans = math.inf
        
        
        while True:
            epoch += 1
            train_x = self.add_noise(train_x_do, dic_x_do)
            loss_do = model_do.fit(train_x, train_y_do, self.batch_size, epoch)
            loss_ans = model_ans.fit(train_x, train_y_do, train_y_ans, self.batch_size, epoch)
            
            print('epoch: {0} | loss_do: {1:3f} | loss_ans: {2:3f}'.format(epoch, loss_do, loss_ans))
            
            #store the models
            if epoch%100 == 0:
                min_loss_do = loss_do
                model_do.save()
                min_loss_ans = loss_ans
                model_ans.save()
                
          
    def test(self, args): 
        #get command list
        command_dic = data_format('command_dic')
        
        #get test data
        train_data_do, train_data_ans = self.get_train_data(args)
        train_x_do, train_y_do, dic_x_do, dic_y_do = train_data_do
        train_x_ans, train_y_ans, dic_x_ans, dic_y_ans = train_data_ans
        
        #get model
        model_do = self.get_model_do(args, dic_x_do, dic_y_do)
        model_ans = self.get_model_ans(args, dic_x_do, dic_y_do, dic_y_ans)

        
        while True:
            ques = input('請說話...')
            
            respond_do, respond_do_digit = model_do.predict(str(ques))
            respond_ans = model_ans.predict(str(ques), respond_do_digit)
            res_do = ''
            res_ans =respond_ans
            
            for i in respond_do:
                res_do += command_dic[int(i)]
       
               
            print(res_do)
            print(res_ans)
            
    def get_train_data(self, args):
        data = DataManager(self.file_name_do, self.file_name_ans, self.max_len
            , self.batch_size)
        data_do = data.get_train_data_do()
        data_ans = data.get_train_data_ans()
        
        return data_do, data_ans
     
            
    def get_test_data(self, args):                
        return self.get_train_data(args);  

    def get_model_do(self, args, dic_x, dic_y):
        print('getting model_do...')
        
        model = Model_do(batch_size = self.batch_size , max_len = self.max_len ,
            args = args, dic_x = dic_x, dic_y = dic_y)#

        if (args.train and args.keep) or args.test:
            model.restore()   
        
        return model     
    def get_model_ans(self, args, dic_x_do, dic_y_do, dic_y_ans):
        print('getting model_ans...')
        
        model = Model_ans(batch_size = self.batch_size , max_len = self.max_len ,
            args = args, dic_x_do = dic_x_do,
            dic_y_do=dic_y_do, dic_y_ans = dic_y_ans)#

        if (args.train and args.keep) or args.test:
            model.restore()   
        
        return model          
     
    def add_noise(self, xs, xs_dic):
        PP = 0.1#possibility of adding noise
        SP = 0.05#possibility of delet data
        xs_all = []
        
        for s in range(len(xs)):
            xs_sen = np.zeros(self.max_len)
            xs_i = 0
            NoMore = False
            for w in range(self.max_len):
                #add noise
                while(random.random() < PP and xs_i < self.max_len and not NoMore):
                    xs_sen[xs_i] = random.randint(0,len(xs_dic)-1)
                    xs_i += 1;
                if(xs_i >= self.max_len):
                    break;
                #copy data by possibility
                if(xs[s][w] <= 2):
                    xs_sen[xs_i] = xs[s][w]
                    xs_i += 1;
                    NoMore = True
                else:
                    if(random.random() > SP):
                        xs_sen[xs_i] = xs[s][w]
                        xs_i += 1;
                #add noise
                while(random.random() < PP and xs_i < self.max_len and not NoMore):
                    xs_sen[xs_i] = random.randint(0,len(xs_dic)-1)
                    xs_i += 1;
                if(xs_i >= self.max_len):
                    break;
                    
            xs_all.append(xs_sen)
            
        return xs_all            
            
if __name__ == '__main__':
    assistent = main()
    
    args = assistent.parse()
    
    assistent.initialization(args)
    from model.model_do import Model_do
    from model.model_ans import Model_ans
    from dataset.datamanager import DataManager
        
    if args.train:
        assistent.train(args)
        
    elif args.test:
        assistent.test(args)            
