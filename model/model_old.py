import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple, MultiRNNCell
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #共4個gpu

"""Usage
Declare a Seq2Seq instance : model = Seq2Seq(vac_size)
Compile model : model.compile()
Train model : model.fit(xs, ys, batch_size, epoch)
"""
class Model():
    def __init__(self, batch_size , max_len, args , dictionary_x, dictionary_y, dtype = tf.float32):
        # model parameter
        self.dtype = dtype
        self.max_len = max_len
        self.dictionary_x = dictionary_x
        self.dictionary_y = dictionary_y
        self.val_size_x = len(dictionary_x)
        self.val_size_y = len(dictionary_y)
        self.encoder_units = self.val_size_x*2
        self.encoder_lay_Num = 3
        self.args = args
        self.batch_size = batch_size
        
        # feed tensor
        self.xs_PH = tf.placeholder(dtype = self.dtype, shape = [self.batch_size, self.max_len])
        self.ys_PH = tf.placeholder(dtype = self.dtype, shape = [self.batch_size, self.max_len])
        self.inputs_length_PH = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
        self.inputs_length_test_PH = tf.placeholder(dtype = tf.int32, shape = [1])

        # define session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)  
        
        #compile the model
        self.compile()
        
    def Encoder(self, xs):
        encoder_input = tf.one_hot(tf.cast(xs, tf.int32), self.val_size) 
    
        encoder_input = tf.layers.dense(inputs=encoder_input, 
            units=self.encoder_units, activation=tf.nn.relu)
        
        if self.args.train:
            inputs_length = self.inputs_length_PH
        elif self.args.test:
            inputs_length = self.inputs_length_test_PH
            
        multirnn_cell = MultiRNNCell([LSTMCell(self.encoder_units) 
            for _ in range(self.encoder_lay_Num)],  state_is_tuple=True)
            
        (fw_outputs, bw_outputs), (fw_final_state, bw_final_state) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=multirnn_cell, 
                                            cell_bw=multirnn_cell, inputs=encoder_input,
                                            sequence_length=inputs_length, dtype=self.dtype))
                                            
        sentence_code = tf.concat((fw_outputs, bw_outputs), axis = 2)
        
      
        sentence_code_ = []
        for i in range(self.batch_size):
            sentence_code_.append(sentence_code[i,inputs_length[i]-1,:])
            
        
        sentence_code = tf.stack(sentence_code_)
        
        encoder_output = tf.layers.dense(inputs=sentence_code, units=self.encoder_units, activation=tf.nn.relu)
        encoder_output = tf.layers.dense(inputs=encoder_output, units=self.encoder_units, activation=tf.nn.relu)
        encoder_output = tf.layers.dense(inputs=encoder_output, units=self.classification_num, activation=tf.nn.relu)
        encoder_output = tf.layers.dense(inputs=encoder_output, units=self.classification_num, activation=tf.nn.relu)
        
        return encoder_output
    
    def compile(self):
    
        encoder_output = self.Encoder(self.xs_PH)
     
        # compute model loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=encoder_output,
            labels=self.ys_PH)
            
        self.loss = tf.reduce_mean(cross_entropy)
        
        # predict tensor
        self.prediction = tf.argmax(encoder_output, 1)
        
        # define train_op and initialize variable
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        
    def fit(self, xs, ys, batch_size, epoch):
    
        data_len = len(xs)
        batch_offset = 0
        ep_loss = 0
        batch_run = 0
        
        while batch_offset < (data_len-batch_size):
            _, batch_loss = self.sess.run([self.train_op, self.loss], 
                                          feed_dict = {self.xs_PH : xs[batch_offset:batch_offset + batch_size],
                                                       self.ys_PH : ys[batch_offset:batch_offset + batch_size],
                                                       self.inputs_length_PH : self.get_inputs_length(batch_size , xs[batch_offset:batch_offset + batch_size])})
                                                       
            batch_offset += batch_size
            batch_run += 1
            
            #print('loss: {0:3f} | finish: {1:3f}%'.format(batch_loss, (batch_offset/data_len)*100))
            
            ep_loss += batch_loss

                
        print('epoch: {0} | loss: {1:3f}'.format(epoch, ep_loss/batch_run))
        
        return (ep_loss/batch_run)
            
            
    def predict(self, x):
        index = self.sess.run([self.prediction] ,  feed_dict = {
            self.xs_PH : self.create_index(x),
            self.inputs_length_test_PH : self.get_inputs_length(1 , self.create_index(x))})
                     
        return index[0][0]
        
    def get_inputs_length(self , batch_size , x):
        inputs_length = np.zeros((batch_size))
        for i in range(batch_size):
            inputs_length[i] = self.max_len 
            for ii in range(self.max_len):
                if int(x[i][ii]) == 2:
                    inputs_length[i] = int(ii)                    
                    break                    
        return inputs_length
        
    def create_index(self , x):
        test_x = np.zeros((1,self.max_len))
        
        for i in range(self.max_len):
            if i < len(x):
                if x[i] in self.dictionary:
                    test_x[0][i] = self.dictionary.index(x[i])
                else:
                    test_x[0][i] = 3
            elif i == len(x):
                test_x[0][i] = 2
            else:
                test_x[0][i] = 0
        
        #clean the unknown word
        test_x_ = np.zeros((1,self.max_len))        
        index = 0
        for i in range(self.max_len):
            if test_x[0][i] == 3:
                pass
            else:
                test_x_[0][index] = test_x[0][i]
                index += 1
        test_x = test_x_        
                
                
        return test_x
        
    def save(self):
        save_path = os.getcwd() + '/model_file/domain_classification/model.ckpt'
        
        self.saver.save(self.sess, save_path)
        
        return 
        

    def restore(self):
        
        model_file = os.getcwd() + '/model_file/domain_classification/model.ckpt'
        
        if os.path.isdir(os.path.dirname(model_file)):
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, model_file)
            
        return 