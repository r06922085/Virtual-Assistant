import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple, MultiRNNCell
import os
#model's input: do_result、xs
#        output: ys_ans
class Model_ans():
    def __init__(self, batch_size , max_len, args , dic_x_do, dic_y_do, dic_y_ans, dtype = tf.float32):
        # model parameter
        self.dtype = dtype
        self.max_len = max_len
        self.dic_x = dic_x_do
        self.dic_y_do = dic_y_do
        self.dic_y_ans = dic_y_ans
        self.val_size_x_do = len(dic_x_do)
        self.val_size_y_do = len(dic_y_do)
        self.val_size_y_ans = len(dic_y_ans)
        self.encoder_units = 16
        self.decoder_units = 32
        self.encoder_lay_Num = 3
        self.decoder_lay_Num = 3
        self.args = args
        self.batch_size = batch_size
        
        #tensor
        self.decoder_W = tf.Variable(tf.truncated_normal([self.decoder_units, self.val_size_y_ans ], 
                        stddev=1.0 / np.sqrt(self.encoder_units), dtype = self.dtype))
        self.decoder_b = tf.Variable(tf.zeros([self.val_size_y_ans], dtype = self.dtype))
        
        # feed tensor
        self.xs_PH = tf.placeholder(dtype = self.dtype, shape = [self.batch_size, self.max_len])
        self.ys_do_PH = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.max_len])
        self.ys_ans_PH = tf.placeholder(dtype = tf.int32, shape = [self.batch_size, self.max_len])
        self.inputs_length_xs_PH = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
        self.inputs_length_ys_do_PH = tf.placeholder(dtype = tf.int32, shape = [self.batch_size])
        self.inputs_length_test_xs_PH = tf.placeholder(dtype = tf.int32, shape = [1])
        self.inputs_length_test_ys_do_PH = tf.placeholder(dtype = tf.int32, shape = [1])

        # define session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)  
        
        #compile the model
        self.compile()
        
        
    def compile(self):
        encoder_output_xs = self.Encoder_xs(self.xs_PH)
        encoder_output_ys = self.Encoder_ys(self.ys_do_PH)
        encoder_output = tf.concat((encoder_output_xs,encoder_output_ys), axis = 1)
        decoder_output = self.Decoder(encoder_output)
     
        # compute model loss
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=decoder_output,
            labels=self.ys_ans_PH)
            
        self.loss = tf.reduce_mean(cross_entropy)
        
        # predict tensor
        self.prediction = tf.argmax(decoder_output, 2)
        
        # define train_op and initialize variable
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def Encoder_xs(self, input):
        with tf.variable_scope('Encoder_xs', reuse=tf.AUTO_REUSE):
            encoder_input = tf.one_hot(tf.cast(input, tf.int32), self.val_size_x_do) 
        
            encoder_input = self.WordEmb_xs(encoder_input)
            
            if self.args.train:
                inputs_length = self.inputs_length_xs_PH
            elif self.args.test:
                inputs_length = self.inputs_length_test_xs_PH
                
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
            
            encoder_output = tf.stack(sentence_code_)
            
            encoder_output = tf.layers.dense(inputs=encoder_output, units=self.encoder_units, activation=tf.nn.relu)
            

      
            return encoder_output
    
    def Encoder_ys(self, input):
        with tf.variable_scope('Encoder_ys', reuse=tf.AUTO_REUSE):
            encoder_input = tf.one_hot(tf.cast(input, tf.int32), self.val_size_y_do) 
        
            encoder_input = self.WordEmb_ys(encoder_input)
            
            if self.args.train:
                inputs_length = self.inputs_length_ys_do_PH
            elif self.args.test:
                inputs_length = self.inputs_length_test_ys_do_PH
                
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
            
            encoder_output = tf.stack(sentence_code_)
            
            encoder_output = tf.layers.dense(inputs=encoder_output, units=self.encoder_units, activation=tf.nn.relu)
            
            return encoder_output
    
    def Decoder(self, encoder_output):
        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
            encoder_output = tf.layers.dense(inputs=encoder_output, units=self.encoder_units, activation=tf.nn.relu)
            def loop_fn(time, cell_output, cell_state, loop_state):
                if cell_output is None:#time = 0
                    # initialization
                    input = tf.concat((encoder_output, encoder_output), axis = 1)
                    state = (multirnn_cell.zero_state(self.batch_size, tf.float32))
                    emit_output = None
                    loop_state = None
                    elements_finished = False
                else:
                    emit_output = cell_output
                    if self.args.test:
                        #decoder_units to val_size_y 
                        transformed_output = tf.nn.xw_plus_b(cell_output, self.decoder_W, self.decoder_b)#decoder_units to vac_size 
                        #argmax
                        transformed_output = tf.argmax(transformed_output, 1)
                        transformed_output = tf.one_hot(transformed_output, self.val_size_y_ans,on_value=1.0, off_value=0.0,axis=-1)
                        #val_size_y to decoder_units//2
                        transformed_output = self.CmdEmb(transformed_output)
                    elif self.args.train:
                        ys_onehot = tf.one_hot(self.ys_ans_PH[:,(time-1)], self.val_size_y_ans,on_value=1.0, off_value=0.0,axis=-1)
                        transformed_output = self.CmdEmb(ys_onehot)
                   
                    input = tf.concat([transformed_output, encoder_output], axis = 1)
                    state = cell_state
                    loop_state = None
                elements_finished = (time >= self.max_len)
                return (elements_finished, input, state, emit_output, loop_state)
            
            multirnn_cell = MultiRNNCell([LSTMCell(self.decoder_units) 
                for _ in range(self.decoder_lay_Num)],  state_is_tuple=True)
            emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(multirnn_cell, loop_fn)
            
            # transpose for putting batch dimension to first dimension
            outputs = tf.transpose(emit_ta.stack(), [1, 0, 2]) 
            
            #transform decoder_units to val_size_y
            decoder_output_flat = tf.reshape(outputs, [-1, self.decoder_units])
            decoder_output_transform_flat = tf.nn.xw_plus_b(decoder_output_flat, self.decoder_W, self.decoder_b)
            decoder_logits = tf.reshape(decoder_output_transform_flat, (self.batch_size, self.max_len, self.val_size_y_ans))
            
            return decoder_logits
    
    def CmdEmb(self, input):
        #emb val_size_y to decoder_unit//2
        with tf.variable_scope('CmdEmb', reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(inputs=input, 
                units=self.decoder_units//2, activation=tf.nn.relu)
                
            return output
    
    def WordEmb_xs(self, input):
        #emb val_size_x to encoder_unit
        with tf.variable_scope('wordemb_xs', reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(inputs=input, 
                units=self.encoder_units, activation=tf.nn.relu)
                
            return output
    def WordEmb_ys(self, input):
        #emb val_size_x to encoder_unit
        with tf.variable_scope('wordemb_ys', reuse=tf.AUTO_REUSE):
            output = tf.layers.dense(inputs=input, 
                units=self.encoder_units, activation=tf.nn.relu)
                
            return output
    
    def fit(self, xs, ys_do, ys_ans, batch_size, epoch):
        data_len = len(xs)
        batch_offset = 0
        ep_loss = 0
        batch_run = 0
        
        while batch_offset < (data_len-batch_size):
            _, batch_loss = self.sess.run([self.train_op, self.loss], 
              feed_dict = {self.xs_PH : xs[batch_offset:batch_offset + batch_size],
                           self.ys_do_PH : ys_do[batch_offset:batch_offset + batch_size],
                           self.ys_ans_PH : ys_ans[batch_offset:batch_offset + batch_size],
                           self.inputs_length_xs_PH : self.get_inputs_length(batch_size 
                               , xs[batch_offset:batch_offset + batch_size]),
                           self.inputs_length_ys_do_PH : self.get_inputs_length(batch_size 
                               , ys_do[batch_offset:batch_offset + batch_size])})
                                                       
            batch_offset += batch_size
            batch_run += 1
            
            
            ep_loss += batch_loss

                
        #print('epoch: {0} | ans_loss: {1:3f}'.format(epoch, ep_loss/batch_run))
        
        return (ep_loss/batch_run)
            
    def predict(self, xs, do):
        index_list = self.sess.run([self.prediction] ,  feed_dict = {
            self.xs_PH : self.create_index(xs, self.dic_x).astype(int),
            self.ys_do_PH : [do],
            self.inputs_length_test_xs_PH : 
                self.get_inputs_length(1 , self.create_index(xs, self.dic_x)),
            self.inputs_length_test_ys_do_PH : 
                self.get_inputs_length(1 , [do])})
        ans = ''
        index_list = index_list[0][0];
 
        for i in range(self.max_len):
            if index_list[i] > 3:
                ans += self.dic_y_ans[index_list[i]]
        return ans
        
      
    def get_inputs_length(self , batch_size , x):
        inputs_length = np.zeros((batch_size))
        for i in range(batch_size):
            inputs_length[i] = self.max_len 
            for ii in range(self.max_len):
                if int(x[i][ii]) == 2:
                    inputs_length[i] = int(ii+1)     
                    break                    
        return inputs_length
        
    def create_index(self , x, dic):
        test_x = np.zeros((1,self.max_len))
        for i in range(self.max_len):
            if i < len(x):
                if x[i] in dic:
                    test_x[0][i] = dic.index(x[i])
                else:
                    test_x[0][i] = 3
            elif i == len(x):
                test_x[0][i] = 2
            else:
                test_x[0][i] = 0
        return test_x
        
    def save(self):
        model_file = os.getcwd() + '/model_file/ans/model.ckpt'
        if not os.path.isdir(os.path.dirname(model_file)):
            os.mkdir('model')
        #saver = tf.train.Saver()
        self.saver.save(self.sess, model_file)
        return 

    def restore(self):
        
        model_file = os.getcwd() + '/model_file/ans/model.ckpt'
        
        if os.path.isdir(os.path.dirname(model_file)):
            #self.saver = tf.train.Saver()
            self.saver.restore(self.sess, model_file)
            
        return 