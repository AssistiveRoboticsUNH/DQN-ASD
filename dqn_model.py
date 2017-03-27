# might need to put variables into the model somehow
# it would be nice if I can train and execute on a
# model in a separate folder

from __future__ import print_function

import tensorflow as tf 
import numpy as np
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from bn_lstm import BNLSTMCell


MAX_BN_LEN=129
FIXED_LEN = 1
VAL=15552
c_size = 80


img_dtype = {"size": c_size, "num_c": 3}
pnt_dtype = {"size": c_size, "num_c": 1}
aud_dtype = {"size": c_size, "num_c": 3}


layer_elements = [-1, 64, 128, 128, 4]
#layer_elements = [-1, 32, 64, 4]
'''
output_sizes = [40,40,20]
filter_sizes = [4,3,2]
stride_sizes = [2,1,2]
'''
'''
output_sizes = [20,9,7]
filter_sizes = [8,4,3]
stride_sizes = [4,2,1]
'''

output_sizes = [20,9,7]
filter_sizes = [8,4,3]
stride_sizes = [4,2,1]
padding_size = [2,0,0]
'''
output_sizes = [20,9,9]
filter_sizes = [8,4,4]
stride_sizes = [4,2,2]
'''
'''
#four layer CNN
output_sizes = [40,20,9,7]
filter_sizes = [10,8,4,3]
stride_sizes = [2,2,2,1]
padding_size = [4,3,0,0]
'''


TOTAL_PARAMS = 3
num_params = 1

class DQNModel:
	# filename - saved model parameters
	def __init__(self, graphbuild = [1]*TOTAL_PARAMS, batch_size=1, filename="", name="dqn", learning_rate=1e-4):
		self.graphbuild = graphbuild
		self.__batch_size = batch_size
		self.__name = name
		self.__alpha = learning_rate

		#variables
		def weight_variable(name, shape):
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial, name=name)

		def bias_variable(name, shape):
			initial = tf.constant(0.1, shape=shape)
			return tf.Variable(initial, name=name)

		self.variables_img = {
			#[size, size, depth, output(Num_filters)]
			"W1" : weight_variable("W_conv1_img", [filter_sizes[0],filter_sizes[0],img_dtype["num_c"],layer_elements[1]]),
			"b1" : bias_variable("b_conv1_img", [layer_elements[1]]),
			"W2" : weight_variable("W_conv2_img", [filter_sizes[1],filter_sizes[1],layer_elements[1],layer_elements[2]]),
			"b2" : bias_variable("b_conv2_img", [layer_elements[2]]),
			"W3" : weight_variable("W_conv3_img", [filter_sizes[2],filter_sizes[2],layer_elements[2],layer_elements[-2]]),
			"b3" : bias_variable("b_conv3_img", [layer_elements[-2]])
		}

		self.variables_pnt = {
			"W1" : weight_variable("W_conv1_pnt", [filter_sizes[0],filter_sizes[0],pnt_dtype["num_c"],layer_elements[1]]),
			"b1" : bias_variable("b_conv1_pnt", [layer_elements[1]]),
			"W2" : weight_variable("W_conv2_pnt", [filter_sizes[1],filter_sizes[1],layer_elements[1],layer_elements[2]]),
			"b2" : bias_variable("b_conv2_pnt", [layer_elements[2]]),
			"W3" : weight_variable("W_conv3_pnt", [filter_sizes[2],filter_sizes[2],layer_elements[2],layer_elements[-2]]),
			"b3" : bias_variable("b_conv3_pnt", [layer_elements[-2]])#,
			#"W4" : weight_variable("W_conv4_pnt", [filter_sizes[2],filter_sizes[2],layer_elements[2],layer_elements[-2]]),
			#"b4" : bias_variable("b_conv4_pnt", [layer_elements[-2]])
		}

		self.variables_aud = {
			"W1" : weight_variable("W_conv1_aud", [filter_sizes[0],filter_sizes[0],aud_dtype["num_c"],layer_elements[1]]),
			"b1" : bias_variable("b_conv1_aud", [layer_elements[1]]),
			"W2" : weight_variable("W_conv2_aud", [filter_sizes[1],filter_sizes[1],layer_elements[1],layer_elements[2]]),
			"b2" : bias_variable("b_conv2_aud", [layer_elements[2]]),
			#"W3" : weight_variable("W_conv3_aud", [filter_sizes[2],filter_sizes[2],layer_elements[2],layer_elements[-2]]), ##########<<<--- for 3 layer
			"W3" : weight_variable("W_conv3_aud", [filter_sizes[2],filter_sizes[2],layer_elements[2],layer_elements[-2]]),
			"b3" : bias_variable("b_conv3_aud", [layer_elements[-2]])
		}

		self.variables_lstm = {
			#"W_lstm" : weight_variable("W_lstm", [VAL*FIXED_LEN,layer_elements[-1]]),
			"W_lstm" : weight_variable("W_lstm", [layer_elements[-2]+1,layer_elements[-1]]),
			"b_lstm" : bias_variable("b_lstm", [layer_elements[-1]])
		}

		#placeholders
		self.img_ph = tf.placeholder("float", 
			[self.__batch_size, None, img_dtype["size"] * img_dtype["size"] * img_dtype["num_c"]],
			name="img_placeholder")
		self.pnt_ph = tf.placeholder("float", 
			[self.__batch_size, None, pnt_dtype["size"] * pnt_dtype["size"] * pnt_dtype["num_c"]], 
			name="pnt_placeholder")
		self.aud_ph = tf.placeholder("float", 
			[self.__batch_size, None, aud_dtype["size"] * aud_dtype["size"] * aud_dtype["num_c"]], 
			name="aud_placeholder")
		self.seq_length_ph = tf.placeholder("int32", [self.__batch_size], name="seq_len_placeholder")
		self.partitions_ph = tf.placeholder("int32",  [self.__batch_size, None], name="partition_placeholder" )
		self.train_ph = tf.placeholder("bool", [], name="train_placeholder")
		self.prompts_ph = tf.placeholder("float32", [self.__batch_size], name="prompts_placeholder")
		
		self.y_ph = tf.placeholder("float", [None, layer_elements[-1]], name="y_placeholder")

		#model
		self.pred = self.model(self.seq_length_ph, self.img_ph, self.pnt_ph, self.aud_ph, self.partitions_ph, self.train_ph, self.prompts_ph)#

		#training
		'''
		self.diff = self.y_ph - self.pred
		self.clipped_error = tf.where(tf.abs(self.diff) < 1.0,
                                    0.5 * tf.square(self.diff),
                                    tf.abs(self.diff) - 0.5, name='clipped_error')

		self.cross_entropy = tf.reduce_mean(self.clipped_error)#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y_ph))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.__alpha).minimize(self.cross_entropy)
		'''
		self.diff = self.y_ph - self.pred
		self.cross_entropy = tf.reduce_mean(tf.square(self.diff))#tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y_ph))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.__alpha).minimize(self.cross_entropy)
		

		'''
		#Clipping the results generates nans and infs. because the CE is still nan
		opt = tf.train.AdamOptimizer(learning_rate=self.__alpha)#.minimize(self.cross_entropy)
		gvs = opt.compute_gradients(self.cross_entropy)
		#print("Gvs: ", gvs)

		capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad != None]
		self.optimizer = opt.apply_gradients(capped_gvs)
		'''

		self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y_ph,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		self.get_max_q = tf.reduce_max(self.pred,1)
		self.predict_output = tf.argmax(self.pred,1)

		#session
		self.saver = tf.train.Saver()
		self.sess = tf.Session()
		if(len(filename) == 0):
			init_op = tf.global_variables_initializer()
			self.sess.run(init_op)# remove when using a saved file
		else:
			print("RESTORING VALUES")
			self.saver.restore(self.sess, filename)

	def saveModel(self):
		self.saver.save(self.sess, "model.ckpt")

	def assignVariables(self, other_dqn):
		def copyVals(dict, other_dict):
			for key in dict:
				dict[key].assign(other_dict[key])

		copyVals(self.variables_img, other_dqn.variables_img)
		copyVals(self.variables_pnt, other_dqn.variables_pnt)
		copyVals(self.variables_aud, other_dqn.variables_aud)
		copyVals(self.variables_lstm, other_dqn.variables_lstm)
		

	def genPrediction(self, num_frames, img_data, pnt_data, aud_data, num_prompts):
		partitions = np.zeros((1, num_frames))
		#print("n_seq: ", n_seq)
		print("partitions.shape: ", partitions.shape)
		#print("n_seq2: ", n_seq2)
		partitions[0][-1] = 1
		print("num_prompts: ", num_prompts)
		#partitions = np.array([num_frames]).reshape([1, 1])
		print("partitions: ", partitions)
		with tf.variable_scope(self.__name) as scope:
			prediction = self.sess.run(self.predict_output, feed_dict={
				self.seq_length_ph: [num_frames], 
				self.img_ph: img_data, 
				self.pnt_ph: pnt_data,
				self.aud_ph: aud_data,
				self.partitions_ph: partitions,
				self.train_ph: False,
				self.prompts_ph: [num_prompts]
				})

			return prediction[0]

	def conv2d(self, x, W, S=1, P='SAME'):
		return tf.nn.conv2d(x, W, strides=[1, S, S, 1], padding=P)

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
													strides=[1, 2, 2, 1], padding='SAME')

	def process_vars(self, seq, data_type):
		#cast inputs to the correct data type
		seq_inp = tf.cast(seq, tf.float32)
		seq_inp_s = tf.reshape(seq_inp, (self.__batch_size, -1, data_type["size"], data_type["size"], data_type["num_c"]))

		return seq_inp_s

	def gen_convolved_output(self, sequence, W, b, stride, num_hidden, new_size, train_ph, padding='SAME'):
		#conv = self.conv2d(sequence, W, S=stride, P=padding) #+ b
		#conv = batch_norm(conv, is_training=train_ph) #MAKE  is_training false when passing in a single example
		#conv = tf.nn.relu(conv)
		conv = self.conv2d(sequence, W, S=stride, P=padding) 
		#conv = tf.Print(conv, [conv], message="conv output: ")
		conv = tf.nn.relu(conv + b)
		#conv = tf.sigmoid(conv + b)
		return conv

	def model(self, seq_length, img_ph, pnt_ph, aud_ph, partitions_ph, train_ph, prompts_ph):#
		with tf.variable_scope(self.__name) as scope:
			#pass different data types through conv networks
			inp_data = [0]*TOTAL_PARAMS
			conv_inp = [0]*TOTAL_PARAMS

			def pad_tf(x, p):
				q = tf.pad(x, [[0,0],[p,p],[p,p],[0,0]], "CONSTANT")
				return q

			def convolve_data(input_data, val, variables, n, dtype):
				#print("inp_data[val] 0: ", input_data.get_shape())
				input_data = tf.reshape(input_data, [-1, dtype["size"], dtype["size"], dtype["num_c"]], name=n+"_inp_reshape")
				
				input_data = pad_tf(input_data, 2)
				padding = "VALID"#"SAME"

				#print("inp_data[val] 2: ", input_data.get_shape())
				input_data = self.gen_convolved_output(input_data, variables["W1"], variables["b1"], stride_sizes[0], layer_elements[1], output_sizes[0], train_ph, padding)
				#print("conv_inp[val]: ", input_data.get_shape())
				#print("inp_data[val] 3: ", input_data.get_shape())
				#conv_inp[val] = self.gen_convolved_output(conv_inp[val], self.variables_aud["W2"], self.variables_aud["b2"], stride_sizes[1], layer_elements[2], output_sizes[1])
				
				#input_data = tf.Print(input_data, [input_data])
				#REMOVED TO SEE IF DIFFERENT SETUP IS BETTER
				input_data = self.gen_convolved_output(input_data, variables["W2"], variables["b2"], stride_sizes[1], layer_elements[2], output_sizes[1], train_ph, padding)
				



				input_data = self.gen_convolved_output(input_data, variables["W3"], variables["b3"], stride_sizes[-1], layer_elements[-2], output_sizes[-1], train_ph, padding)
				
				#input_data = self.gen_convolved_output(input_data, variables["W4"], variables["b4"], stride_sizes[-1], layer_elements[-2], output_sizes[-1], train_ph, "SAME")
				'''
				input_data = self.gen_convolved_output(input_data, variables["W1"], variables["b1"], stride_sizes[0], layer_elements[1], output_sizes[0], "SAME")
				#conv_inp[val] = self.gen_convolved_output(conv_inp[val], self.variables_img["W2"], self.variables_img["b2"], stride_sizes[1], layer_elements[2], output_sizes[1])
				input_data = self.gen_convolved_output(input_data, variables["W3"], variables["b3"], stride_sizes[-1], layer_elements[-2], output_sizes[-1], "SAME")
				#print("conv_inp[val]: ", input_data.get_shape(), output_sizes[-1]*output_sizes[-1]*layer_elements[-2])
				'''
				return tf.reshape(input_data, [-1, 1, output_sizes[-1]*output_sizes[-1]*layer_elements[-2]], name="conv_"+n+"_reshape")
				
			
			if(self.graphbuild[0]):
				val = 0
				inp_data[val] = self.process_vars(img_ph, img_dtype)
				conv_inp[val] = convolve_data(inp_data[val], val, self.variables_img, "img", img_dtype)

			if(self.graphbuild[1]):
				val = 1
				inp_data[val] = self.process_vars(pnt_ph, pnt_dtype)
				conv_inp[val] = convolve_data(inp_data[val], val, self.variables_pnt, "pnt", pnt_dtype)

			if(self.graphbuild[2]):
				val = 2
				inp_data[val] = self.process_vars(aud_ph, aud_dtype)
				conv_inp[val] = convolve_data(inp_data[val], val, self.variables_aud, "aud", aud_dtype)


			#combine different inputs together
			combined_data = None
			for i in range(TOTAL_PARAMS):

				if(self.graphbuild[i]):
					conv_inp[i] = tf.reshape(conv_inp[i], [self.__batch_size, -1, output_sizes[-1]*output_sizes[-1]*layer_elements[-2]], name="conv_reshape2")
					if(combined_data == None):
						combined_data = conv_inp[i]
					else:
						combined_data = tf.concat([combined_data, conv_inp[i]], 2)
			
			#pass combined input to rnn
			#rnn_x = self.rnn(combined_data, seq_length, self.variables_lstm)
			#combined_data = tf.reshape(combined_data,[self.__batch_size, VAL*FIXED_LEN], name="matMult_rehsape")
			#print("CD: ", combined_data.get_shape())
			
			#print("CD2: ", combined_data.get_shape())
			
			W_lstm = self.variables_lstm["W_lstm"]
			b_lstm = self.variables_lstm["b_lstm"]



			#lstm_cell = BasicLSTMCell(layer_elements[-2], forget_bias=1.0, state_is_tuple=True)
			#initial_state = state = tf.zeros([self.__batch_size, lstm_cell.state_size])
			lstm_cell = BNLSTMCell(layer_elements[-2], is_training_tensor=train_ph, max_bn_steps=MAX_BN_LEN) 
			#lstm_cell_bw = BNLSTMCell(layer_elements[-2], is_training_tensor=train_ph, max_bn_steps=MAX_BN_LEN)

			number_of_layers = 2
			stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell] * number_of_layers, state_is_tuple=True)
			#combined_data = tf.transpose(combined_data, [1,0,2])
			#print("CD2: ", combined_data.get_shape())
			#y_tensor = tf.Print(combined_data, [combined_data], message="COMBINED: ")
			outputs, states = tf.nn.dynamic_rnn(
													cell=stacked_lstm, 
													inputs=combined_data, 
													dtype=tf.float32,
													sequence_length=seq_length,
													time_major=False
													)
			
			#outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell, cell_bw=lstm_cell_bw,inputs=combined_data, dtype=tf.float32,sequence_length=seq_length)
			
			#print("outputs: ", outputs.get_shape())
			#print("partitions_ph: ", partitions_ph.get_shape())
			num_partitions = 2
			#partitions_ph = tf.transpose(partitions_ph, [1,0])
			res_out = tf.dynamic_partition(outputs, partitions_ph, num_partitions)
			#generate prediction
			#print("RO:" ,  res_out[1].get_shape())
			#x_tensor = tf.Print(res_out[1], [states], message="LSTM: ")
			#print("shape: ", res_out[1].shape)
			#x_tensor = res_out[1]
			prompts_ph = tf.reshape(prompts_ph, [-1, 1])
			x_tensor = tf.concat([res_out[1], prompts_ph], 1)
			#print("shape: ", x_tensor.shape)
			rnn_x = tf.matmul(x_tensor, W_lstm) + b_lstm
			
			#rnn_x = tf.matmul(outputs[-1], W_lstm) + b_lstm
			#rnn_x = tf.matmul(combined_data, self.variables_lstm["W_lstm"]) + self.variables_lstm["b_lstm"]
			
			return rnn_x

if __name__ == '__main__':
	dqn = DQNModel([1,0,0])

