# might need to put variables into the model somehow
# it would be nice if I can train and execute on a
# model in a separate folder

from __future__ import print_function

import tensorflow as tf 
import numpy as np

from dqn_model import *
from gen_tfrecord3 import parse_sequence_example

import sys
import os
from os.path import isfile, join
from datetime import datetime
import math, random

import cv2, random

BATCH_SIZE = 10

NUM_EPOCHS = 80000
GAMMA = 0.9
ALPHA = 1e-4
NUM_ITER = 5000#2500
FOLDS = 1
NUM_REMOVED = 1

TEST_ITER = 3

# alpha or 1e-3 was unsuccesful when not using BN
# see if I can play with bn_lstm parameters to get a better result

def input_pipeline(filenames):
	filename_queue = tf.train.string_input_producer(
			filenames, num_epochs=NUM_EPOCHS, shuffle=True)

	min_after_dequeue = 50 # buffer to shuffle with (bigger=better shuffeling)
	capacity = min_after_dequeue + 3 * BATCH_SIZE
	
	# deserialize is a custom function that deserializes string
	# representation of a single tf.train.Example into tensors
	# (features, label) representing single training example
	context_parsed, sequence_parsed = parse_sequence_example(filename_queue)

	seq_len = context_parsed["length"]# sequence length
	seq_len_t2 = context_parsed["length_t2"]# sequence length
	pre_lab = context_parsed["pre_act"]# label
	labels = context_parsed["act"]# label

	def extractFeature(name, data_type, sequence_parsed):
		data_t = tf.reshape(sequence_parsed[name], [-1, data_type["size"] * data_type["size"] * data_type["num_c"]])
		data_t = tf.cast(data_t, tf.int32)
		return data_t

	img_raw = extractFeature("image_raw", img_dtype, sequence_parsed)
	points = extractFeature("points", pnt_dtype, sequence_parsed)
	audio_raw = extractFeature("audio_raw", aud_dtype, sequence_parsed)

	###################################################

	img_raw_t2 = extractFeature("image_raw_t2", img_dtype, sequence_parsed)
	points_t2 = extractFeature("points_t2", pnt_dtype, sequence_parsed)
	audio_raw_t2 = extractFeature("audio_raw_t2", aud_dtype, sequence_parsed)

	pre_lab = tf.cast(pre_lab, tf.float32)
	labels_oh = tf.expand_dims(tf.one_hot(labels, 4), 0)
	lab = tf.cast(labels_oh, tf.float32)

	# Imagine inputs is a list or tuple of tensors representing single training example.
	# In my case, inputs is a tuple (features, label) obtained by reading TFRecords.
	NUM_THREADS = 1
	QUEUE_RUNNERS = 1

	inputs = [seq_len, seq_len_t2, img_raw, points, audio_raw, pre_lab, lab,  img_raw_t2, points_t2, audio_raw_t2]

	dtypes = list(map(lambda x: x.dtype, inputs))
	shapes = list(map(lambda x: x.get_shape(), inputs))

	queue = tf.RandomShuffleQueue(capacity, min_after_dequeue, dtypes)
	#queue = tf.FIFOQueue(capacity, dtypes)

	enqueue_op = queue.enqueue(inputs)
	qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)

	tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, qr)
	inputs = queue.dequeue()

	for tensor, shape in zip(inputs, shapes):
		tensor.set_shape(shape)
	
	inputs_batch = tf.train.batch(inputs, 
																BATCH_SIZE, 
																capacity=capacity,
																dynamic_pad=True
																)
															
	return inputs_batch[0], inputs_batch[1], inputs_batch[2], inputs_batch[3], inputs_batch[4], inputs_batch[5], inputs_batch[6], inputs_batch[7], inputs_batch[8], inputs_batch[9]
				 #seq_len,        #seq_len_2, 		 #img_raw,        #points,         #audio_raw,      #previous lab, 	 #lab,            #img_raw_t2,     #points_t2,      #audio_raw_t2

if __name__ == '__main__':
	#################################
	# Input params
	#################################
	ts = datetime.now()
	print("time start: ", ts)
	graphbuild = [0]*TOTAL_PARAMS
	if(len(sys.argv) > 1):
		graphbuild[int(sys.argv[1])] = 1
	else:
		graphbuild = [1]*TOTAL_PARAMS

	num_params = np.sum(graphbuild)

	

	#################################
	# Read contents of TFRecord file
	#################################
	
	path = "../tfrecords_modified/"
	
	# all files (216)
	filenames = [f for f in os.listdir(path) if isfile(join(path, f))]
	filenames = [path +x for x in filenames ]#if (x.find("flip") == -1)
	filenames.sort()

	available_sections = range(1, 12)
	accuracies = []
	folds = []
	for fold in range(FOLDS):
		test_sections = []
		for n in range(NUM_REMOVED):
			sec = int(math.floor(random.random()* len(available_sections)))
			#print(sec)
			test_sections.append(available_sections.pop(sec))
		folds.append(test_sections[:])
		testing = []
		print(test_sections)
		for n in test_sections:
			nid = str(n)
			if(len(nid) == 1):
				nid = '0'+nid
			testing.extend([f for f in filenames if (f.find("iant_"+nid+"_") >= 0 and f.find("_flip") == -1)])
		training = [f for f in filenames if f not in testing ]#and (f.find("iant_01_") >= 0)
		print(training)
		#################################
		# Generate Model
		#################################
		dqn = DQNModel(graphbuild, batch_size=BATCH_SIZE, learning_rate=ALPHA)#, filename="model.ckpt")
		dqn_hat = DQNModel(graphbuild, batch_size=BATCH_SIZE, learning_rate=ALPHA, name="dqn_hat")
	
		#################################
		# Train Model
		#################################
		
		coord = tf.train.Coordinator()

		#sequence length - slen
		#sequence length prime- slen_pr
		#image raw - i
		#points raw - p
		#audio raw - a
		#previous action - pl
		#action - l
		#image raw prime - i_pr
		#points raw prime - p_pr
		#audio raw prime - a_pr
		
		slen, slen_pr, i, p, a, pl, l, i_pr, p_pr, a_pr = input_pipeline(training)
		slen_t, slen_pr_t, i_t, p_t, a_t, pl_t, l_t, i_pr_t, p_pr_t, a_pr_t = input_pipeline(testing)
		l = tf.squeeze(l, [1])
		l_t = tf.squeeze(l_t, [1])

		dqn.sess.run(tf.local_variables_initializer())#initializes batch queue

		threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

		print("Num epochs: "+str(NUM_EPOCHS)+", Batch Size: "+str(BATCH_SIZE)+", Num Files: "+str(len(filenames))+", Num steps: "+str(NUM_ITER))
		for iteration in range(NUM_ITER):
			#get data
			n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2, pnt_data2, aud_data2 = dqn.sess.run([slen, slen_pr, i, p, a, pl, l, i_pr, p_pr, a_pr])
			'''
			print("img_data.shape: ", img_data.shape)
			img_data = np.reshape(img_data, (img_data.shape[0], img_data.shape[1], 80, 80, 3)).astype(np.uint8)
			img = img_data[0:BATCH_SIZE,0:1,][:]
			
			timg = []
			for f in range(img.shape[0]):
				mod = np.reshape(img[f], (80,80,3))
				print("img_data.shape: ", mod.shape)

				hsv = cv2.cvtColor(mod, cv2.COLOR_BGR2HSV)
				h, s, v = cv2.split(hsv)
				
				if(random.randint(0,1)):
					s += random.randint(0,5)
				else:
					s += random.randint(250,255)
				
				final_hsv = cv2.merge((h, s, v))
				mod = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

				if (len(timg) == 0):
					timg = mod
				else:
					timg = np.concatenate((timg, mod), 0)
				print(timg.shape)
			
			
			cv2.imshow('img_pre',timg)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			
			'''
			#generate partition information in order to get the prediction from LSTM
			partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
			partitions_2 = np.zeros((BATCH_SIZE, np.max(n_seq2)))

			for x in range(BATCH_SIZE):
				if(np.max(n_seq) > 0):
					v = n_seq[x]-1
					if v < 0:
						v = 0
					partitions_1[x][v] = 1
				if(np.max(n_seq2) > 0):
					v = n_seq2[x]-1
					if v < 0:
						v = 0
					partitions_2[x][v] = 1
			
			#modify rewards for non-terminal states
			if(np.max(n_seq2) > 0):
				z_line = np.sign(n_seq2)*-1 + 1
				mod_n_seq2 = n_seq2 + z_line
				
				newy = dqn_hat.sess.run(dqn_hat.get_max_q, feed_dict={
					dqn_hat.seq_length_ph: mod_n_seq2, 
					dqn_hat.img_ph: img_data2, 
					dqn_hat.pnt_ph: pnt_data2, 
					dqn_hat.aud_ph: aud_data2
					,dqn_hat.partitions_ph: partitions_2
					,dqn_hat.train_ph: False
					,dqn_hat.prompts_ph: np.sign(n_seq2)
				})

				newy *= np.sign(n_seq2)
			else:
				newy = np.zeros(BATCH_SIZE)
			
			r = np.array(label_data) # an array equal in length to batch that describes the reward being received in the next state
			
			#for x in r:
			#	if x[2] == 0:
			#		x[2] = -6
			#	if x[3] == 0:
			#		x[3] = -3
			
			r[:,0] = 0#r[:,0]*-5
			r[:,1] = r[:,1]*2
			r[:,2] = r[:,2]*10
			r[:,3] = r[:,3]*10
			
			for j in range(r.shape[0]):
				for v in range(r.shape[1]):
					if r[j][v] != 0:
						if(v < 2):
							r[j][v]+= newy[j] * GAMMA
			###
			#optimize model based on rewards
			dqn.sess.run(dqn.optimizer, feed_dict={
				dqn.seq_length_ph: n_seq, 
				dqn.img_ph: img_data, 
				dqn.pnt_ph: pnt_data, 
				dqn.aud_ph: aud_data, 
				dqn.y_ph: r,
				dqn.partitions_ph: partitions_1
				,dqn.train_ph: True
				,dqn.prompts_ph: num_prompts
				})
			
			#list the current cross entropy
			ce = dqn.sess.run(dqn.cross_entropy, feed_dict={
				dqn.seq_length_ph: n_seq, 
				dqn.img_ph: img_data, 
				dqn.pnt_ph: pnt_data, 
				dqn.aud_ph: aud_data, 
				dqn.y_ph: r
				,dqn.partitions_ph: partitions_1
				,dqn.train_ph: False
				,dqn.prompts_ph: num_prompts
				})

			print(iteration, "cross_entropy "+str(iteration)+": ", ce)

			if(iteration == NUM_ITER-1):
				pred = dqn.sess.run(dqn.pred, feed_dict={
					dqn.seq_length_ph: n_seq, 
					dqn.img_ph: img_data, 
					dqn.pnt_ph: pnt_data, 
					dqn.aud_ph: aud_data, 
					dqn.y_ph: label_data
					,dqn.partitions_ph: partitions_1
					,dqn.train_ph: False
					,dqn.prompts_ph: num_prompts
					})
				print("pred: ", pred)
				print("label: ", label_data)
				print("-------")
			
			#save the model after a 10th of the computations have completed
			if(iteration % (NUM_ITER/100) == 0):
				

				acc = dqn.sess.run(dqn.accuracy, feed_dict={
					dqn.seq_length_ph: n_seq, 
					dqn.img_ph: img_data, 
					dqn.pnt_ph: pnt_data, 
					dqn.aud_ph: aud_data, 
					dqn.y_ph: label_data
					,dqn.partitions_ph: partitions_1
					,dqn.train_ph: False
					,dqn.prompts_ph: num_prompts
					})
				print("acc of train: ", acc)

				n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2, pnt_data2, aud_data2 = dqn.sess.run([slen_t, slen_pr_t, i_t, p_t, a_t, pl_t, l_t, i_pr_t, p_pr_t, a_pr_t])

				partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
				#print("n_seq: ", n_seq)
				#print("img_data.shape: ", img_data.shape)
				#print("n_seq2: ", n_seq2)
				
				for x in range(BATCH_SIZE):
					if(np.max(n_seq) > 0):
						v = n_seq[x]-1
						if v < 0:
							v = 0
						partitions_1[x][v] = 1

				acc = dqn.sess.run(dqn.accuracy, feed_dict={
					dqn.seq_length_ph: n_seq, 
					dqn.img_ph: img_data, 
					dqn.pnt_ph: pnt_data, 
					dqn.aud_ph: aud_data, 
					dqn.y_ph: label_data
					,dqn.partitions_ph: partitions_1
					,dqn.train_ph: False
					,dqn.prompts_ph: num_prompts
					})
				print("acc of test: ", acc)
				dqn_hat.assignVariables(dqn)
				if(iteration % (NUM_ITER/10) == 0):
					#update q hat
					dqn.saveModel()
					#dqn_hat.assignVariables(dqn)
			
		#######################
		## TESTING
		#######################
		
		print("BEGIN TESTING")
		
		total_acc = 0
		for iteration in range(TEST_ITER):
			n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2, pnt_data2, aud_data2 = dqn.sess.run([slen_t, slen_pr_t, i_t, p_t, a_t, pl_t, l_t, i_pr_t, p_pr_t, a_pr_t])

			partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
			#print("n_seq: ", n_seq)
			#print("img_data.shape: ", img_data.shape)
			#print("n_seq2: ", n_seq2)
			
			for x in range(BATCH_SIZE):
				if(np.max(n_seq) > 0):
					v = n_seq[x]-1
					if v < 0:
						v = 0
					partitions_1[x][v] = 1

			acc = dqn.sess.run(dqn.accuracy, feed_dict={
				dqn.seq_length_ph: n_seq, 
				dqn.img_ph: img_data, 
				dqn.pnt_ph: pnt_data, 
				dqn.aud_ph: aud_data, 
				dqn.y_ph: label_data
				,dqn.partitions_ph: partitions_1
				,dqn.train_ph: False
				,dqn.prompts_ph: num_prompts
				})
			total_acc += acc

			pred = dqn.sess.run(dqn.pred, feed_dict={
				dqn.seq_length_ph: n_seq, 
				dqn.img_ph: img_data, 
				dqn.pnt_ph: pnt_data, 
				dqn.aud_ph: aud_data, 
				dqn.y_ph: label_data
				,dqn.partitions_ph: partitions_1
				,dqn.train_ph: False
				,dqn.prompts_ph: num_prompts
				})
			print("pred: ", pred)
			print("label: ", label_data)
			
		print("accuracy: ", total_acc/float(TEST_ITER))
		accuracies.append(total_acc/float(TEST_ITER))
		dqn.saveModel()

		te = datetime.now()
		print("time end: ", te)
		print("elapsed: ", te-ts)
		
	print("ENDING!")
	for i in range(len(folds)):
		print("subjects in test: ", folds[i], " had accuracy: ", accuracies[i])
