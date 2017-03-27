#!/usr/bin/env python
import tensorflow as tf
import numpy as np
from dqn_packager import * 
import rosbag
import time

import os
from os.path import isfile, join

def make_sequence_example(image_raw, image_data, points, point_data, 
		audio_raw, audio_data, pre_act, act, pos_act, state, image_raw_t2, 
		points_t2, audio_raw_t2):
	#print("write: ")

	#converts a sequence and its labels into a SequenceExample
	image_raw = np.squeeze(image_raw, axis=0) # only necessary to get the seq lengths
	image_raw_t2 = np.squeeze(image_raw_t2, axis=0) # only necessary to get the seq lengths


	# The object we return
	ex = tf.train.SequenceExample()
	# A non-sequential feature of our example
	sequence_length = len(image_raw)
	sequence_length_t2 = len(image_raw_t2)

	#print("#########actions: ", pre_act, act, pos_act)
	#print(sequence_length, len(image_raw), image_raw)
	#print(sequence_length_t2, len(image_raw_t2), image_raw_t2)

	ex.context.feature["length"].int64_list.value.append(sequence_length)
	ex.context.feature["length_t2"].int64_list.value.append(sequence_length_t2)
	ex.context.feature["img_h"].int64_list.value.append(image_data["cmp_h"])
	ex.context.feature["img_w"].int64_list.value.append(image_data["cmp_w"])
	ex.context.feature["img_c"].int64_list.value.append(image_data["num_c"])
	ex.context.feature["pnt_h"].int64_list.value.append(point_data["cmp_h"])
	ex.context.feature["pnt_w"].int64_list.value.append(point_data["cmp_w"])
	ex.context.feature["pnt_c"].int64_list.value.append(point_data["num_c"])
	ex.context.feature["aud_h"].int64_list.value.append(audio_data["cmp_h"])
	ex.context.feature["aud_w"].int64_list.value.append(audio_data["cmp_w"])
	ex.context.feature["aud_c"].int64_list.value.append(audio_data["num_c"])
	ex.context.feature["pre_act"].int64_list.value.append(pre_act)# act perfomed prior to s
	ex.context.feature["act"].int64_list.value.append(act)#act performed in s
	ex.context.feature["pos_act"].int64_list.value.append(pos_act)# act performed in s'
	ex.context.feature["state"].int64_list.value.append(state)
	# Feature lists for the two sequential features of our example

	def load_array(example, name, data):
		fl_data = example.feature_lists.feature_list[name].feature.add().bytes_list.value
		fl_data.append(np.asarray(data).astype(np.uint8).tostring())

	load_array(ex, "image_raw", image_raw)
	load_array(ex, "points", points)
	load_array(ex, "audio_raw", audio_raw)
	load_array(ex, "image_raw_t2", image_raw_t2)
	load_array(ex, "points_t2", points_t2)
	load_array(ex, "audio_raw_t2", audio_raw_t2)

	return ex

def parse_sequence_example(filename_queue):
	#reads a TFRecord into its constituent parts
	reader = tf.TFRecordReader()
	_, example = reader.read(filename_queue)
	
	context_features = {
		"length": tf.FixedLenFeature([], dtype=tf.int64),
		"length_t2": tf.FixedLenFeature([], dtype=tf.int64),
		"img_h": tf.FixedLenFeature([], dtype=tf.int64),
		"img_c": tf.FixedLenFeature([], dtype=tf.int64),
		"pnt_h": tf.FixedLenFeature([], dtype=tf.int64),
		"pnt_c": tf.FixedLenFeature([], dtype=tf.int64),
		"pre_act": tf.FixedLenFeature([], dtype=tf.int64),
		"act": tf.FixedLenFeature([], dtype=tf.int64),
		"pos_act": tf.FixedLenFeature([], dtype=tf.int64),
		"state": tf.FixedLenFeature([], dtype=tf.int64)
	}
	sequence_features = {
		"image_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"points": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"audio_raw": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"image_raw_t2": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"points_t2": tf.FixedLenSequenceFeature([], dtype=tf.string),
		"audio_raw_t2": tf.FixedLenSequenceFeature([], dtype=tf.string)
	}
	
	# Parse the example
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
		serialized=example,
		context_features=context_features,
		sequence_features=sequence_features
	)
	
	sequence_data = {
		"image_raw": tf.decode_raw(sequence_parsed["image_raw"], tf.uint8),
		"points": tf.decode_raw(sequence_parsed["points"], tf.uint8),
		"audio_raw": tf.decode_raw(sequence_parsed["audio_raw"], tf.uint8),
		"image_raw_t2": tf.decode_raw(sequence_parsed["image_raw_t2"], tf.uint8),
		"points_t2": tf.decode_raw(sequence_parsed["points_t2"], tf.uint8),
		"audio_raw_t2": tf.decode_raw(sequence_parsed["audio_raw_t2"], tf.uint8)
	}
	
	return context_parsed, sequence_data

def gen_TFRecord_from_file(writer, bagfile, state):
	packager = DQNPackager()
	mostrecent_act = -1
	history = []
	bag = rosbag.Bag(bagfile)	
	actions = []
	packager.p = False
	for topic, msg, t in bag.read_messages(topics=topic_names):
		if(topic == topic_names[0]):
			actions.append(msg.data)
			print("act_list: ", actions)
			print("---")
			packager.actCallback(msg)
			#print("mostrecent_act: ", msg.data)
			
			if(len(actions) >= 2):
				packager.formatOutput()
				if(len(actions) >= 3):
					#add frame to record
					
					pre_act = actions[-3]
					act = actions[-2]
					next_act = actions[-1]
					print(pre_act, act, next_act)
					print("history: ", len(history))
					
					
					ex = make_sequence_example (history[0], image_data, history[1], point_data, 
						history[2], audio_data, pre_act, act, next_act, state,
						packager.getImgStack(), packager.getPntStack(), packager.getAudStack())
					#writer.write(ex.SerializeToString())
					
				history = [packager.getImgStack(), packager.getPntStack(), packager.getAudStack()]

			if(msg.data == 2 or msg.data == 3):# break if terminate action
				break
			else:
				packager.reset()

			print("================")
		elif(topic == topic_names[1]):
			packager.imgCallback(msg)
		elif(topic == topic_names[2]):
			packager.pntCallback(msg)
		elif(topic == topic_names[3]):
			packager.audCallback(msg)
	pre_act = actions[-2]
	act = actions[-1]
	next_act = -1
	print(pre_act, act, next_act)
	print("history: ", len(history))
	print("packager.getImgStack(): ", packager.getImgStack().shape)
	
	ex = make_sequence_example (packager.getImgStack(), image_data, packager.getPntStack(), point_data, 
		packager.getAudStack(), audio_data, pre_act, act, next_act, state, [], [], [])
	writer.write(ex.SerializeToString())
	packager.reset()
	
	bag.close()


if __name__ == '__main__':
	rospy.init_node('gen_tfrecord', anonymous=True)
	
	bagfile = os.environ["HOME"] + "/Documents/AssistiveRobotics/AutismAssistant/pomdpData/sub01/compliant/cur_nao_asd_auto_2016-11-18-14-58-13.bag"#"test.bag"#
	outfile = "../tfrecords/test_records/c0_299.tfrecord"#"img.tfrecord"

	
	#
	#

	#'../tfrecords/compliant_01_0.tfrecord', 
	#,'../tfrecords/compliant_01_2.tfrecord', cur_nao_asd_auto_2016-11-18-14-58-50.bag
	#,'../tfrecords/compliant_01_5.tfrecord', cur_nao_asd_auto_2016-11-18-14-59-46.bag

	state = 1

	rospy.init_node('gen_tfrecord', anonymous=True)
	write = True
	if (write):
		writer = tf.python_io.TFRecordWriter(outfile)
		gen_TFRecord_from_file(writer, bagfile, state)
		writer.close()
	
	
	c_size = 299
	img_dtype = {"size": c_size, "num_c": 3}
	pnt_dtype = {"size": c_size, "num_c": 1}
	aud_dtype = {"size": c_size, "num_c": 3}

	'''
	print("READING...")
	coord = tf.train.Coordinator()
	filename_queue = tf.train.string_input_producer([outfile])
	
	with tf.Session() as sess:
		sess.run(tf.local_variables_initializer())
		context_parsed, sequence_parsed = parse_sequence_example(filename_queue)
		threads = tf.train.start_queue_runners(coord=coord)
		
		seq_len = context_parsed["length"]# sequence length
		seq_len2 = context_parsed["length_t2"]# sequence length
		labels = context_parsed["act"]# label
		labels2 = context_parsed["pos_act"]# label

		def processData(inp, data_type):
			data_s = tf.reshape(inp, [-1, data_type["size"] * data_type["size"] * data_type["num_c"]])
			return tf.cast(data_s, tf.uint8)
		
		img_raw = processData(sequence_parsed["image_raw"], img_dtype)
		#points = processData(sequence_parsed["points"], pnt_dtype)
		#audio_raw = processData(sequence_parsed["audio_raw"], aud_dtype)
		#img_raw_t2 = processData(sequence_parsed["img_raw_t2"], img_dtype)
		#points_t2 = processData(sequence_parsed["points_t2"], pnt_dtype)
		#audio_raw_t2 = processData(sequence_parsed["audio_raw_t2"], aud_dtype)
		
		for i in range(2):
			l, a= sess.run([labels, img_raw])
			
			imf = np.reshape(a[0], (aud_dtype["size"], aud_dtype["size"], aud_dtype["num_c"]))
			#print("imf.shape:", imf)
			cv2.imshow('imf',imf)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			def f(x):
				return len(x)
			
			print("labels: ", l)
		#s, s2, i, p, a, i2, p2, a2, l, l2 = sess.run([seq_len, seq_len2, img_raw, points, audio_raw, img_raw_t2, points_t2, audio_raw_t2, labels, labels2])
		
	
	
	
	coord.request_stop()
	coord.join(threads)
	'''
	'''
	#convert all sessions to individual files
	outdir = "../tfrecords_modified/"

	subdir = ['compliant/', 'noncompliant/']
	for i in range(1,12):
		filenum = str(i)
		if(i < 10):
			filenum = '0'+filenum
		for s in range(len(subdir)):
			tail = "sub"+filenum+"/"+subdir[s]
			pathfromhome = "Documents/AssistiveRobotics/AutismAssistant/pomdpData/"
			path = os.environ["HOME"]+'/'+pathfromhome+tail
			

			files = [f for f in os.listdir(path) if isfile(join(path, f))]
			print("sub"+str(i) + ' ' + subdir[s])
			files = [x for x in files if (x.find("cur_") > -1)]
			
			# execute file using: $ rosrun asdpomdp src/convertbag2wav.py
			files.sort()

			count = 0
			for f in files:
				outfile = outdir+subdir[s][:-1]+"_"+filenum+"_"+str(count)+"_mod_flip.tfrecord"
				#today = time.mktime(time.strptime("28 Feb 17", "%d %b %y"))
				#if(os.stat(outfile).st_mtime < today):
				print("processing..."+f)
				#print(subdir[s][:-1]+"_"+filenum+"_"+str(count), os.stat(outfile).st_mtime)
				writer = tf.python_io.TFRecordWriter(outfile)
				gen_TFRecord_from_file(writer, path+f, (s+1)%2)
				writer.close()
				count += 1
			
		
			
	'''
	'''
	outdir = "../tfrecords/"
	outfile = outdir+"asd_sessions_aud.tfrecord"
	writer = tf.python_io.TFRecordWriter(outfile)
	
	filenames = [f for f in os.listdir(outdir) if isfile(join(path, f))]
	filenames = [os.path.join(os.getcwd(), x) for x in filenames if (x.find("asd") == -1)]
	
	filenames.sort()

	f in filenames:
		coord = tf.train.Coordinator()
		filename_queue = tf.train.string_input_producer([f])

		with tf.Session() as sess:
			sess.run(tf.initialize_local_variables())
			context_parsed, sequence_parsed = parse_sequence_example([filename_queue])
			threads = tf.train.start_queue_runners(coord=coord)

	writer.close()
	'''

	'''
	#convert all sessions to single file
	outdir = "../tfrecords/"
	outfile = outdir+"asd_sessions_aud.tfrecord"
	writer = tf.python_io.TFRecordWriter(outfile)
	subdir = ['compliant/', 'noncompliant/']

	for i in range(1,12):
		filenum = str(i)
		if(i < 10):
			filenum = '0'+filenum
		for s in range(len(subdir)):
			tail = "sub"+filenum+"/"+subdir[s]
			pathfromhome = "Documents/AssistiveRobotics/AutismAssistant/pomdpData/"
			path = os.environ["HOME"]+'/'+pathfromhome+tail
			
			files = [f for f in os.listdir(path) if isfile(join(path, f))]
			print("sub"+str(i) + ' ' + subdir[s])
			files = [x for x in files if (x.find("cur_") > -1)]
			
			# execute file using: $ rosrun asdpomdp src/convertbag2wav.py
			files.sort()

			count = 0
			for f in files:
				print("processing..."+f)				
				gen_TFRecord_from_file(writer, path+f, (s+1)%2)

				count += 1
	writer.close()
	'''