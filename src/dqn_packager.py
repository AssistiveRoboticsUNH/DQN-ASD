# collects observations and passes them into a DQN model in order
# to return the next action that should be performed
import tensorflow as tf
import numpy as np

import rospy

from std_msgs.msg import Int8

from sensor_msgs.msg import Image, PointCloud2
from deep_q_network.srv import PointCloud2Array

import cv2
from cv_bridge import CvBridge, CvBridgeError

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nao_msgs.msg import AudioBuffer
import tempfile
from scipy import signal

from dqn_model import DQNModel

import threading

# Constants
service_names = [
	('batch_convert_pointcloud', PointCloud2Array)
]

topic_names = [
	'/action_finished',
	'/nao_robot/camera/top/camera/image_raw',
	'/camera/depth/points',
	'/nao_robot/microphone/naoqi_microphone/audio_raw'
]

c_size = 80

image_data = {
	"img_h": 480, 
	"img_w": 640, 
	"num_c": 3,
	"cmp_h": c_size,
	"cmp_w": c_size
}

point_data = {
	"img_h": 480, 
	"img_w": 640,
	"num_c": 1,
	"cmp_h": c_size,
	"cmp_w": c_size
}

audio_data = {	
	"img_h": 480, 
	"img_w": 640,
	"num_c": 3,
	"cmp_h": c_size,
	"cmp_w": c_size
}

'''
DQN Packager observes topics for images, points, and audio.
Processes those input into sequences and passes the result to 
the DQN model
'''
class DQNPackager:

	def __init__(self, dqn=None):
		# dqn model
		self.__dqn = dqn

		# variables for tracking images received
		self.__most_recent_act = -1
		self.__lock = threading.Lock()
		self.reset()

		# subscribers
		QUEUE_SIZE = 1
		self.sub_act = rospy.Subscriber(topic_names[0], 
			Int8, self.actCallback,  queue_size = QUEUE_SIZE)
		self.sub_img = rospy.Subscriber(topic_names[1], 
			Image, self.imgCallback,  queue_size = QUEUE_SIZE)
		self.sub_pnt = rospy.Subscriber(topic_names[2], 
			PointCloud2, self.pntCallback,  queue_size = QUEUE_SIZE)
		self.sub_aud = rospy.Subscriber(topic_names[3], 
			AudioBuffer, self.audCallback,  queue_size = QUEUE_SIZE)

		# services
		print("Waiting for "+service_names[0][0])
		rospy.wait_for_service(service_names[0][0])
		print("Found "+service_names[0][0])
		self.srv_pcl = rospy.ServiceProxy(service_names[0][0], service_names[0][1])
		self.p = 0
		plt.gca().set_position([0, 0, 1, 1])
		#self.__fig = plt.figure()
		self.__fig = plt.figure(frameon=False)
		self.__face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		self.__face_cascade_profile = cv2.CascadeClassifier('haarcascade_profileface.xml')

	def getRecentAct(self):
		return self.__most_recent_act

	def getImgStack(self):
		return self.__imgStack

	def getPntStack(self):
		return self.__pntStack

	def getAudStack(self):
		return self.__audStack

	############################
	# Collect Data into Frames #
	############################

	def setPrint(self,p):
		self.p = p

	def clearMsgs(self):
		self.__recent_msgs = [False]*3


	def reset(self, already_locked=False):
		if(not already_locked):
			self.__lock.acquire()
		self.clearMsgs()
		self.__imgStack = 0
		self.__pntStack = 0
		self.__audStack = 0
		if(not already_locked):	
			self.__lock.release()
		print("reset complete")

	def actCallback(self, msg):
		self.__most_recent_act = msg.data
		return

	def imgCallback(self, msg):
		self.__recent_msgs[0] = msg
		self.checkMsgs()
		return

	def pntCallback(self, msg):
		self.__recent_msgs[1] = msg
		self.checkMsgs()
		return

	def audCallback(self, msg):
		self.__recent_msgs[2] = msg
		self.checkMsgs()
		return

	def checkMsgs(self):
		#may need to use mutexes on self.__recent_msgs
		self.__lock.acquire()
		if False in self.__recent_msgs:
			self.__lock.release()
			return
		if(self.p):
			print("FRAME ADDED!")
		#organize and send data
		img = self.formatImg(self.__recent_msgs[0])
		pnt = self.__recent_msgs[1] # process all point data together prior to sending
		aud = self.formatAud(self.__recent_msgs[2]) # process all audio data together prior to sending

		if(type(self.__imgStack) == int):
			self.__imgStack = img
			self.__pntStack = [pnt]
			self.__audStack = [aud]
		else:
			#print("type: ", type(img), type(self.__imgStack))
			#print("type: ", img.shape, self.__imgStack.shape)
			self.__imgStack = np.vstack((self.__imgStack, img))
			self.__pntStack.append(pnt)
			self.__audStack.append(aud)

		#print("img: ", self.__imgStack)
		#print("pnt: ", len(self.__pntStack))
		#print("aud: ", len(self.__audStack))

		self.clearMsgs()
		self.__lock.release()

	###############
	# Format Data #
	###############

	def formatImg(self, img_msg):
		# converts msg to image, reduces size and adds a border
		# to make h and w equal
		img = CvBridge().imgmsg_to_cv2(img_msg, "bgr8")
		
		#foveat image
		buffer_edge = 30
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.__face_cascade.detectMultiScale(gray, 1.3, 5)
		x,y,w,h = -1,-1,-1,-1
		#print("num_faces: ", len(faces))
		if(len(faces) > 0):

			for (xf,yf,wf,hf) in faces:
				if wf*hf > w*h:
					x,y,w,h = xf,yf,wf,hf

			x,y,w,h = x - buffer_edge,y - buffer_edge,w+ 2*buffer_edge,h+2*buffer_edge
			#cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

			if(x >= 0 and y >= 0 and x+w < 320 and y+h < 240):	
				img = img[y:y+h, x:x+w]

		#resize image
		#print(x,y,w,h)
		y_mod = 1/(img.shape[0]/float(image_data["cmp_h"]))
		x_mod = 1/(img.shape[1]/float(image_data["cmp_w"]))
		img = cv2.resize(img,None,fx=x_mod, fy=y_mod, interpolation = cv2.INTER_CUBIC)
		
		return np.asarray(img).flatten()

	def formatPntBatch(self, pnt_msg_array):
		# converts batch of msgs to array of points, reduces size 
		# and adds a border to make h and w equal
		point_array = None
		num_frames = len(pnt_msg_array)

		try:
			point_array = self.srv_pcl(num_seq=num_frames, msg_array=pnt_msg_array)
		except rospy.ServiceException, e:
			print "Service call failed: %s"%e

		points = np.reshape(point_array.points, (num_frames, 
			point_data["img_h"], point_data["img_w"])).astype(np.uint8)
		newshape = []

		y_mod = 1/(point_data["img_h"]/float(image_data["cmp_h"]))
		x_mod = 1/(point_data["img_w"]/float(image_data["cmp_w"]))
		for i in range(len(points)):
			img = points[i]
			img = cv2.resize(img,None,fx=x_mod, fy=y_mod, interpolation = cv2.INTER_CUBIC)
			'''
			img = cv2.pyrDown(img)  # decrease resolution
			img = cv2.pyrDown(img)  # decrease resolution
			div = (img.shape[1]-img.shape[0])/2
			img=cv2.copyMakeBorder(img,div,div,0,0,cv2.BORDER_CONSTANT,value=[0, 0, 0])
			#img= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
			'''
			newshape.append(np.asarray(img).flatten())
		return np.array(newshape)#.flatten()

	'''
	def formatAud(self, aud_msg):
		# converts audio to a spectrogram, reduces size 
		# and stretches image to make h and w equal
		data = aud_msg.data
		rate= 16000
		nfft = 256  # Length of the windowing segments
		fs = 256    # Sampling frequency
		data = np.reshape(data, (-1,4))
		data = data.transpose([1,0])

		frame = None
		
		pxx, freqs, bins, im = plt.specgram(data[0], nfft,fs)
		plt.axis('off')
		
		self.__fig.canvas.draw()
		data = np.fromstring(self.__fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		img = data.reshape(self.__fig.canvas.get_width_height()[::-1] + (3,))
		img = img[:,:,[2,1,0]]
		
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
		contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours[0]
		x,y,w,h = cv2.boundingRect(cnt)
		img = img[y:y+h,x:x+w]

		y_mod = 1/(h/float(audio_data["cmp_h"]))
		x_mod = 1/(w/float(audio_data["cmp_w"]))	
			
		img = cv2.resize(img,None,fx=x_mod, fy=y_mod, interpolation = cv2.INTER_CUBIC)
		frame = img

		return np.asarray(frame).flatten()
	'''

	def formatAud(self, aud_msg):
		# converts audio to a spectrogram, reduces size 
		# and stretches image to make h and w equal
		data = aud_msg.data
		
		data = np.reshape(data, (-1,4))
		data = data.transpose([1,0])

		return data[0]
	

	def formatAudBatch(self, aud_msg_array):
		# converts audio to a spectrogram, reduces size 
		# and stretches image to make h and w equal

		num_frames = len(aud_msg_array)
		#print("num_frames: ", num_frames, np.array(aud_msg_array).shape)
		#print("new shape: ", (num_frames*len(aud_msg_array[0])))
		input_data = np.reshape(aud_msg_array, (num_frames*len(aud_msg_array[0])))

		rate= 16000
		nfft = 256  # Length of the windowing segments
		fs = 256    # Sampling frequency

		b, a = signal.butter(3, 0.05)
		filtered_input = signal.lfilter(b,a,np.array(input_data))

		pxx, freqs, bins, im = plt.specgram(filtered_input, nfft,fs)

		#get RGB data of specgram
		plt.axis('off')
		self.__fig.canvas.draw()
		str_data = np.fromstring(self.__fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
		img = str_data.reshape(self.__fig.canvas.get_width_height()[::-1] + (3,))
		img = img[:,:,[2,1,0]]
		'''
		cv2.imshow('img_pre',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		'''
		#remove border on RGB data
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
		contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnt = contours[0]
		x,y,w,h = cv2.boundingRect(cnt)
		img = img[y:y+h,x:x+w]
		
		y_mod = 1/(h/float(audio_data["cmp_h"]))
		x_mod = 1/(w/float(num_frames*audio_data["cmp_w"]))	
			
		img = cv2.resize(img,None,fx=x_mod, fy=y_mod, interpolation = cv2.INTER_CUBIC)
		
		splits = [s for s in range(audio_data["cmp_w"], num_frames*audio_data["cmp_w"], audio_data["cmp_w"])]
		frames = np.hsplit(img, splits)
		return np.reshape(frames, (num_frames, -1))

	#############
	# Send Data #
	#############

	def formatOutput(self):

		

		self.__imgStack = np.expand_dims(self.__imgStack, axis=0)#.flatten().tolist()
		self.__pntStack = np.expand_dims(self.formatPntBatch(self.__pntStack), axis=0)#.flatten().tolist()
		self.__audStack = np.expand_dims(self.formatAudBatch(self.__audStack), axis=0)#.flatten().tolist()

		#self.__imgStack = np.reshape(self.__imgStack, (1, seq_len, ))
		
		print("img: ", self.__imgStack.shape)#len(self.__imgStack))
		print("pnt: ", self.__pntStack.shape)
		print("aud: ", self.__audStack.shape)
		
		
		print("Formated output! ", rospy.Time.now())

	def getNextAction(self, num_prompt):
		# finishes processing the data and passes the entire sequence to
		# the model in order to predict the next action
		if(self.__dqn == None):
			print("model not provided!")
			return -1
		self.__lock.acquire()
		num_frames = len(self.__imgStack)

		print("num_frames: ", num_frames)
		self.formatOutput()

		print("Prediction has "+str(num_frames)+" frames")

		nextact = self.__dqn.genPrediction(num_frames, self.__imgStack, self.__pntStack, self.__audStack, num_prompt)
		self.reset(already_locked=True)
		self.__lock.release()
		
		print("nextact: ",nextact)
		return nextact

if __name__ == '__main__':
	packager = DQNPackager()
