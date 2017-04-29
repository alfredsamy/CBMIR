################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from six.moves import cPickle as pickle
import tensorflow as tf

Dataset_PATH = '/media/alfred/B8308B68308B2D06/IRMA/'

net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
batch_size = 16
num_labels = 57

def unpickle(file):
	print("unpicle " + file)
	fo = open(file, 'rb')
	dict = pickle.load(fo)
	fo.close()
	return dict

def gen_batch(index,imgs):
	train_dataset = []
	train_labels = []
	img_names = []
	while(index < len(imgs)):
		i = imgs[index]
		img = (imread(Dataset_PATH+i,mode='RGB')).astype(float32)
		img = imresize(img,(227,227))
		img = img - mean(img)
		img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
		
		name = i.split('/')[1]
		name = name[:len(name)-4]
		
		if(dict[name] != '\\N'):
			train_dataset.append(img)
			train_labels.append(dict[name])
			img_names.append(name+'.png')
		if(len(train_dataset) == batch_size):
			break
		index += 1
	train_dataset = np.array(train_dataset)
	train_labels = np.array(train_labels)
	img_names = np.array(img_names)
	return train_dataset, train_labels, img_names
################################################################################
# AlexNet:
# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])


graph = tf.Graph()
with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, 227, 227, 3))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	global_step = tf.Variable(0)
	#initialize variable
	conv1W = tf.Variable(net_data["conv1"][0])
	conv1b = tf.Variable(net_data["conv1"][1])

	conv2W = tf.Variable(net_data["conv2"][0])
	conv2b = tf.Variable(net_data["conv2"][1])

	conv3W = tf.Variable(net_data["conv3"][0])
	conv3b = tf.Variable(net_data["conv3"][1])

	conv4W = tf.Variable(net_data["conv4"][0])
	conv4b = tf.Variable(net_data["conv4"][1])

	conv5W = tf.Variable(net_data["conv5"][0])
	conv5b = tf.Variable(net_data["conv5"][1])

	fc6W = tf.Variable(net_data["fc6"][0])
	fc6b = tf.Variable(net_data["fc6"][1])

	fc7W = tf.Variable(net_data["fc7"][0])
	fc7b = tf.Variable(net_data["fc7"][1])
	
	# fc8W = tf.Variable(net_data["fc8"][0])
	# fc8b = tf.Variable(net_data["fc8"][1])
	fc8W = tf.Variable(tf.truncated_normal([4096, num_labels],stddev=0.1))
	fc8b = tf.Variable(tf.zeros([num_labels]))

	# Model.
	def model(data):
		#conv1
		#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
		k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
		conv1_in = conv(data, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
		conv1 = tf.nn.relu(conv1_in)
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn1 = tf.nn.local_response_normalization(conv1,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


		#conv2
		#conv(5, 5, 256, 1, 1, group=2, name='conv2')
		k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv2 = tf.nn.relu(conv2_in)
		radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
		lrn2 = tf.nn.local_response_normalization(conv2,depth_radius=radius,alpha=alpha,beta=beta,bias=bias)
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

		#conv3
		#conv(3, 3, 384, 1, 1, name='conv3')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
		conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv3 = tf.nn.relu(conv3_in)

		#conv4
		#conv(3, 3, 384, 1, 1, group=2, name='conv4')
		k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
		conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv4 = tf.nn.relu(conv4_in)


		#conv5
		#conv(3, 3, 256, 1, 1, group=2, name='conv5')
		k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
		conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
		conv5 = tf.nn.relu(conv5_in)
		k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
		maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

		#fc6
		#fc(4096, name='fc6')
		fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

		#fc7
		#fc(4096, name='fc7')
		# fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

		#fc8
		#fc(1000, relu=False, name='fc8')
		# fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
		fc7 = tf.nn.xw_plus_b(fc6, fc7W, fc7b)
		return fc7
	#prob
	#softmax(name='prob'))
	fc7 = model(tf_train_dataset)
	fc7_rl = tf.nn.relu(fc7)
	logits = tf.nn.xw_plus_b(fc7_rl, fc8W, fc8b)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) 
	learning_rate = tf.train.exponential_decay(0.1, global_step, 1000, 0.95, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	prob = tf.nn.softmax(logits)



def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
		  / predictions.shape[0])


if __name__ == '__main__':
	imgs = []
	file = 'dataset.txt'
	with open(file, 'r') as fo:
		for line in fo.read().split('\n'):
			imgs.append(line)
		fo.close()

	dict = unpickle('dataset_labels')['labels']
	random.shuffle(imgs)

	num_steps = 50001
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		print('Initialized')
		
		l = 0
		for step in range(num_steps):
			offset = random.randint(0,len(imgs) - batch_size)
			batch_data,batch_label,_ = gen_batch(offset, imgs)

			# batch_labels = (np.arange(num_labels) == batch_label[:,None]).astype(np.float32)
			batch_labels = np.zeros((batch_size,num_labels))
			for i in range(batch_size):
				batch_labels[i][int(batch_label[i])-1] = 1.0
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			ll, _, predictions = session.run([loss, optimizer, prob], feed_dict=feed_dict)
			
			l += ll
			if (step % 100 == 0):
				print('Minibatch loss at step %d: %f  %f' % (step, l, ll))
				print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
				print()
				l = 0
			if (step % 1000 == 0 and step != 0):
				saver = tf.train.Saver()
				save_path = saver.save(session, "model.ckpt")
				print("Model saved in file: %s" % save_path)
		# saving weights for future reuse
		saver = tf.train.Saver()
		save_path = saver.save(session, "model.ckpt")
		print("Model saved in file: %s" % save_path)
	
	print("DONE")