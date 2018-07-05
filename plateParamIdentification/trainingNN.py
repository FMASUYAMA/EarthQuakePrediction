# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:32:45 2018

@author: yu
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import pdb
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import matplotlib.pylab as plt
import pickle
import glob
import EarthQuakePlateModelKDE_FFT as eqp

#---------------------
# Define some handy network layers
def weight_variable(name,shape):
	return tf.get_variable(name,shape,initializer=tf.random_normal_initializer(stddev=0.1))

def bias_variable(name,shape):
	return tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))

def softmax(inputs,w,b,keepProb):
	softmax = tf.matmul(inputs,w) + b
	softmax = tf.nn.dropout(softmax, keepProb)
	softmax = tf.nn.softmax(softmax)
	return softmax

def fc_relu(inputs, w, b, keepProb):
	relu = tf.matmul(inputs, w) + b
	relu = tf.nn.dropout(relu, keepProb)
	relu = tf.nn.relu(relu)
	return relu

#---------------------

#---------------------

def nn(x,reuse=False):
	with tf.variable_scope('nn') as scope:  
		keepProb = 0.5 
		if reuse:
			keepProb = 1.0			
			scope.reuse_variables()

		#input -> hidden
		w1 = weight_variable('w1',[8*10,20])
		b1 = bias_variable('b1',[20])
		h = fc_relu(x,w1,b1,keepProb) 

		#hidden -> output
		w2 = weight_variable('w2',[20,1])
		b2 = bias_variable('b2',[1])
		y = fc_relu(h,w2,b2,keepProb)

		return y

# load data
nCell = 8
nWindow = 10
sYear = 2000
eYear = 10000
bInd = 0
size = 25

#myData = eqp.Data(fname="yVlog_10*",trainRatio=0.8,nCell=nCell,sYear=sYear, eYear=eYear, bInd=bInd, isTensorflow=False)
#kde = myData.KDE(nYear=8000)
#myData.FFT(slice_size=size,eFrq=250,nYear=8000,nCell=nCell,trainRatio=0.8,isTensorflow=True)

myData = eqp.Data(fname="kde_fft_log_10*", trainRatio=0.8, bInd=bInd, isWindows=False)

# Build the computation graph for training
x = tf.placeholder(tf.float32, shape=[None,nCell*nWindow])
y_ = tf.placeholder(tf.float32, shape=[None])

predict_op = nn(x)
predict_test_op = nn(x,reuse=True)
#---------------------

#---------------------
# loss
loss= tf.reduce_mean(tf.abs(predict_op - y_))
loss_test= tf.reduce_mean(tf.abs(predict_test_op - y_))

trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)
#---------------------

#---------------------
batchSize = 50
sess = tf.Session()
sess.run(tf.global_variables_initializer())

testX = np.reshape(myData.xTest,[-1,nCell*nWindow])
testY = myData.yTest 

# Start training
for i in range(70000):
	
	batchX, batchY = myData.nextBatch(batchSize)
	batchX = np.reshape(batchX,[-1,nCell*nWindow])
	#pdb.set_trace()
	
	_, lossTrain, predTrain = sess.run([trainer, loss, predict_op], feed_dict={x:batchX, y_:batchY})
	
	
	if i % 100 == 0:
		print("iteration: %d,loss: %f" % (i,lossTrain))
	if i % 500 == 0:
		lossTest, predTest  = sess.run([loss_test, predict_test_op], feed_dict={x:testX, y_:testY})
		
		with open("training/process_{}.pickle".format(i), "wb") as fp:
			pickle.dump(predTest,fp)
			pickle.dump(myData.xTest,fp)
			pickle.dump(myData.yTest,fp)
			pickle.dump(lossTest,fp)
			#pickle.dump(myData.minY,fp)
			#pickle.dump(myData.maxY,fp)

		# save model to file
		saver = tf.train.Saver()
		saver.save(sess,"models/model_{}.ckpt".format(i))
#---------------------

