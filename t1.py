# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 20:33:03 2018

@author: User
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from zipfile import ZipFile
import os
import random
#from data_augment import augment_data

image_size = 128
num_channels = 3
num_classes = 120
num_input = image_size*image_size
learning_rate = 0.00001

batch_size = 64
num_iter = 150
num_epoch = 100
dropout = 1

dir_name = 'C:/Users/User/Downloads/Capstone Project/'
filenames1 = ZipFile('C:/Users/User/Downloads/Capstone Project/train.zip','r') 
train = filenames1.namelist()
train = [os.path.join(dir_name,i) for i in train]
train = train[1:]



labels = pd.read_csv("C:/Users/User/Downloads/Capstone Project/labels.csv")

label = pd.get_dummies(list(labels['breed']))

def generate_batch(train,label):
    filename = train[step*batch_size :(step+1)*batch_size]
    batch_y = label[step*batch_size :(step+1)*batch_size]
    batch_x = [cv2.imread(i)for i in filename]
    batch_x = [cv2.resize(i,(image_size,image_size))for i in batch_x]
    batch_x = np.array(batch_x)
    batch_x = batch_x/255
    batch_x = np.reshape(batch_x,(batch_size,image_size,image_size,3))
    
    return batch_x, batch_y



# tf Graph input
X = tf.placeholder(tf.float32, [None, image_size,image_size,num_channels])
Y = tf.placeholder(tf.float32, [None, num_classes])
# dropout (keep probability)
keep_prob = tf.placeholder(tf.float32)

#def conv2d(x, W, b, strides=1):
#    # Conv2D wrapper, with bias and relu activation
#    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
#    x = tf.nn.bias_add(x, b)
#    return tf.nn.relu(x)

def conv2d(x, shape, num):
   w = tf.get_variable('w' + str(num), shape = shape, initializer = tf.contrib.layers.xavier_initializer(seed=0))
   b = tf.zeros(shape[-1], name='b' + str(num))
   conv = tf.nn.conv2d(x, w, [1,1,1,1], padding="SAME")
   conv = tf.nn.bias_add(conv, b)
   return(tf.nn.relu(conv))

conv1 = conv2d(X, [3,3,3,16], 1)
conv2 = conv2d(conv1, [3,3,16,16], 2)
maxp1 = tf.nn.max_pool(conv2, [1,2,2,1], [1,2,2,1], padding='SAME')

bn1   = tf.layers.batch_normalization(maxp1)

conv3 = conv2d(bn1, [5,5,16,32], 3)
conv4 = conv2d(conv3, [5,5,32,32], 4)
maxp2 = tf.nn.max_pool(conv4, [1,2,2,1], [1,2,2,1], padding='SAME')

fc    = tf.contrib.layers.flatten(maxp2)
kp    = tf.placeholder(tf.float32)
fc    = tf.nn.dropout(fc, keep_prob)

fc1   = tf.contrib.layers.fully_connected(fc, 1024)

bn2   = tf.layers.batch_normalization(fc1)

fc2   = tf.contrib.layers.fully_connected(bn2, 1024)
fc2   = tf.nn.dropout(fc2, keep_prob)

fc3   = tf.contrib.layers.fully_connected(fc2, 120, activation_fn=None)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fc3, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss_op)

prediction = tf.nn.softmax(fc3)
# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

filename = []

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    
    
    for epoch in range(num_epoch):
        
        label1 = label.values.tolist()
        c = list(zip(train, label1))
        
        random.shuffle(c)

        train, label1 = zip(*c)
        train = list(train)
        label1 = np.array(label1)
        label2 = pd.DataFrame(np.array(label1))
        label2.columns = list(label.columns.values)
        
        for step in range(num_iter):
            
            batch_x,batch_y = generate_batch(train, label2)
            
            # Run optimization op (backprop)
            _,loss,acc = sess.run([optimizer,loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y, keep_prob: dropout})
                                                                     
            
            if(step%10 == 0):
                print("Epoch " + str(epoch) + " Iteration " + str(step) + " Batch Loss= " + \
                      str(loss) + " Training Accuracy= " + \
                      str(acc))
    
            

    print("Optimization Finished!")