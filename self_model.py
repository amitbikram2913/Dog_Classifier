# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:47:01 2018

@author: User
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from zipfile import ZipFile
import os
import random
from data_augment import augment_data

image_size = 128
num_channels = 3
num_classes = 120
num_input = image_size*image_size
learning_rate = 0.0001

batch_size = 64
num_iter = 150
num_epoch = 20
dropout = 0.2

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
    batch_x = np.reshape(batch_x,(batch_size,image_size*image_size,3))
    
    return batch_x, batch_y



# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input,num_channels])
Y = tf.placeholder(tf.float32, [None, num_classes])
# dropout (keep probability)
keep_prob = tf.placeholder(tf.float32) 

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 3, 32],stddev=0.001)),
    # 3x3 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 32, 64],stddev=0.001)),
    # 3x3 conv, 64 inputs, 128 outputs
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 64, 128],stddev=0.001)),
    # 3x3 conv, 128 inputs, 128 outputs
#    'wc4': tf.Variable(tf.truncated_normal([3, 3, 128, 128],stddev=0.001)),
#    # 3x3 conv, 128 inputs, 128 outputs
#    'wc5': tf.Variable(tf.truncated_normal([3, 3, 128, 128],stddev=0.001)),
#    # 3x3 conv, 128 inputs, 256 outputs
#    'wc6': tf.Variable(tf.truncated_normal([3, 3, 128, 256],stddev=0.001)),
#    # 3x3 conv, 256 inputs, 512 outputs
#    'wc7': tf.Variable(tf.truncated_normal([3, 3, 256, 512],stddev=0.001)),
    
    # fully connected, 4*4*256 inputs, 2048 outputs
    'wd1': tf.Variable(tf.truncated_normal([16*16*128, 1024],stddev=0.001)),
    # fully connected, 2048 inputs, 1024 outputs
    'wd2': tf.Variable(tf.truncated_normal([1024, 512],stddev=0.001)),
    # 1024 inputs, 120 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([512, num_classes],stddev=0.001))
}



biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bc3': tf.Variable(tf.zeros([128])),
#    'bc4': tf.Variable(tf.zeros([128])),
#    'bc5': tf.Variable(tf.zeros([128])),
#    'bc6': tf.Variable(tf.zeros([256])),
#    'bc7': tf.Variable(tf.zeros([512])),
    
    'bd1': tf.Variable(tf.zeros([1024])),
    'bd2': tf.Variable(tf.zeros([512])),
    'out': tf.Variable(tf.zeros([num_classes]))
}

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights,biases,dropout):   
    
    x = tf.reshape(x, shape=[-1, image_size, image_size, 3])
    
    conv_1 = conv2d(x,weights['wc1'],biases['bc1'])
    max_1 = maxpool2d(conv_1)
    conv_2 = conv2d(max_1,weights['wc2'],biases['bc2'])
    max_2 = maxpool2d(conv_2)
    
    
#    conv_3 = conv2d(max_2,weights['wc3'],biases['bc3'])
#    max_3 = maxpool2d(conv_3)
#    conv_4 = conv2d(max_3,weights['wc4'],biases['bc4'])
#    max_4 = maxpool2d(conv_4)
    
    conv_3 = conv2d(max_2,weights['wc3'],biases['bc3'])
#    conv_6 = conv2d(conv_5,weights['wc6'],biases['bc6'])
#    conv_7 = conv2d(conv_6,weights['wc7'],biases['bc7'])
    max_3 = maxpool2d(conv_3)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.contrib.layers.flatten(max_3)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.layers.batch_normalisation(fc1)
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer and the model
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


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
            _,loss,acc,out = sess.run([train_op,loss_op, accuracy,logits], feed_dict={X: batch_x,
                                                                     Y: batch_y, keep_prob: dropout})
                                                                     
            
            if(step%10 == 0):
                print("Epoch " + str(epoch) + " Iteration " + str(step) + " Batch Loss= " + \
                      str(loss) + " Training Accuracy= " + \
                      str(acc))
    
            

    print("Optimization Finished!")