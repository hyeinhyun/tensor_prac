#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:43:59 2019

@author: hihyun
"""
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
learning_rate = 0.001
training_epochs = 15
batch_size = 100
epoch=15

tf.reset_default_graph()

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x_data=tf.placeholder(tf.float32,[None,784])
y_data=tf.placeholder(tf.float32,[None,10])

W1=tf.get_variable("W1",shape=[784,512],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([512]))
L1=tf.nn.relu(tf.matmul(x_data, W1) + b1)

W2=tf.get_variable("W2",shape=[512,256],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b2=tf.Variable(tf.random_normal([256]))
L2=tf.nn.relu(tf.matmul(L1, W2) + b2)

W3=tf.get_variable("W3",shape=[256,256],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b3=tf.Variable(tf.random_normal([256]))
L3=tf.nn.relu(tf.matmul(L2, W3) + b3)

W4=tf.get_variable("W4",shape=[256,256],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b4=tf.Variable(tf.random_normal([256]))
L4=tf.nn.relu(tf.matmul(L3, W4) + b4)

W5=tf.get_variable("W5",shape=[256,10],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
b5=tf.Variable(tf.random_normal([10]))
logit=tf.matmul(L4, W5) + b5
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=y_data))


optimizer=tf.train.AdamOptimizer(0.001).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epochs in range(15):
        avg_cost=0
        total_batch=int(mnist.train.num_examples / batch_size)
    
        for steps in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c,_=sess.run([cost,optimizer],feed_dict={x_data:batch_xs,y_data:batch_ys})
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print("learning finish!")
    
    
    
    correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={
          x_data: mnist.test.images, y_data: mnist.test.labels}))
    
    