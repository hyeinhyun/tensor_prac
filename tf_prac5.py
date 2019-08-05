#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:45:14 2019

@author: hihyun
"""

import numpy as np
import tensorflow as tf

X=tf.placeholder(tf.float32,[None,784])
Y=tf.placeholder(tf.float32,[None,10])

W1=tf.Variable(tf.random_normal([784,256],dtype=tf.float32))
b1=tf.Variable(tf.random_normal([256],dtype=tf.float32))
layer1=tf.nn.softmax(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.random_normal([256,256],dtype=tf.float32))
b2=tf.Variable(tf.random_normal([256],dtype=tf.float32))
layer2=tf.nn.softmax(tf.matmul(layer1,W2)+b2)

W3=tf.Variable(tf.random_normal([256,10],dtype=tf.float32))
b3=tf.Variable(tf.random_normal([10],dtype=tf.float32))
hypothesis=tf.nn.softmax(tf.matmul(layer2,W3)+b3)


cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
is_correct=tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))


training_epochs=15
batch_size=100


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            c,_=sess.run([cost,optimizer],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost+=c/total_batch
        print("cost=","{:.9f}".format(avg_cost))
        
    H,A=sess.run([hypothesis,accuracy],feed_dict={X:mnist.test.images,Y:mnist.test.labels})
    print("ACC : ",A)
    print("AAA : ",accuracy.eval(session=sess,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))