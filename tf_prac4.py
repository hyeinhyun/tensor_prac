#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:00:24 2019

@author: hihyun
"""
import numpy as np
import tensorflow as tf
x_data=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=np.float32)
y_data=np.array([[1],[0],[0],[1]],dtype=np.float32)

X=tf.placeholder(tf.float32,[None,2])
Y=tf.placeholder(tf.float32,[None,1])

W1=tf.Variable(tf.random_normal([2,10],name='weight1'))
b1=tf.Variable(tf.random_normal([1,10]),name='bias1')
layer1=tf.sigmoid(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.random_normal([10,1]),name='weight2')
b2=tf.Variable(tf.random_normal([1]),name='bias2')
hypothesis=tf.sigmoid(tf.matmul(layer1,W2)+b2)

#cost
cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

train=tf.train.GradientDescentOptimizer(0.1).minimize(cost)



predict=tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        _,cost_val=sess.run(
                [train,cost],feed_dict={X:x_data, Y:y_data})
        if step%1000==0:
            print(step, cost_val)
            
    acc,pre=sess.run([accuracy,predict],feed_dict={X:x_data,Y:y_data})
    print(f"accuracy{acc}{pre}")