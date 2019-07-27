#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:29:09 2019

@author: hihyun
"""

import tensorflow as tf
import matplotlib.pyplot as plt

###Linear Regression
#tf.set_random_seed(777)
x_train=tf.placeholder(tf.float32,shape=[None])
y_train=tf.placeholder(tf.float32,shape=[None])

W=tf.Variable(tf.random_normal([1]), name='weight')

#b=tf.Variable(tf.random_normal([1]), name='bias')

#hypothesis
hypothesis=x_train*W

#cost
cost=tf.reduce_mean(tf.square(hypothesis-y_train))

#optimizer
#optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
#result=optimizer.minimize(cost)

#manually
learning_rate=0.1
gradient=tf.reduce_mean((W*x_train-y_train)*x_train)
descent=W-learning_rate*gradient
update=W.assign(descent)

W_re=[]
co_re=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30):
        #feed_rate=i*0.1
        update_,cost_=sess.run([update,cost],feed_dict={x_train:[1,2,3],y_train:[1,2,3]})
            #print(str(resul_)+"+"+str(W_)+"+"+str(b_))
        print(cost_)

"""
W_re.append(feed_rate)
co_re.append(cost_)

plt.plot(W_re,co_re)
plt.show()
"""
           