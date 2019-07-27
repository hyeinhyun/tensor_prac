#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:19:46 2019

@author: hihyun
"""

import tensorflow as tf
##multi-variable regression
x_data2 = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data2 = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
x=tf.placeholder("float",[None,3])
y=tf.placeholder("float",[None,1])
W=tf.Variable(tf.random_normal([3,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='weight')
hypothesis=tf.matmul(x,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val,hy_val,_=sess.run([cost,hypothesis,train],
                               feed_dict={x:x_data,y:y_data})

    if step%100==0:
        print(step,"cost:", cost_val)
        print("prediction : ",hy_val)




##soft max _ logistic classification

X=tf.placeholder("float",[None,4])
Y=tf.placeholder("float",[None,3])
nb_classes=3
W=tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b=tf.Variable(tf.random_normal([nb_classes]),name='weight')

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer,feed_dict={X:x_data2,Y:y_data2})
        

    a=sess.run(hypothesis,feed_dict={X:[[1,11,7,9]]})
    print(a,sess.run(tf.arg_max(a,1)))

    





"""
x=tf.placeholder("float",[None,3])
y=tf.placeholder("float",[None,3])
W=tf.Variable(tf.random_normal([3,3]))
b=tf.Variable(tf.random_normal([3]))
"""