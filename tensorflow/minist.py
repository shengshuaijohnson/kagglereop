#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
x = tf.placeholder("float", [None, 784])
# print x

W = tf.Variable(tf.zeros([784,10]))  # 权重
# b = tf.Variable(tf.zeros([10]))      # 偏移量 

y = tf.nn.softmax(tf.matmul(x,W) )  # matmul = math multiply
y_ = tf.placeholder('float', [None,10]) # 是实际的分布，每个数字N对应一个矢量（矢量的第N个元素为1其他为0）
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # 交叉熵的log去掉，感觉也没什么影响，至少几乎没负影响 

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()



sess = tf.Session()
sess.run(init)

for i in range(1000):
    print i
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

batch_xs, batch_ys = mnist.train.next_batch(10000)

print sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})

