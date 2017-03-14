#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

import numpy as np
import pandas as pd
import random

import tensorflow as tf

# import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

images = 0
labels = 0
def spread(x):
    res = np.ndarray(10)
    res[num] = 1
    return res

def get_train_data(filename):
    global images, labels
    data = pd.read_csv(filename)
    images = data.iloc[:,1:].values
    raw_labels = data.iloc[:,:1].values

    images = images.astype(np.float)
    raw_labels = raw_labels.astype(np.int)  #(42000, 785)
    labels = np.ndarray((raw_labels.shape[0], 10))
    for i in xrange(raw_labels.shape[0]):
        num = raw_labels[i]
        labels[i][num] = 1      # 每副图的形状以向量形式储存，代表数字N向量的第N个元素为1，其它为0
        




# print images.shape


def training(step, train_times):
    x = tf.placeholder("float", [None, 784])
    weight = tf.Variable(tf.zeros([784, 10]))
    y = tf.nn.softmax(tf.matmul(x, weight))
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y)) # 对小于1的数取对数会变负,所以前面要负号？
    train_step = tf.train.GradientDescentOptimizer(step).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    global images, labels
    batch_xs = images
    batch_ys = labels 
    size = 180 # 接近200后（170、180左右）准确率就会开始急剧缩小
    length = labels.shape[0]
    for i in range(train_times):
        # start = int((images.shape[0] - size) * random.random())
        print i
        batch_xs = images[i*size%length:(i+1)*size%length]/256  # 这里TMD没转化！！之前不除就会出事！
        batch_ys = labels[i*size%length:(i+1)*size%length]
        # print i*size%length , (i+1)*size%length
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    batch_xs = images
    batch_ys = labels
    print sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
    file_writer = tf.summary.FileWriter('/Users/johnson/kagglereop', sess.graph)

    return
    # out put the data in below part
    data = pd.read_csv('../test.csv')
    images = data.iloc[:,:].values
    batch_xs = images
    raw_res = sess.run(y, feed_dict={x: batch_xs})
    res = np.ndarray((raw_res.shape[0],1))
    for index in range(raw_res.shape[0]):
        res[index] = np.argmax(raw_res[index])

    np.savetxt('output.csv', res, fmt='%d', delimiter=',', header="Label")
    print 'writting'

def main():
    get_train_data('../train.csv')
    training(step=0.01, train_times=100)

if __name__ == '__main__':
    main()

