# random#!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

# import tensorflow as tf
import numpy as np
x_data = np.float32(np.random.rand(2, 100)) # 随机输入  二维x点
y_data = np.dot([0.100, 0.200], x_data) + 9.5
# print np.dot([1, 2], x_data)
# # print (x_data)
print help(np.dot)
# # 构造一个线性模型
# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b


# # 最小化方差
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

# # 初始化变量
# init = tf.global_variables_initializer()

# # 启动图 (graph)
# sess = tf.Session()
# sess.run(init)

# # 拟合平面
# for step in xrange(0, 201):
#     break
#     sess.run(train)
#     if step % 20 == 0:
#         print step, sess.run(W), sess.run(b)

# # 你和结果正好是y_data中输入的[0.1, 0.2]和9.5结合



# import requests


# url = 'http://www.wwhhll.com/forum.php?mod=viewthread&tid=3972&extra=page%3D2%26filter%3Dsortid%26orderby%3Dlastpost%26sortid%3D13'

# print requests.get(url).text