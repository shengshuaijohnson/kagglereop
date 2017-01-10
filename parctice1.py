#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

import datetime
import json
import os
import urllib
import sys
import xlrd
import pandas as pd
import csv
from rurulib.hzlib.libfile import *

import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf

# settings
LEARNING_RATE = 1e-4
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500        
    
DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10
#############
training_path = '/Users/johnson/Desktop/train.csv'
# training_path = '/Users/johnson/Desktop/small.csv'
test_path     = '/Users/johnson/Desktop/test.csv'
submisson_path = '/Users/johnson/Desktop/sample_submission.csv'


data = pd.read_csv(training_path)

# print('data({0[0]},{0[1]})'.format(data.shape))
# print (data.head())  # 显示前4、5行


images = data.iloc[:,1:].values
images = images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
# images = np.multiply(images, 1.0 / 255.0)
# not nessary dose not matter the final images' display  ????<-not sure

print('images({0[0]},{0[1]})'.format(images.shape))
print type(images.shape)



image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))


# display image
def display(img):
    print type(img)
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()
# output image 
for i in range(21,29):    
    display(images[i])
    print images[i]