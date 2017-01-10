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

training_path = '/Users/johnson/Desktop/train.csv'
# training_path = '/Users/johnson/Desktop/small.csv'
test_path     = '/Users/johnson/Desktop/test.csv'
submisson_path = '/Users/johnson/Desktop/sample_submission.csv'


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


# display image
def display(img):
    print type(img)
    # (784) => (28,28)
    width = height = int(np.sqrt(len(img)))
    one_image = img.reshape(width,height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)
    plt.show()
# output image
from numpy import array 
# display(array([1,1,1,1,0,1,1,1,1]))
img = array([0,2,1,5])
display(img)
display(np.multiply(img, 1.0/255.0))

# display(array([255,0,0,255]))
# display(array([255,0,0,1]))
