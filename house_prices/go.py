# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle       # 这几行import都好妖路。。。跟以前看到的不太一样



import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test   = pd.read_csv('test.csv')
print train.shape
print test.shape
# test.columns.values[-1] = '123'
print (train.columns).difference(test.columns)    # '-'   also worked, but would show  a FutureWarning
# 另外，不会显示index的差异来源，即，如果train多一个SalePrice，test多一个123，返回结果是['SalePrice','123'],共存不区分

