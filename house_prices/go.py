# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>
# tutorial： https://www.kaggle.com/neviadomski/house-prices-advanced-regression-techniques/how-to-get-to-top-25-with-simple-model-sklearn

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle       # 这几行import都好妖路。。。跟以前看到的不太一样



# import warnings
# warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test   = pd.read_csv('test.csv')
# print train.shape
# print test.shape
# test.columns.values[-1] = '123'
# print (train.columns).difference(test.columns)    # '-'   also worked, but would show  a FutureWarning
# 另外，不会显示index的差异来源，即，如果train多一个SalePrice，test多一个123，返回结果是['SalePrice','123'],共存不区分

# print  train.head
NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
# NAs =  pd.DataFrame(train.isnull().sum())

# print NAs.sum(axis=1)  # axis : {index (0), columns (1)}  index代表一行一行相加，矩阵的r(index)在减少，columns代表一列一列的相加，矩阵的c(columns)在减少
# print (train['SalePrice'].count())
# print type(NAs.sum(axis=1) > 0)
count = 0
x = NAs[NAs.sum(axis=1) > 0]  # 这种语法只要记得 NAs.sum(axis=1) > 0 返回的是一个boolean series就可以了
print x.sum()
# print train['LotFrontage'].isnull()

# s =  train.isnull().sum(axis=0)  # 其实和上面是一样的功能，只是自己尝试一下别的表达方法
# print s.iloc[s.nonzero()[0]].count() # train 19个,test 33个 可以发现在这个问题里有大量的可能为空的feature   另外空数据各自有7000个

# print train.isnull()