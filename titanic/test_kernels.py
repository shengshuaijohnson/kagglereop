# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression         # 逻辑回归
# from sklearn.svm import SVC, LinearSVC                      # svm：支持向量机
from sklearn.ensemble import RandomForestClassifier         # 随机森林分类
from sklearn.neighbors import KNeighborsClassifier          # KN(?) 分类
from sklearn.naive_bayes import GaussianNB                  # 朴素贝叶斯 import 高斯分布

titanic_df = pd.read_csv("train.csv")
test_df    = pd.read_csv("test.csv")

# print titanic_df.head()
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# plot
sns.factorplot('Pclass','Survived', data=titanic_df,size=4,aspect=3)
sns.plt.show()  # 我靠，上面这个好方便！！！昨天自己写的那个统计的是多余的啊，这个直接可视化了！！
# if not use "%matplotlib inline", should use sns.plt.show() !!!!