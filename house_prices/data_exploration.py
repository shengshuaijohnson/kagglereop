# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

# this scrip concentrate on data visualization and learn the features but not machine learning -- qrr
# kernel url: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler        # 标准化工具，注意用法，赶脚很实用！
from scipy import stats                     # library of statistical functions

# import warnings
# warnings.filterwarnings('ignore')

df_train = pd.read_csv('train.csv')

# print df_train.columns


# In order to have some discipline in our analysis, we can create an Excel spreadsheet with the following columns:
# Variable,Type,Segment,Expectation,Conclusion,Comments

# Type： 'numerical' or 'categorical' By 'numerical' we mean variables for which the values are numbers, and by 'categorical' we mean variables for which the values are categories.
# 即数值型和类别型

# MD，这个B对数据的描述也太详细了吧，各种segment什么的。。。好多。。。。重点注意在评估出Expectation的高中低之后怎么起作用
# 因为在以往的学习过程中人工对参数的评估并不参与训练过程，这个作者是单纯评级，还是说会有更多操作？



# ...... scatter plots instead of boxplots, which are more suitable for categorical variables visualization. The way we visualize data often influences our conclusions.
# 注意散点图和箱型图的用法和区别，主要是箱型图以前比较少见 ，参考 http://www.statisticshowto.com/how-to-read-a-box-plot/
# 这里的图形与上述网站的说法好像有所不同，因为最顶端的线不是最大值，而线外面还有若干点，也是有效数据（这些点好像是经判断为异常值？？）
# 异常值的判定规则： whis 默认为1.5时，可参照https://sanwen8.cn/p/11bqgOb.html ，也就是说box是固定的，whis影响内外限范围取值。
# 之前的网站没仔细读，就留了个URL，仔细读之后好像的确如此，whis即whisker
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
print df_train[df_train['OverallQual']==7]['SalePrice']


corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# heatmap我曹！看起来屌屌的!







sns.plt.show()

