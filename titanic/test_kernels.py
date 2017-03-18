# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>
import time
start = time.time()
import pandas as pd
from pandas import Series,DataFrame

import numpy as np
import matplotlib.pyplot as plt
# print time.time() - start

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression         # 逻辑回归
from sklearn.svm import SVC, LinearSVC                      # svm：支持向量机
from sklearn.ensemble import RandomForestClassifier         # 随机森林分类
from sklearn.neighbors import KNeighborsClassifier          # KN(?) 分类
from sklearn.naive_bayes import GaussianNB                  # 朴素贝叶斯 import 高斯分布
# print time.time() - start         土制测速方法，在大程序里狂复制这条快速发现时间开销，大量时间花费在sns加载（？）上


titanic_df = pd.read_csv("train.csv")
test_df    = pd.read_csv("test.csv")

# print titanic_df.head()
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis=1)       # 其实不是很必要，drop了舒服一点（or节约内存？）
test_df    = test_df.drop(['Name','Ticket'], axis=1)

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

# fig1, (axis1, axis2, axis3) = plt.subplots(1,3,figsize=(15,5))



# sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
# sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2) # hue 代表相同x情况下以此再进行柱状图的分隔
# sns.countplot(x='Survived', hue="Sex", data=titanic_df, order=[1,0], ax=axis2) # hue 代表相同x情况下以此再进行柱状图的分隔

# 我靠，上面这个好方便！！！昨天自己写的那个统计的是多余的啊，这个直接可视化了！！


# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()  # 分组取平均，我佛，可以注意一下数据取mean前typye  pandas.core.groupby.DataFrameGroupBy 

# sns.barplot(x='Embarked', y='Survived', data=titanic_df, order=['S','C','Q'],ax=axis3)         # 


embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked']) # 根据help：Convert categorical variable into dummy/indicator variables  一个重要的概念 
# 另外，这个表达变换的过程有点类似于 minist里将一个数字展开成向量的过程，或者说spam问题里词典中的词是否在邮件中出现的表示
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)


embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)

# print embark_dummies_titanic
titanic_df = titanic_df.join(embark_dummies_titanic)  # 注意先后embark col的变化
test_df    = test_df.join(embark_dummies_test)


titanic_df.drop(['Embarked'], axis=1,inplace=True)     # attention!  print help(titanic_df.drop) 后发现 inplace = 1和 0 的区别类似于 sorted 和sort的区别
test_df.drop(['Embarked'], axis=1,inplace=True)
# print 'ggg'
# print test_df


# MD，用了一大堆代码清理数据和做展示，下面还有好多，跳到最后面真正进行计算的代码少的一批
# print titanic_df









# X_train = titanic_df.drop("Survived",axis=1,inplace=True)
# # Y_train = titanic_df["Survived"]
# X_test  = test_df.drop("PassengerId",axis=1).copy()     # 这附近对test train数据处理的手法要注意，以前老是两张csv处理起来一头包

# logreg = LogisticRegression()               # 很基础的逻辑回归，明天再康

# # if not use "%matplotlib inline", should use sns.plt.show() instead!!!!



test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare']    = test_df['Fare'].astype(int)

fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0] 
# actually titanic_df[titanic_df["Survived"] == 0]["Fare"] 也可以
fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]


avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()]) # 注意顺序，直接把not放前面，就是第0个，此后index.name再改成survived则可表示是否生还
std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])   # 标准差

# print DataFrame([fare_not_survived.std()])

# titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50)) 
# hist 直方图，bins:Number of histogram bins to be used  xlim字面意思



avgerage_fare.index.names = std_fare.index.names = ["Survived"]
# avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)

#Ageeeeeeeee
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()
print  count_nan_age_titanic

average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()


# generate random numbers between (mean - std) & (mean + std)
# 随机生成年龄填补空白。。。注意取的上下限，是用std和aver来取的，而不是max和min
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)# 也可以用plot，参数里设置kind=hist



sns.plt.show()
