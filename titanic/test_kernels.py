# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>
import time
start = time.time()
import pandas as pd
from pandas import Series,DataFrame
pd.options.mode.chained_assignment = None  # default='warn'   from http://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

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

data_survived = titanic_df.loc[titanic_df["Survived"]==0]

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
# fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
# axis1.set_title('Original Age values - Titanic')
# axis2.set_title('New Age values - Titanic')

average_age_titanic   = titanic_df["Age"].mean()
std_age_titanic       = titanic_df["Age"].std()
count_nan_age_titanic = titanic_df["Age"].isnull().sum()


average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()


# generate random numbers between (mean - std) & (mean + std)
# 随机生成年龄填补空白。。。注意取的上下限，是用std和aver来取的，而不是max和min
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)


# titanic_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)# 也可以用plot()，参数里设置kind=hist
# 上面一条和下面一条相比可知dropna没有改变df本身数据

titanic_df["Age"][np.isnan(titanic_df["Age"])] = rand_1  # 奇葩语法。。。
test_df["Age"][np.isnan(test_df["Age"])] = rand_2

titanic_df['Age'] = titanic_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)
# titanic_df['Age'].hist(bins=70, ax=axis2)


# facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=4) # 类介绍：Subplot grid for plotting conditional relationships.
# facet.map(sns.kdeplot,'Age',shade= True) # sns.kdeplot：支持单变量or双变量的密度估计绘图，图片离散化    map用法没详细看，应该比较广泛

# 无限画图我日!
# facet.set(xlim=(0, titanic_df['Age'].max())) # 限制x坐标
# facet.add_legend()  # 加图例
# fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = titanic_df[["Age", "Survived"]].groupby(['Age'],as_index=False) # 每个年纪的人的生存率/平均生还几率
# sns.barplot(x='Age', y='Survived', data=average_age)

titanic_df.drop("Cabin",axis=1,inplace=True)  # 直接舍弃
test_df.drop("Cabin",axis=1,inplace=True)

# Family

titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1  # loc好像不必要？还是说会出现copy的问题= =  其实不用这个归一化还可以发现三个亲戚的生存率最高
# titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0 这一条感觉完全不需要，自己注了

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)  # axis?
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

# fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))
# sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)

family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean() # 注意groupby以及切片的用法，还不太熟悉= =
# sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)




def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex







def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex


titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)  # ATTENTION!!!这里的用法！；另外，这里表示取age和sex两列的数据
# axis : 0表示按列，1表示按行  （这就是之前德神说的那个批量转换，很好用！）

test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)
# print person_dummies_titanic

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)





# x =  titanic_df[['Age','Sex']]
# print help(x.apply)
sns.plt.show()
