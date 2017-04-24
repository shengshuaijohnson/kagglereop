# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>
# tutorial： https://www.kaggle.com/neviadomski/house-prices-advanced-regression-techniques/how-to-get-to-top-25-with-simple-model-sklearn

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle       # 这几行import都好妖路。。。跟以前看到的不太一样
# r2_score 量化表示拟合准确度，最高为1，可为负
# mean_squared_error 均方误差  （别误解error的意思了）  (ai-bi) ** 2 i=1->n累加  再除以n 后开平方
# 和方差 SSE(The sum of squares due to error)= 对应点误差的平方和
# 均方差 MSE(Mean squared error)             = SSE/n
# 均方根 RMSE(Root mean squared error)       = sqrt(MSE)

# train_test_split 是用来在测试数据中分割部分训练，部分测试的（之前用的是全用train，再用同一批数据看拟合程度）

# Elastic net  弹性网络  -- hybrid of Lasso  and Ridge Regression techniques

# TODO:学习正则化，L1，L2相关知识，备忘录里的文章记得看

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
# print train[train['BsmtFullBath'].isnull()]


# print train['LotFrontage'].isnull()

# s =  train.isnull().sum(axis=0)  # 其实和上面是一样的功能，只是自己尝试一下别的表达方法
# print s.iloc[s.nonzero()[0]].count() # train 19个,test 33个 可以发现在这个问题里有大量的可能为空的feature   另外空数据各自有7000个


def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))                # help里写明参数顺序是true再pre，但是即使反过来结果也是一样的
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# print help(mean_squared_error)
def train_test(estimator, x_train, x_test, y_train, y_test):         
    # 这个函数的设计可以在各类不同模型的评估上得到复用，值得借鉴
    # 主要原因是sklearn的模型里的预测都是统一的predict方法，才得以实现，否则可能要传入函数之类的
    # 感觉x_train,y_train,x_test,y_test的参数顺序会更好
    prediction_train = estimator.predict(x_train)           
    print  (estimator)
    get_score(prediction_train, y_train)

    prediction_test = estimator.predict(x_test)
    print "Test"
    get_score(prediction_test, y_test)


def fillna_with_popular(df, col):
    # 经测试df是可变类型，不需额外返回（当然也可以设计成带inplace参数的）
    # 或者说，既然存在可以设置inplace参数的方法，那么不测试也可以推出df必然是可变类型
    df[col] = df[col].fillna(df[col].mode()[0])


# facet = sns.FacetGrid(train, aspect=4)
# facet.map(sns.kdeplot,'SalePrice',shade= True) # 房价分布，这个函数比较陌生，纵坐标还不会改


# ax = sns.distplot(train['SalePrice'],kde=False)           # 我晕，直接用这个就可以代替上面的了，当然结果不太一样，不过大致可视化上都差不多。
                                                            # 另外此方法里也有默认为True的kde参数：Whether to plot a gaussian kernel density estimate.影响是否绘制预估曲线。
                                                            # 注意这里的高斯 kernel


train_labels = train.pop('SalePrice')       # 干脆叫train_Y多方便=。=

features = pd.concat([train, test], keys=['tarin', 'test']) # 默认axis=0,即按行添加
# 这个操作比较不一样,直接按行拼在一起了。。


# 下面的drop操作比较主观，这里是丢弃空数据较多或者作者认为与价格无关的feature(岂不是意味着要吧80来个feature含义都看一遍?)
# 但是看到Alley没被drop，或许是作者这个or表述不精确，是同时考虑NA以及价格相关两个因素？到后面noacess的处理后可以再回来看这边
features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)


# print features.columns            # 缩减到56 features(include Id)

features['MSSubClass'] = features['MSSubClass'].astype(str)     # 蛤？数字转成字符？？

# print set(features['MSZoning'].values)      # 大概看一下有哪些值，我凭直觉直接写出这个表达式，太TM机智了


fillna_with_popular(features, 'MSZoning')
# print (features['LandContour'].mode())       # mode返回出现频率最高的data,如果有并列情况则一并返回(先后顺序未知) (自测过)

fillna_with_popular(features, 'LotFrontage')
# Alley  NA in all. NA means no access
features['Alley'] = features['Alley'].fillna('NOACCESS')


features.OverallCond = features.OverallCond.astype(str) # 我靠，这种语法也是支持的么？

fillna_with_popular(features, 'MasVnrType')


# 这个和之前的ALLEY一样，都是吧desp里NA有定义的给换成其相应定义，话说这几个词真是XNMBYY。。。
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBSMT')

# 这个是作者假设的
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)


fillna_with_popular(features, 'Electrical')


features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)     
# 全部都要转成str？还是部分转？全部的话可以干脆些一个函数全转了啊，TODO：看看有没有显示每一列属性的方法
# 感觉kaggle上有些人的写法好僵硬，代码疯狂repeat yourself，为什么不封装成通用的函数，是coding方面比较弱，还是在展示解法的时候不喜欢写函数？有其它的考量？
fillna_with_popular(features, 'KitchenQual') 

features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')

for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    features[col] = features[col].fillna('NoGRG')

features['GarageCars'] = features['GarageCars'].fillna(0.0)     



fillna_with_popular(features, 'SaleType')

# Year and Month to categorical
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)


# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)
# 直接算总面积，我佛。

# 到这里我明白了，一些量化的feature为数字形式就不转，而一些是数字形式，含义却是一类标志，没有相应 1+1=2运算法则的feature就转成str
# 这里有个chinglish使用者的究极问题在： 表述“一些feature”的时候到底单数形式还是用复数形式的“一些features”呢？ （港三小？）
# ax = sns.distplot(train_labels)


numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]       # 切片还可以不是数字的！注意loc和iloc的区别
# print numeric_features
numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()
# attention！！！利用除以标准差进行standardize，统一转为 -1 ~ 1之间的数字      TODO：测试模型的时候不转看看是否有影响，我感觉理论上最佳情况并不会影响结果，反正都是对拟合出的omega向量进行缩放

ax = sns.pairplot(numeric_features_standardized)        # 好屌啊，不同栏之间的相对分布,图的个数为L*L，L是传入df的col长度


conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']]) 

# conditions2 = set(list(features['Condition1'].values + features['Condition2'].values))  # 这样写之所以会出错是因为ndarray的相加会将其中各个元素的值直接相加，而不是列表的extend效果




# ======== 以下为自己粗暴地抛去所有na的拟合练手，主要熟悉pd操作,以及拟合的模型（此前用的都是分类的）
'''
NAs = pd.DataFrame(train.isnull().sum())

# print type(NAs>0)
print '##########'
# print type(NAs.sum(axis=1) > 0)
null_cols_df =  NAs[NAs.sum(axis=1) > 0] # 之所以不字节用NAs是因为会返回df，而不是Series


NAs_test = pd.DataFrame(test.isnull().sum())
null_cols_df_test =  NAs_test[NAs_test.sum(axis=1) > 0] 

all_null =  null_cols_df_test.index.union(null_cols_df.index)

train.drop(list(all_null), axis=1, inplace=1)
test.drop(list(all_null),  axis=1, inplace=1)


def string_to_num(df,col):  
    # 用dummies其实也可以，但是因为有多个值的情况索性直接这样了
    # 准确的说用dummies的模式应该是更好的，否则例如0-24那一栏拟合就拟合爆炸了，分成不同栏以1，0表示在计算上更合理
    try:
        a = float(df[col][0])
    except:
        val_set = set()
        for i in df[col]:
            val_set.add(i)
        # print len(val_set)        # 最多有25个
        val_dict = dict(zip(list(val_set),range(len(val_set))))
        df[col] = df[col].apply(lambda a:val_dict[a])
    # 写完发现过于暴力，搜不鸟列




for col in train.columns:
    string_to_num(train, col)
for col in test.columns:
    string_to_num(test, col)
train_features = train.drop('SalePrice',axis=1)
train_labels = pd.DataFrame(train['SalePrice'])




x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.0001, 0.001, 0.01, 0.1, 1], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train, y_train) # 这个参数未收敛，暂时不管了
train_test(ENSTest, x_train, x_test, y_train, y_test)
# TODO：查阅 ElasticNetCV模型相关知识，参数含义
# 妈个鸡，RMSE大的批爆，几万，果然不行

# Final_labels = ENSTest.predict(test)
# pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('2017-04-17.csv', index =False)  
# 这样算出来的数据甚至有负数，kaggle上提交也是error，不知道是不是单纯负值的原因
# ======
'''

# print help(linear_model.ElasticNetCV)


sns.plt.show()
