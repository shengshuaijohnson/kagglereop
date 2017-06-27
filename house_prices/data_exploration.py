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


def get_boxplot(var='OverallQual', figsize=(8, 6)):
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=figsize)
    fig = sns.boxplot(x=var, y="SalePrice", data=data, whis=1.5)
    fig.axis(ymin=0, ymax=800000);
# print df_train[df_train['OverallQual']==7]['SalePrice']
# 封装好后查看其它feature如可以发现有时会有很多‘异常值’产生，思考：异常值较多的case（尤其是不设定X的情况）与较少的case相比较的含义？

# 2. First things first: analysing 'SalePrice'
# print df_train['SalePrice'].describe()      # hin强大的函数！注意The output DataFrame index depends on the requested dtypes:
# 对于数值型，返回count，平均数，标准差，最大最小，四分位的值

# sns.distplot(df_train['SalePrice'])     # 常见的distplot绘制分布情况（单因子）
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())


# important！！！看图后得出的结论： 1.Deviate from the normal distribution 2.Have appreciable positive skewness. 3.Show peakedness.
# 以前看这种图都没什么特别的感想，这次有角度可以参考一下
# 注意skewness在统计学中的定义，通俗的说正态分布偏度为0，左偏态为正偏度，右偏态为负偏度
# Kurtosis是峰度。大于0代表比正态陡，小于0代表比正态平缓      more info： https://en.wikipedia.org/wiki/Kurtosis
# 观察wiki中各类函数的kurt及图形比较，可以发现6.5已属于较高的锋度


# 了解完价格本身后了解相关feature

def get_scatterplot(var='GrLivArea'):   # GrLivArea: Above grade (ground) living area square feet
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    # 散点图评价方法：X-Y关系接近线性，说明相关性强；对于TotalBsmtSF也比较像线性，但是可以看出几个强烈的不相关点，balabala
    # 另一个极端的case就是YrSold，明显相关性不强
# get_scatterplot('TotalBsmtSF')


# get_boxplot('OverallQual')      # OverallQual: Rates the overall material and finish of the house
# 一方面看异常值多不多，一方面看box的位置，可以看出价格随着OverallQual是稳步上升的，当然data desc已经说是材料估值评级之类的概念了
# 如何看图很重要！！很长一段时间内对这图没什么感想！带失误！！！

# get_boxplot('YearBuilt',)    # 与制造年份相关性较低，略有正相关的趋势,看不清年份的话用xtick 设置rotation
'''
Summary:
'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships are positive, which means that as one variable increases, the other also increases. In the case of 'TotalBsmtSF', we can see that the slope of the linear relationship is particularly high.
'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of 'OverallQual', where the box plot shows how sales prices increase with the overall quality.

We just analysed four variables, but there are many other that we should analyse. The trick here seems to be the choice of the right features (feature selection) and not the definition of complex relationships between them (feature engineering).
That said, let's separate the wheat from the chaff.

'''
# 注意 feature election和feature engineering的区别！


# 3. Keep calm and work smart     Let's overcome inertia and do a more objective analysis.

# To explore the universe, we will start with some practical recipes to make sense of our 'plasma soup':
# Correlation matrix (heatmap style).
# 'SalePrice' correlation matrix (zoomed heatmap style).
# Scatter plots between the most correlated variables (move like Jagger style).

# 开始进行定量分析与矩阵的学习, 重点学习heatmap的使用！

#correlation matrix

corrmat = df_train.corr()   # 注意此处是先计算相关性，然后用相关矩阵做heatmap
# f, ax = plt.subplots(figsize=(12, 8))
# sns.heatmap(corrmat, vmax=0.8, square=True)
# corrmat.sort(columns=['SalePrice'], inplace=True)
# print corrmat['SalePrice']
# 通过上面这两行排序后看也是可以的，图片可能更明显点

# 与Saleprice 之间颜色最深的是Overall
# 相关性的模式(系数)有三种：
# pearson : standard correlation coefficient 默认模式   https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
# kendall : Kendall Tau correlation coefficient   https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
# spearman : Spearman rank correlation       https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
# 深入看各种统计方式的话好复杂=。= 先大概看一下默认的pearson

# Pearson correlation coefficient：用来衡量线性相关度，（相较之下，spearman则是衡量单调关系，未必要线性）在-1~+1之间，正负号代表正相关或负相关，绝对值越接近1代表线性吻合度越高，为0时则是杂乱无规律 
# 计算方式是协方差（X,Y）/标准差(X)*标准差(Y)

# This function tries to infer a good colormap to use from the data, but this is not guaranteed to work, so take care to make sure the kind of colormap (sequential or diverging) and its limits are appropriate.
# 关于tick label方向的问题： https://stackoverflow.com/questions/27037241/changing-the-rotation-of-tick-labels-in-seaborn-heatmap




'''
HEAT MAP:
wiki   https://en.wikipedia.org/wiki/Heat_map
事实上这里比较关键的是corr函数，算出矩阵后放到heatmap的过程反而比较简便，当然heatmap也有各种复杂情况拓展应用。
上面的作图法只显示颜色（下面那个显示数值），对色盲还是有一定程度的不友好的，而且颜色我赶脚也不是很精确。
另实测即使数据整体乘一个大系数，图片仍然不变，因此应该是内置了类似归一化的计算
vmax和vmin影响颜色，虽然help里描述不多，但应该是上下限。  （数值越大颜色越浅, not sure）
square 参数为True保持小方格是正方形
看heatmap的方法除了上述的光看一下与价格相关的行列，还可以看除了对角线外其他深色格子，如'TotalBsmtSF'和'1stFlrSF' 接近黑色
注意这句：Actually, this correlation is so strong that it can indicate a situation of multicollinearity.
multicollinearity 多重共线性 ： multicollinearity (also collinearity) is a phenomenon in which two or more predictor variables in a multiple regression model are highly correlated, meaning that one can be linearly predicted from the others with a substantial degree of accuracy. 
In this situation the coefficient estimates of the multiple regression may change erratically in response to small changes in the model or the data.
关于多重共线性导致的结果，可以有更进一步的理解。
'''




#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index # get the rows of a DataFrame sorted by the `n` largest
# 返回的结果也已经排好序了
cm = np.corrcoef(df_train[cols].values.T)
# cm = np.corrcoef(df_train[cols].values, rawvar=0)

# corrcoef :Return Pearson product-moment correlation coefficients  (as ndarray)
# 注意这里之所以要转置，是因为原来的shape 为(1460,10),rowvar默认为1是根据行为变量算的，返回1460*1460的结果，耗时长，结果也无意义
# 可以选择转置，也可以选择设置rowvar为0,得到10*10的正确结果
sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# 传入了许多参数的heatmap。。。  annot 影响格子里是否有数字
# 记得生成图之后要浏览一下这前10的feature！

# heatmap我曹！看起来屌屌的!



# 下面开始生成散点图对
sns.set()       # help里是Set aesthetic parameters in one step.

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 1.5)            # pairplot,传入N组数据自动扩展成N * N个散点图，要生成半天速度好慢，图太J2多了点


# 选取相关性最高的几个

plt.yticks(rotation=0)      # 这里控制坐标轴标签反转！
plt.xticks(rotation=-90)
sns.plt.show()
