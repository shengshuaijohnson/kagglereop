# !/usr/bin/env python 
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

# this scrip concentrate on data visualization and learn the features but not machine learning -- qrr
# kernel url: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python



import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
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







