# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

train_df = pd.read_csv('train.csv')
test_df   = pd.read_csv('test.csv')
print train_df.head()
print train_df.shape


