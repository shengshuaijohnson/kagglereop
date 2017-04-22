# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>


def fill_with_popular(df, col):
    # 经测试df是可变类型，不需额外返回（当然也可以设计成带inplace参数的）
    # 或者说，既然存在可以设置inplace参数的方法，那么不测试也可以推出df必然是可变类型
    df[col] = df[col].fillna(df[col].mode()[0])


def remove_duplication_by_line(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f]
        print 'Before:{}'.format(len(lines))
        lines = list(set(lines))
        print 'After:{}'.format(len(lines))

    with open('{}_new'.format(filename), 'w') as f: # 本来文件名是不变的, 但是感觉太危险了一点,问题是加个new感觉用起来也很不方便，尴尬。。。
        for line in lines:
            f.write(line)
            f.write('\n')
