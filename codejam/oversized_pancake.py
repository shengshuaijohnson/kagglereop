# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

# f = open('A-large-practice.in','r')
# import sys
# sys.stdin = f

T = int(raw_input())
def flip(pancake, start, flipper_size):
    for index in  range(start, start + flipper_size):
        if pancake[index] == '+':
            pancake[index] = '-'
        else:
            pancake[index] = '+'

for t in xrange(1, T + 1):
    # N = int(raw_input())
    # nums = [int(s) for s in raw_input().split(' ')]
    raw_string = raw_input()
    pancake, flipper_size = raw_string.split(' ')[0], int(raw_string.split(' ')[1])
    pancake = [i for i in pancake]
    index = 0
    res   = 0
    length = len(pancake)
    for i in pancake:
        if index + flipper_size > length:
            if '-' in pancake:
                res = 'IMPOSSIBLE'
                break
        else:
            if i == '+':
                pass
            else:
                res += 1
                flip(pancake, index, flipper_size)
        index += 1

    print "Case #{}: {}".format(t, res)
    