# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

# f = open('B-small-attempt1.in','r')
# import sys
# sys.stdin = f


T = int(raw_input())

for t in xrange(1, T + 1):
    # N = raw_input()
    N = [i for i in raw_input()]
    for index in range(len(N))[:-1]:
        current = N[index]
        right    = N[index+1]
        if current <= right:
            continue
        # elif right == '0': 
        else:
            N[index + 1:] = ['9' for i in range(index + 1, len(N))]
            N[index] = chr(ord(N[index]) -1)
        # else:
        #     N[index + 1] = '9'
        #     N[index] = chr(ord(N[index]) -1)
    # print N
    for index in range(len(N))[1::][::-1]:
        current = N[index]
        left    = N[index-1]
        if current >= left:
            continue
        elif current == '0':
            N[index] = '9'
            N[index-1] = chr(ord(N[index-1]) -1)
        else:
            N[index] = '9' 
            N[index-1] = chr(ord(N[index-1]) -1)
    if N[0] == '0':
        N.pop(0)
    print "Case #{}: {}".format(t, ''.join(N))


