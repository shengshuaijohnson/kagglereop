# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

# f = open('B-small-attempt1.in','r')
# import sys
# sys.stdin = f


T = int(raw_input())
# T =1
for t in xrange(1, T + 1):
    stalls, people = [int(i) for i in raw_input().split(' ')]
    # stalls, people = 1000, 1

    pos = ['1'] + ['0'] * stalls + ['1']
    while people:
        chosen = -1
        max_right = -1
        max_left  = -1
        for index in range(len(pos)):
            if pos[index] == '1':
                right = 0
            else:
                left = pos[index:].index('1') - 1
                # print right, left
                if min(right, left) > min(max_right, max_left):
                    max_right, max_left = right, left
                    chosen = index
                elif min(right, left) == min(max_right, max_left):
                    if max(right, left) > max(max_right, max_left):
                        max_right, max_left = right, left
                        chosen = index
                right += 1
        pos[chosen] = '1'
        people -= 1
        # print pos
    print "Case #{}: {} {}".format(t, max(max_right, max_left), min(max_right, max_left))

