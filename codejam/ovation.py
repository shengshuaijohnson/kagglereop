# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>


# raw_input() reads a string with a line of input, stripping the '\n' (newline) at the end.
# This is all you need for most Google Code Jam problems.
# t = int(raw_input())  # read a line with a single integer
# for i in xrange(1, t + 1):
#   n, m = [int(s) for s in raw_input().split(" ")]  # read a list of integers, 2 in this case
#   print "Case #{}: {} {}".format(i, n + m, n * m)
#   # check out .format's specification for more formatting options
#   

T = int(raw_input())
for t in xrange(1, T + 1):
    smax, audience = raw_input().split(' ')
    smax = int(smax)
    shyness = 0
    stand = 0
    res = 0
    for i in audience:
        audi_num = int(i)
        # print stand,shyness, res
        if stand >= shyness:    # ok situation
            stand += audi_num
        elif audi_num > 0:
            res += shyness - stand
            stand = shyness + audi_num
        shyness += 1
    print "Case #{}: {}".format(t, res)
