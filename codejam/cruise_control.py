# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>




T = int(raw_input())


for t in xrange(1, T + 1):
    destination, horse_num = [int(num) for num in raw_input().split(' ')]
    time_spend = []
    for _ in range(horse_num):
        pos, speed = [float(num) for num in raw_input().split(' ')]
        time_spend.append((destination - pos) / speed)
    res = destination / max(time_spend)
    print "Case #{}: {}".format(t, res)



