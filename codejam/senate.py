# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>



f = open('A-large-practice.in','r')
import sys
sys.stdin = f

T = int(raw_input())
for t in xrange(1, T + 1):
    N = int(raw_input())
    nums = [int(s) for s in raw_input().split(' ')]
    partys = len(nums)
    total = sum(nums)
    left = right = ''
    print 'Case #{}: '.format(t),
    while partys:
        max_index = nums.index(max(nums))
        if len(left) >= len(right):
            right += nums[max_index]*chr(ord('A') + max_index)
        else:
            left += nums[max_index]*chr(ord('A') + max_index)

        nums[max_index] = -1
        partys -= 1
    while len(left) > len(right):
        print left[-1],
        left = left[:-1]
    while len(right) > len(left):
        print right[-1],
        right = right[:-1]
    for index in range(len(right))[::-1]:
        print '{}{}'.format(left[index],right[index]),
    print 
