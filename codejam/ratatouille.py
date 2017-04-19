# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>
# https://code.google.com/codejam/contest/5304486/dashboard#s=p1&a=1



T = int(raw_input())


def check(range1, range2):
    for i in range(range1[0], range1[1] + 1):
        if i >= range2[0] and i <= range2[1]:
            return True
    else:
        return False

for t in xrange(1, T + 1):
    find = 0
    N, P = [int(i) for i in raw_input().split(' ')]                  # 对于python的读取方式，P在读取数据过程中其实不是必要的（当然其他语言也不能算完全必要啦）
    recipe = [int(i) for i in raw_input().split(' ')]                # The i-th of these represents the number of grams of the i-th ingredient needed to make one serving of ratatouille.
    # 接下来的数据是N个原料，每个原料有P个包裹
    ingredients = []
    for _ in xrange(N):
        ingredients.append(sorted([int(i) for i in raw_input().split(' ')]))    # 复杂度为N * O(Plog(P))
    range_list = []
    for index in range(N): # 复杂度为 O(NP)
        ingredient = ingredients[index]  # [100,400,500,...]
        ranges = [(gram / 1.1 / recipe[index], gram / 0.9 / recipe[index]) for gram in ingredient]  
        range_list.append(ranges)
    points = [0] * N   # 给每个材料一个初始指针，这个指针只会扩大（右移），
    res = 0
    for std_point in range(P):  # 以第一个材料为标准，开始移动所有指针：
        std_range = range_list[0][std_point]    # format (10, 20)
        std_range = list(std_range)
        std_range = [int(std_range[0] + 0.99999999), int(std_range[1])]
        if std_range[0] > std_range[1]:
            continue
        for index in range(1, N):       # 遍历一以外的材料
            for point in range(points[index], P):
                if check(std_range, range_list[index][point]):
                    points[index] = point
                    break
            else:
                pass
                break   #return?
        else:
            res += 1
            continue
    print "Case #{}: {}".format(t, res)













