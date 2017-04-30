# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

T = int(raw_input())



def ride(current_horse, remain_distance, current_city, used_time):
    global graph, horse_distance, speed, N
    if remain_distance < graph[current_city][current_city + 1]: # 跑不动
        return 99999999
    # print current_city, current_horse
    used_time = used_time + 1.0 * graph[current_city][current_city + 1] / speed[current_horse]
    remain_distance -= graph[current_city][current_city + 1]
    next_city = current_city + 1
    if next_city == N -1:        # 跑完了
        return used_time
    else:
        return min(ride(next_city, horse_distance[next_city], next_city, used_time), ride(current_horse, remain_distance, next_city, used_time))


def get_farest_city(graph, pos):
    global N
    for i in range(N)[::-1]:
        if graph[i] <= pos:
            return i


for t in xrange(1, T + 1):
    N, Q  = [int(i) for i in raw_input().split(' ')]
    horse_distance = []
    speed    = []
    graph = [0]
    for _ in range(N):
        e, s = [int(i) for i in raw_input().split(' ')]
        horse_distance.append(e)
        speed.append(s)
    for index in range(1, N + 1):
        # graph.append([int(i) for i in raw_input().split(' ')])        # 这是二维的，一维情况用数组就可以
        a = [int(i) for i in raw_input().split(' ')]
        # if index == 0:
        if index < N:
            graph.append(a[index] + graph[index -1])


    start = []
    end = []
    for _ in range(Q):
        U, V = [int(i) for i in raw_input().split(' ')]         # begin from 1!!!
        start.append(U - 1)
        end.append(V - 1)
    current_horse = 0
    current_horse_distance = horse_distance[current_horse]  # remain
    current_speed    = speed[current_horse]
    used_time = 0
    time_map = []
    # print ride(0, horse_distance[0], 0, 0)
    # time_map.append(
    #     {
    #         'used_time': 0,
    #         'remain_distance': horse_distance[0],
    #         'speed'         
    #     }
    # )

    time_map = {}
    time_map[0] = 0
    for city in range(N):
        far_city = get_farest_city(graph, graph[city] + horse_distance[city])
        for check_city in range(city + 1, far_city + 1):
            used_time = 1.0 * (graph[check_city] - graph[city]) / speed[city] + time_map[city]
            if not time_map.get(check_city, False):
                time_map[check_city] = used_time
            else:
                time_map[check_city] = min(time_map[check_city], used_time)
    print "Case #{}: {}".format(t, time_map[N-1])



