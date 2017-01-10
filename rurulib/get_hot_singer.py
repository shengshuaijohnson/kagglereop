# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>

import requests
import time
import datetime
import json
import lxml.html
import sys
reload(sys)
sys.setdefaultencoding('utf-8')




def main():
    # 热歌前500
    url = 'http://music.baidu.com/top/dayhot'
    r = requests.get(url)

    dom = lxml.html.fromstring(r.content)
    node = dom.xpath('//span[@class="author_list"]')
    with open('baidu_hot_singer.txt', 'w') as f:
        for i in node:
            name =  i.xpath('.//text()')[1]
            f.write(name + '\n')

def from_singer_rank():
    url = 'http://music.baidu.com/top/artist'
    r = requests.get(url)

    dom = lxml.html.fromstring(r.content)
    node = dom.xpath('//div[@class="artist-name"]')
    node.extend(dom.xpath('//span[@class="artist-name"]'))
    with open('baidu_hot_singer.txt', 'w') as f:
        for i in node:
            name =  i.xpath('.//text()')[0]
            print name
            f.write(name + '\n')
# from_singer_rank()
def bijiao():
    fname1 = 'xiami_list_hot.txt'
    fname2 = 'alias_by_line.txt'
    name_pool = set()
    failed = set()
    with open(fname2, 'r') as f:
        for line in f:
            name_pool.add(line.strip())
    # print len(name_pool)
    with open(fname1, 'r') as f:
        for line in f:
            name = line.strip()
            if not name in name_pool:
                failed.add(name)
    for i in failed:
        print i

from_singer_rank()