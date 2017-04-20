# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yixuan Zhao <johnsonqrr (at) gmail.com>
# 测例：正常：532499544594,418766752579;无进展：982202832284;国际：312574842215


import json
import requests
import lxml.html
import time
import re

header = {
'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
'Accept-Encoding':'gzip, deflate',
'Accept-Language':'zh-CN,zh;q=0.8,en;q=0.6,zh-TW;q=0.4,ja;q=0.2',
'Cache-Control':'max-age=0',
'Connection':'keep-alive',
'Content-Length':'20',
'Content-Type':'application/x-www-form-urlencoded',
# 'Cookie':'ASP.NET_SessionId=34osrid5tfrdomtgfabftc2e; UserBills=["982202832284","419301776745","509985361111","128663450338"]; Hm_lvt_a568d570cd1ef66819cda1e7d5cbbe08=1492684486; Hm_lpvt_a568d570cd1ef66819cda1e7d5cbbe08=1492685053; Hm_lvt_53a93979e64ab8e76c06653f6830c385=1492684486; Hm_lpvt_53a93979e64ab8e76c06653f6830c385=1492685053',
'Cookie' : 'ASP.NET_SessionId=34osrid5tfrdomtgfabftc2e; UserBills=["982202832284","419301776745","509985361111","128663450338","405887775211","532122677728","408065610697","419334339883","415669707661"]; Hm_lvt_a568d570cd1ef66819cda1e7d5cbbe08=1492684486; Hm_lpvt_a568d570cd1ef66819cda1e7d5cbbe08=1492687476; Hm_lvt_53a93979e64ab8e76c06653f6830c385=1492684486; Hm_lpvt_53a93979e64ab8e76c06653f6830c385=1492687476',
'Host':'www.zto.com',
'Origin':'http://www.zto.com',
'Referer':'http://www.zto.com/GuestService/BillNew',
'Upgrade-Insecure-Requests':'1',
'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
}



url = 'http://www.zto.com/GuestService/BillNew'


def write_to_file(item, output_file='output.txt'):
    with open(output_file, 'a') as f:
        a = json.dumps(item, ensure_ascii=False)
        print a
        f.write(a.encode('utf-8'))
        f.write('\n')

def process_single_bill(bill_id):
    # TODO:2秒间隔下爬取时间也会有3小时，考虑到学校网络不稳定，可以作暴力一点的断点续传设计
    # 快递状态大致分类：未寄件（本批数据较老，不会有这种情况）；已寄件未签收（本批数据也应该不会有）；已寄件且已签收；海外快递；无运单信息。
    for _ in range(3):
        try:
            res = requests.post(url, data={'txtBill': bill_id}, headers=header)
            break
        except requests.exceptions.ConnectionError:
            print 'Oops!ConnectionError:('
            time.sleep(10)
    else:
        return False            # 连续3次失败后，返回False

    content = res.content

    
    item = {
        u'bill_id'               : bill_id,
        u'origin_city'           : None,             # 印象中某些情况下数据库对于这个None不太友好？不合适的话可以改成''
        u'origin_post_part'           : None,
        u'destination_city'      : None,
        u'destination_post_part'      : None,
        u'delivery_date'         : None,
        u'arrival_date'          : None,
        u'postman_phonenumber'   : None,
        u'status'                : 'OK',                    
    }
    dom = lxml.html.fromstring(res.content)
    state_node = dom.xpath('//div[@class="state"]')[0]
    steps = state_node.xpath('.//li')
    if not steps:              
        # 订单信息异常,直接记录整个文本在status里了,暂时粗暴一点不做细化分类
        item[u'status']  = u''.join(state_node.xpath('.//text()')).strip()
        write_to_file(item)
    else:       #   正常情况，开始取数据
        origin_node = steps[0].xpath('./div[@class="on"]')[0]
        origin_city = origin_node.xpath('./span/text()')[0]
        origin_city = origin_city.replace(u'[', u'').replace(u']', u'')
        origin_post_part = origin_node.xpath('./a/text()')[0]
        item[u'origin_city'] = origin_city
        item[u'origin_post_part'] = origin_post_part
        item['delivery_date'] = steps[0].xpath('./div[@class="time"]/text()')[0]    
        # 时间直接保留原格式了，进一步的处理待数据清洗步骤完成

        dest_node = steps[-1].xpath('./div')[0]
        destination_city = dest_node.xpath('./span/text()')[0]
        destination_city = destination_city.replace(u'[', u'').replace(u']', u'')
        destination_post_part = dest_node.xpath('./a/text()')[0]
        item[u'destination_city'] =  destination_city
        item[u'destination_post_part'] = destination_post_part
        item[u'arrival_date'] = steps[-1].xpath('./div[@class="time"]/text()')[0]    

        # 地点和时间提取完，接下来检查是否有送货员电话，为了便捷起见，采取较直接的方式获取
        # 判断方法有：匹配[数字]；查找'正在派件'；直接取第二个节点等。
        for node in steps:
            text = u''.join(node.xpath('.//text()')).strip()
            if u'正在派件' in text:
                phone = re.findall('\[(\d+)\]', text)
                if phone:
                    item[u'postman_phonenumber'] = phone[0]
                break

        write_to_file(item)



    # 为了便于数据库储存，使用json格式。如果要保存到csv，需要使用pandas

def test(gap, times):        
    # 最开始测试gap时process方法里的语句只有发送request和打印len(content)，之后的若干处理都还没写
    # 测试结果：无间隔20次ban,持续10~20分钟左右;3秒间隔连续50次未被ban;2秒间隔200次未被ban
    for count in range(times):
        print count
        process_single_bill('532499544594')  
        time.sleep(gap)         

def main():
    file_name = 'zto_order_id.csv'
    with open(file_name) as f:
        zto_numbers = [line.strip() for line in f]
    zto_numbers = list(set(zto_numbers))        # 竟然有数据重复，我太TM机智了,据我观察是文件头尾的数据有重复
    counte = 0
    for bill_number in zto_numbers:
        print counte
        counte += 1
        process_single_bill(bill_number)
        time.sleep(2)
# test(0, 1)

if __name__ == '__main__':
    main()