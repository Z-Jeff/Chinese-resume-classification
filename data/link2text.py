import os
import sys
import ssl
import time
import chardet
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
from urllib import request
from urllib.request import urlopen
ssl._create_default_https_context = ssl._create_unverified_context

# 获取文件列表中所有文件
link_files = os.listdir('./link')
# 也可以使用白名单，只爬取某些学校的数据
whitelist = ['厦门医学院.xlsx', '厦门理工学院.xlsx', '福建中医药大学.xlsx', '福建农林大学.xlsx', '福建医科大学.xlsx', '福建工程学院.xlsx', '福建师范大学.xlsx', '莆田学院.xlsx', '闽南师范大学.xlsx', '闽江学院.xlsx', '集美大学.xlsx']
link_files = whitelist

print(link_files)

# 去除非xlsx文件        
link_files = [f for f in link_files if f[0]!='.' and f.split('.')[-1]=='xlsx']

# 获取断点: (文件名:表格行数)，使得程序可以在断点出继续运行
# 运行时添加 from-scratch 参数，复位断点，从头爬取
breakpoint = '_.xlsx:0'
log_file = open('./link/log.txt', 'r+')
if len(sys.argv)>1 and sys.argv[1] == 'from-scratch':
    breakpoint = '_.xlsx:0'
    print('fetch data from scratch...')
else:
    breakpoint = log_file.readline()
    if breakpoint == '':
        breakpoint = '_.xlsx:0'
print(breakpoint)
breakfile = breakpoint.split(':')[0]
breakline = int(breakpoint.split(':')[1])

# 按文件迭代
for link_file in link_files:
    # 恢复文件断点
    if breakfile != '_.xlsx':
        if breakfile != link_file:
            continue
    df = pd.read_excel('./link/' + link_file)
    
    # 按表格中的行迭代
    for i in range(len(df)):
        # 恢复行数断点
        if breakfile != '_.xlsx':
            if i <= breakline:
                continue
        if not isinstance(df.loc[i]['个人主页'], str):
            continue

        try:
            print('第',i,'条数据: ', df.loc[i]['姓名'], df.loc[i]['个人主页'])
            req = request.Request(df.loc[i]['个人主页'])
            # 添加请求头，伪装成浏览器，获取网页
            req.add_header('User-Agent', 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)') 
            html = urlopen(req, timeout=10).read()
            #time.sleep(0.3)
            # 网页解码尽量使用兼容性高的解码方式
            encoding = chardet.detect(html)['encoding']  
            if encoding == 'GB2312':
                encoding = 'gbk'
            elif encoding == 'iso-8859-1':
                encoding = 'utf-8'
            html = bs(html, 'html.parser', from_encoding=encoding)
            
        except Exception as r:
            print('Error:', r)
            continue
        
        # 去除网页中的 script 和 style，剩下文本信息
        [script.extract() for script in html.findAll('script')]
        [style.extract() for style in html.findAll('style')]
        text = str(html.text)
        # 去除文本中的空行，空格
        text = ''.join([s.strip()+'\n' for s in text.splitlines(True) if s.strip()]) 
        # 将文本信息保存到 text/ 文件夹下
        save_file = './text/original/' + df.loc[i]['学校'].replace('/', '-').replace(' ', '-') + str(i) + '.txt'
        with open(save_file, 'w') as f:
            f.write(text)
        print(text)
        print(save_file, end='\n\n\n')
        # 保存断点
        log_file.seek(0)
        log_file.truncate()
        log_file.write(link_file + ':' + str(i))
    # 更新断点
    if breakfile == '_.xlsx':
        breakfile = link_files[1]
    elif breakfile != link_files[-1]:
        breakfile = link_files[link_files.index(breakfile) + 1]
    breakline = -1
    
    
        
        
        
