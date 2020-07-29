from bs4 import BeautifulSoup as bs
import os
import sys
import time
import jieba
import shutil
import pandas as pd

def data_merge(xlsx_list):
    merge_xlsx = []
    for idx, xlsx in enumerate(xlsx_list):
        # merge xlsx file
        data = pd.read_excel('./link/' + xlsx)
        data['文件id'] = ''
        for i in range(len(data)):
            data.loc[i]['文件id'] = data.loc[i]['学校'] + str(i)
            print(data.loc[i]['文件id'])
            
        merge_xlsx.append(data)
        
        
    writer = pd.ExcelWriter('标注数据.xlsx')
    pd.concat(merge_xlsx).to_excel(writer,'1',index=False)    
    writer.save()
    
if __name__ == '__main__':
    white_list = ['厦门医学院.xlsx', '厦门理工学院.xlsx', '福建中医药大学.xlsx', '福建农林大学.xlsx', 
                  '福建医科大学.xlsx', '福建工程学院.xlsx', '福建师范大学.xlsx', '莆田学院.xlsx',
                  '闽南师范大学.xlsx', '闽江学院.xlsx', '集美大学.xlsx', '福州大学.xlsx', '厦门大学.xlsx']
    data_merge(white_list)
    print('OK.')
    
    
