import os
import re
import sys
import time
import jieba
import shutil
import pandas as pd

### 单个字符的属性判断 ###
class char_():
    # 判断一个unicode是否是汉字
    def is_chinese(uchar):
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5': return True
        else: return False
    # 判断一个unicode是否是数字
    def is_number(uchar):
        if uchar >= u'\u0030' and uchar<=u'\u0039': return True
        else: return False
    # 判断一个unicode是否是英文字母 
    def is_alphabet(uchar):
        if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
        else: return False
    # 判断一个unicode是否是标点符号 
    def is_punctuation(uchar): 
        chinese_punctuation = '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
        english_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        punctuation = chinese_punctuation + english_punctuation
        if uchar in punctuation: return True
        else: return False

### 字符串的属性判断 ###            
class str_():
    # 判断一个字符串是否为数字
    def is_digit(string):
        if (string.split(".")[0]).isdigit() or string.isdigit() or  (string.split('-')[-1]).split(".")[-1].isdigit():    
            return True
        else: return False
    # 判断一个字符串是否为数字与字母组成的代码
    def is_code(string):
        has_other = False
        has_num = 0
        has_alpha = 0
        for c in string:
            if char_.is_number(c):
                has_num = 1
            if char_.is_alphabet(c):
                has_alpha = 1
            if char_.is_number(c) or char_.is_alphabet(c):
                has_other = False
            else:
                has_other = True
        if has_num and has_alpha and not has_other: return True
        else: return False
    # 判断一个字符串是否为百分数
    def is_percent(string):
        if string[-1] == '%' or string[-1] == '％':
            if str_.is_digit(string[:-1]): return True
        return False 

### 判断简历是否符合要求 ###
# 说明：简历中有存在空文档，信息少的文档，乱码文档等需要进行剔除

# 统计42794份简历文档中的中英文字符占文档所有字符的比例，结果如下：
# 平均值: 0.697816593
# 标准差： 0.086882445
# 把比例小于 0.5 的简历文档剔除

# 统计42794份简历文档中的字符个数，结果如下：
# 平均值：3440.491426
# 标准差：6073.776462
# 把字符个数小于 900 的简历文档剔除
def is_resumeFit(text):
    total_num = len(text) + 1
    if total_num < 900: return False
    
    ch_en_num = 0
    for c in text:
        if char_.is_chinese(c) or char_.is_alphabet(c):
            ch_en_num += 1
    if ch_en_num/total_num < 0.5:
        return False
    return True   

### 去除字符串中的空格 ###
def remove_blank(string):
    s = string
    #s = string.strip()
    s = s.replace(' ', '')
    s = s.replace('\xa0', '')
    s = s.replace('\u3000', '')
    s = s.replace('\t', '')
    #s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    return s        
    
### 文本预处理 ###               
def textPreprocess(text):
    # 文本分词
    seq_list = jieba.cut(text, cut_all=False)
    seq_list = list(seq_list)
    
    if os.getcwd().split('/')[-1] == 'data':
        stop_words_file = 'stop_words.txt'
    else:
        stop_words_file = './data/stop_words.txt'
    with open(stop_words_file, 'r') as f:
        stop_words = f.read().split('\n')[:-1]    
    
    for i in range(len(seq_list)-1, -1, -1):
        # 去除空格
        seq_list[i] = remove_blank(seq_list[i])
        if not len(seq_list[i]):
            del seq_list[i] 
            
        else: 
            # 去除停用词
            if seq_list[i] in stop_words:
               del seq_list[i] 
            # 数字替换为 _num_
            if str_.is_digit(seq_list[i]): 
                seq_list[i] = '_num_'
            # 数字+英文 替换成 _code_
            elif str_.is_code(seq_list[i]):
                seq_list[i] = '_code_'
            # 百分数 替换成 _percent_
            elif str_.is_percent(seq_list[i]):
                seq_list[i] = '_percent_'

    text = ' '.join(seq_list) 
    # 去除空行
    text = ''.join([s for s in text.splitlines(True) if s.strip()]) 
    return text
    


        
if __name__ == '__main__':  
    # 获取文件列表
    text_dir = './text/original/'
    save_dir = './text/processed/'
    text_files = os.listdir(text_dir)
    
    
    # 福建省外高校简历处理：需要判断简历是否符合要求
    '''
    num = 0
    for text_file in text_files:
        with open(text_dir + text_file, 'r') as f:
            text = f.read()
        if is_resumeFit(text):
            text = textPreprocess(text)
            with open(save_dir + text_file, 'w') as f:
                f.write(text)
        
            num += 1
            print(num, text_file)
    '''        
    
    # 福建省内高校简历处理：因为标注时需要人工核对，无需程序判断简历是否符号要求
    white_list = ['厦门医学院', '厦门理工学院', '福建中医药大学', '福建农林大学', '福建医科大学', '福建工程学院', '福建师范大学', '莆田学院', '闽南师范大学', '闽江学院', '集美大学', '福州大学', '厦门大学']
    temp = []
    for text_file in text_files:
        for school in white_list:
            if school in text_file:
                temp.append(text_file)
    text_files = temp
    num = 0
    for text_file in text_files:
        with open(text_dir + text_file, 'r') as f:
            text = f.read()
        #if is_resumeFit(text):
        text = textPreprocess(text)
        with open(save_dir + text_file, 'w') as f:
            f.write(text)
    
        num += 1
        print(num, text_file)
    
    
