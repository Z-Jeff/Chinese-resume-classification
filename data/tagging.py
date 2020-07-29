#from selenium import webdriver
#from selenium.webdriver.common.action_chains import ActionChains
#from selenium.webdriver.common.keys import Keys
#from selenium.webdriver.support.select import Select
#import requests
#from bs4 import BeautifulSoup as bs
import os
import sys
import time
import json
import pandas as pd
from tkinter import *  
from tkinter import ttk
import tkinter.font as tkFont

# 白名单，确定哪些文件需要标注
whitelist = ['厦门医学院.xlsx', '厦门大学.xlsx', '厦门理工学院.xlsx',
                          '福州大学.xlsx', '福建中医药大学.xlsx', '福建农林大学.xlsx',
                          '福建医科大学.xlsx', '福建工程学院.xlsx', '福建师范大学.xlsx',
                          '莆田学院.xlsx', '闽南师范大学.xlsx', '闽江学院.xlsx', 
                          '集美大学.xlsx']

class App:  
    def __init__(self, master): 
        ''' GUI 初始化 '''
        frame = Frame(master)  
        frame.pack()
        self.frame = frame
        self.frame.bind("<Button-3>", self.delete)
        
        # 字体
        ft_big    = tkFont.Font(family='Fixdsys', size=21, weight=tkFont.BOLD)
        ft_middle = tkFont.Font(family='Fixdsys', size=13, weight=tkFont.BOLD)
        ft_small  = tkFont.Font(family='Fixdsys', size=9, weight=tkFont.BOLD)
        
        ''' GUI布局设置 '''
        # 1. 按照学科分类标准设置按钮
        # 学科大类 - 学科小类 - 学科 
        row = 1
        column = -1
        # category_merge.txt 中存在合并 :'经管', '文法哲', '教艺' 
        with open('category_merge.txt', 'r') as f:
            self.category = json.load(f)
        self.string_vars = []
        self.selected = ''
        self.subject_label = ''
        for key in self.category:
            column += 1
            self.label = Label(frame, text=key, font=ft_middle, padx=1, pady=1, fg="blue")
            self.label.grid(row=1, column=column)
                
            sub_category = self.category[key].keys()
            row = 0
            for c in sub_category:
                row += 2
                self.button = Label(frame, text=c, font=ft_small, padx=1, pady=1, fg="green")
                self.button.grid(row=row, column=column)
                width_set = [15,14,10,8,8,10,8,8]
                width = width_set[column]
                s = StringVar()
                self.dropdown = ttk.Combobox(self.frame, values=self.category[key][c], 
                                             state='readonly', width=width,
                                             textvariable=s
                                             )
                self.string_vars.append(s)
                self.dropdown.bind("<<ComboboxSelected>>", self.select)
                self.dropdown.grid(row=row+1, column=column)
        # 重新导入一次学科, 这里的 '经','管''文','法','哲','教','艺' 是分开的
        with open('category.txt', 'r') as f:
            self.category = json.load(f)    
        
        # 2. 设置三个按钮: <上一条> <下一条> <跳过>
        self.b_previous = Button(frame, text="跳转至上一条", fg="purple", command=self.previous, width=30, height=2, font=ft_middle)  
        self.b_previous.grid(row=30, column=0, columnspan=3)  
        self.b_skip = Button(frame, text="跳过(不做标注)", fg="purple", command=self.skip, width=10, height=2, font=ft_middle) 
        self.b_skip.grid(row=30, column=3, columnspan=2)
        self.b_next = Button(frame, text="标注并跳转至下一条", fg="purple", command=self.next, width=30, height=2, font=ft_middle)  
        self.b_next.grid(row=30, column=5, columnspan=3)  
          
        
        # 3. 文本框 (用于显示简历文本)
        self.resumeText = Text(frame, font=ft_middle, width=50, height=40)
        self.resumeText.grid(row=0, column=10, rowspan=50)        
        
        # 4. 状态显示框 
        self.condition = StringVar()
        self.conditionLabel = Label(frame, textvariable=self.condition, font=ft_middle, fg="blue")
        self.conditionLabel.grid(row=31, column=0, columnspan=10)
        self.condition.set(' ')
        
        # 获取断点：(文件名：表格行数)，程序意外退出时，能够在上一次标注的地方继续
        self.whitelist = whitelist
        self.log_file = open('./link/log2.txt', 'r+')
        breakpoint = self.log_file.readline()
        if len(sys.argv)>1 and sys.argv[1] == 'from-scratch':
            breakpoint = ''
            print('fetch data from scratch...')
        if breakpoint == '':
            breakpoint = self.whitelist[0] + ':-1'
        self.breakfile = breakpoint.split(':')[0]
        self.breakline = int(breakpoint.split(':')[1])
        print(self.breakfile, self.breakline)
        
        self.df = pd.read_excel('./link/' + self.breakfile)
        self.next()
        self.previous()
        
    ''' 一些Callback函数 '''       
    # 读取所有下拉框的值, 与上一次做比较, 计算出这次点击的值   
    def select(self, event):
        var_s = []
        for s in self.string_vars:
            if s.get() != '':
                var_s.append(s)
                
        if len(var_s) == 1: 
            self.selected = var_s[0].get()
        elif len(var_s) == 2: 
            if var_s[0].get() == self.selected:                   
                var_s[0].set('')
                self.selected = var_s[1].get()
            else:
                var_s[1].set('')
                self.selected = var_s[0].get()
        #print(self.selected)
        
        for key in self.category:
            sub_category = self.category[key].keys()        
            for c in sub_category:
                for subject in self.category[key][c]:
                    #print(subject)
                    if subject == self.selected:
                        label= ('%s-%s-%s' % (key, c, subject)).replace(' ', '')
                        self.subject_label = label
    
    def delete(self, event):
        for s in self.string_vars:
            s.set('')
            
    # 点击按钮 <跳转至上一条> 的回调函数
    def previous(self):
        is_over = 1
        for i in range(self.breakline-1, -1, -1):
            file_name = './text/original/' + self.df['学校'][i] + str(i) + '.txt'
            if not os.path.exists(file_name):
                continue
            # 刷新文本  
            with open(file_name, 'r') as f:
                self.text = f.read()
                text = self.text.split('\n')
                dis_text = ''
                # 不显示字数小于5的行
                for t in text:
                    if len(t) > 5:
                        dis_text += t + '\n'
                self.resumeText.delete(1.0, "end")
                self.resumeText.insert("end", dis_text)
            # 更新并保存断点
            self.breakline = i
            is_over = 0
            break
        
        if is_over and self.breakfile != self.whitelist[0]:
            self.breakfile = self.whitelist[self.whitelist.index(self.breakfile) - 1]
            self.df = pd.read_excel('./link/' + self.breakfile)    
            self.breakline = len(self.df) - 1
            
        self.log_file.seek(0)
        self.log_file.truncate()
        self.log_file.write(self.breakfile + ':' + str(self.breakline)) 
        print(self.breakfile + ':' + str(self.breakline))
        self.condition.set('%s %s --> %s'%(self.breakfile + ':' + str(self.breakline), self.df.loc[self.breakline]['姓名'], self.df.loc[self.breakline]['标签']))
         
    # 点击按钮 <标注并跳转至下一条> 的回调函数    
    def next(self):
        # 恢复文件断点
        is_over = 1
        # 恢复行数断点
        for i in range(self.breakline, len(self.df)):
            if i <= self.breakline:
                continue
            file_name = './text/original/' + self.df['学校'][i] + str(i) + '.txt'
            if not os.path.exists(file_name):
                continue
            # 保存标注结果
            print(self.subject_label)
            #self.df['标签'][self.breakline] = self.subject_label
            if self.breakline >= 0:
                self.df.loc[self.breakline, '标签'] = self.subject_label
                self.df.to_excel('./link/' + self.breakfile, index=False)  
            # 刷新文本  
            with open(file_name, 'r') as f:
                self.text = f.read()
                text = self.text.split('\n')
                dis_text = ''
                # 不显示字数小于5的行
                for t in text:
                    if len(t) > 5:
                        dis_text += t + '\n'
                self.resumeText.delete(1.0, "end")
                self.resumeText.insert("end", dis_text)
            self.breakline = i
            is_over = 0
            break
        # 显示信息
        print(self.breakfile + ':' + str(self.breakline))
        self.condition.set('%s %s --> %s'%(self.breakfile + ':' + str(self.breakline), self.df.loc[self.breakline]['姓名'], self.df.loc[self.breakline]['标签']))
        # 更新并保存断点
        if is_over and self.breakfile != self.whitelist[-1]:
            self.breakfile = self.whitelist[self.whitelist.index(self.breakfile) + 1]
            self.breakline = -1
            self.df = pd.read_excel('./link/' + self.breakfile)
        self.log_file.seek(0)
        self.log_file.truncate()
        self.log_file.write(self.breakfile + ':' + str(self.breakline)) 
        
                
    # 点击按钮 <跳过> 时的回调函数        
    def skip(self):
        is_over = 1
        for i in range(self.breakline+1, len(self.df)):
            file_name = './text/original/' + self.df['学校'][i] + str(i) + '.txt'
            if not os.path.exists(file_name):
                continue
            # 刷新文本  
            #print(file_name)
            with open(file_name, 'r') as f:
                self.text = f.read()
                text = self.text.split('\n')
                dis_text = ''
                # 不显示字数小于5的行
                for t in text:
                    if len(t) > 5:
                        dis_text += t + '\n'
                self.resumeText.delete(1.0, "end")
                self.resumeText.insert("end", dis_text)
            self.breakline = i
            is_over = 0
            break
        # 显示信息
        print(self.breakfile + ':' + str(self.breakline))
        self.condition.set('%s %s --> %s'%(self.breakfile + ':' + str(self.breakline), self.df.loc[self.breakline]['姓名'], self.df.loc[self.breakline]['标签']))    
        # 更新并保存断点
        if is_over and self.breakfile != self.whitelist[-1]:
            self.breakfile = self.whitelist[self.whitelist.index(self.breakfile) + 1]
            self.breakline = -1
            self.df = pd.read_excel('./link/' + self.breakfile)
        self.log_file.seek(0)
        self.log_file.truncate()
        self.log_file.write(self.breakfile + ':' + str(self.breakline)) 
        
          
win = Tk()  
win.title('专家简历标注工具')
app = App(win)  
win.geometry('1600x920')
win.mainloop() 


    
    
    
