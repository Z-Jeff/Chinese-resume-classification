import os
import sys
import time
import pandas as pd
from tkinter import *  
from tkinter import ttk
import tkinter.font as tkFont
import chardet
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import data.textProcess as preprocess

import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
from tqdm import tqdm 
import pandas as pd
from gensim.models import word2vec
import argparse

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
parser.add_argument('-word2vec-model', type=str, default='./word2Vec/model/word2Vec.model', help='filename of word2vec model')
parser.add_argument('-use-word2vec', action='store_true', default=True, help='whether use word2vec or not')

parser.add_argument('-model', type=str, default='CNN', help='model type: CNN, LSTM, GRU')
parser.add_argument('-hidden-size', type=int, default=128, help='hidden layer size of LSTM')
parser.add_argument('-lstm-num', type=int, default=1, help='the number of LSTM')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=True, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

class App:  
    def __init__(self, master): 
        ''' GUI布局 '''
        frame = Frame(master)  
        frame.pack()
        
        self.frame = frame
        self.text = ''
        self.isPreprocess = 0
        
        # 全局
        self.frame.bind("<Button-3>", self.delete)
        
        # 字体
        ft = tkFont.Font(family='Fixdsys', size=13, weight=tkFont.BOLD)
        ft_big = tkFont.Font(family='Fixdsys', size=21, weight=tkFont.BOLD)
        ft_middle = tkFont.Font(family='Fixdsys', size=18, weight=tkFont.BOLD)
        
        # 文本框  <请输入简历链接：>
        self.label1 = Label(frame, text='请输入简历链接：', font=ft, pady=5)
        #self.label1.pack(side=TOP, anchor='w')
        self.label1.grid(row=0, column=0)
        
        # 输入框 (简历链接)
        self.url = StringVar()
        self.urlEntry = Entry(frame, textvariable=self.url, font=ft, width=45)
        self.urlEntry.bind("<Return>", self.input_url)
        self.urlEntry.bind("<Button-3>", self.delete)
        #self.urlEntry.pack(side=TOP, anchor='w')
        self.urlEntry.grid(row=1, column=0)
        
        # 文本框  <或输入简历文本：>
        self.label2 = Label(frame, text='或输入简历文本：', font=ft, pady=5)
        #self.label2.pack(side=TOP, anchor='w')
        self.label2.grid(row=2, column=0)
        
        # 输入框 (简历文本)
        self.resumeText = Text(frame, width=50, height=30, font=ft)
        self.resumeText.bind("<Button-3>", self.delete)
        self.resumeText.bind("<Return>", self.input_text)
        #self.entryText.pack(side=TOP, anchor='w')
        self.resumeText.grid(row=3, column=0)
        
        # 按钮  [开始预测]
        self.predictButton = Button(frame, text="开始预测", fg="red", command=self.predict, font=ft_big)  
        #self.next.pack(side=RIGHT)
        self.predictButton.grid(row=3, column=1)
        
        # 文本框  < --> >
        self.label5 = Label(frame, text="-> ", fg="blue", font=ft_big)  
        #self.next.pack(side=RIGHT)
        self.label5.grid(row=3, column=2)
        
        # 文本框  (预测结果）
        self.result = StringVar()
        self.label4 = Label(frame, textvariable=self.result, bg='yellow', font=ft_middle)
        self.result.set('     ')
        #self.label4.pack(side=RIGHT)
        self.label4.grid(row=3, column=3)
        
        # 文本  (状态显示)
        self.condition = StringVar()
        self.label6 = Label(frame, textvariable=self.condition, fg="green", font=ft_big)  
        self.condition.set('空闲中.')
        #self.next.pack(side=RIGHT)
        self.label6.grid(row=2, column=1)
        
        ''' 模型载入 '''
        excel = './data/标注数据.xlsx'
        text_path = './data/text/processed/'
        args.snapshot = './snapshot/TextCNN_Word2Vec/best_steps_4100.pt'
        #new_model = word2vec.Word2Vec.load(args.word2vec_model)
        
        print("\nLoading data...")
        tokenize = lambda x: re.split(' ', x)
        text_field = data.Field(tokenize=tokenize, lower=True,  stop_words=['\r', '\n', '\t', '\xa0', ' ', ''])
        label_field = data.Field(sequential=False)
        train_iter, dev_iter = mydatasets.resume(excel, text_path, text_field, label_field, 
                                                 args.batch_size, device=-1, repeat=False,
                                                 use_wv=True, 
                                                 wv_model=args.word2vec_model,
                                                 )
        
        args.embed_num = len(text_field.vocab) # 9499
        args.class_num = len(label_field.vocab) - 1 # 16, -1是为了除去<unk>
        args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
        args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
        #args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        
        if args.model == 'CNN':
            print('Using CNN model')
            cnn = model.CNN_Text(args, text_field)
        elif args.model == 'LSTM':
            print('Using LSTM model')
            cnn = model.LSTM_Text(args, text_field)
        
        if args.snapshot is not None:
            print('\nLoading model from {}...'.format(args.snapshot))
            cnn.load_state_dict(torch.load(args.snapshot))

        if args.cuda:
            #torch.cuda.set_device(args.device)
            print('Using CUDA')
            cnn = cnn.cuda()
        else:
            print('Using CPU')
        
        self.cnn = cnn
        self.text_field = text_field
        self.label_field = label_field
        
    def input_url(self, event=None):   
        url =  self.urlEntry.get()
        
        html = urlopen(url).read()
        encoding = chardet.detect(html)['encoding']
        if encoding == 'GB2312':
            encoding = 'gbk'
        elif encoding == 'iso-8859-1':
            encoding = 'utf-8'
            
        html = bs(html, 'html.parser', from_encoding=encoding)
        [script.extract() for script in html.findAll('script')]
        [style.extract() for style in html.findAll('style')]
        
        text = ''.join([s for s in html.text.splitlines(True) if s.strip()])
        self.resumeText.delete(1.0, END)
        self.resumeText.insert(1.0, text)
                
        self.text = preprocess.textPreprocess(text)
        self.isPreprocess = 1
        #print(self.text)
        
    def input_text(self, event=None):
        text = self.resumeText.get(0.0, END)
        self.text = preprocess.textPreprocess(text)
        self.isPreprocess = 1
        #print(text)
        
    def delete(self, event):
        self.urlEntry.delete(0, END)
        self.resumeText.delete(1.0, END)
        self.condition.set('空闲中.')
        self.result.set('     ')
        self.text = ''
              
    def predict(self):
        if not self.isPreprocess:
            # 首先判断简历文本是否有内容
            if len(self.urlEntry.get()) != 0:
                self.input_text()
            # 再判断简历链接是否有内容
            elif len(self.urlEntry.get()) != 0:
                self.input_url()
            else:
                self.condition.set('无内容！')
                return 
                                    
        self.condition.set('预测中.')
        label = train.predict(self.text, self.cnn, self.text_field, self.label_field, args.cuda)
        self.isPreprocess = 0
        self.condition.set('预测完成')
        
        self.result.set(label)
        print('label:', label)
        
            
        '''
        # 按钮
        self.next = Button(frame, text="下一条", fg="blue", command=self.next)  
        self.next.pack(side=BOTTOM)
            
        # 下拉框
        self.string2 = StringVar()
        self.dropdown = ttk.Combobox(self.frame, textvariable=self.string2, values=[''], state='readonly')
        self.dropdown['value'] = [' ']
        self.dropdown.pack(side=BOTTOM)
        
        # 单选框
        for i in range(len(self.category)):
                Radiobutton(frame, variable=self.v1, text=self.category[i], value=i, indicatoron=0, command=self.radio1).pack(side=BOTTOM)
        
        # 按钮
        self.skip = Button(frame, text="跳过", fg="red", command=self.skip)  
        self.previous = Button(frame, text="上一条", fg="blue", command=self.previous)  
        self.previous.pack(side=TOP)
        self.skip.pack(side=RIGHT)
        
        # 文本框
        self.string = StringVar()
        self.label1 = Label(frame, textvariable=self.string)
        self.label1.pack()
        '''
    '''    
    def radio1(self):
        sub_category = self.sub_category[self.v1.get()].split('\n')
        self.dropdown['value'] = sub_category
        self.dropdown.pack(side=BOTTOM)
    
    def skip(self):
        self.idx += 1   
        while str(self.df.loc[self.idx]['个人主页']) == 'nan':
            self.idx += 1
            if self.idx >= self.rows - 1:
                break
        self.string.set(str(self.df.loc[self.idx]))
        self.driver.get(self.df.loc[self.idx]['个人主页'])
        self.idx_file.seek(0)
        self.idx_file.truncate()
        self.idx_file.write(str(self.idx))
                       
    def next(self):
        # 刷新下一条
        self.idx += 1
        while str(self.df.loc[self.idx]['个人主页']) == 'nan':
            self.idx += 1
            if self.idx >= self.rows - 1:
                break
        self.string.set(str(self.df.loc[self.idx]))
        self.driver.get(self.df.loc[self.idx]['个人主页'])
        self.idx_file.seek(0)
        self.idx_file.truncate()
        self.idx_file.write(str(self.idx))
        
        # 标注   
        category = self.category[self.v1.get()]
        sub_category = self.string2.get()
        label = category + '-' + sub_category  
        if not label == '医学-':
            print(label)
            self.df['标签'][self.idx-1] = label
            self.df.to_excel(self.f1) 
        
    def previous(self):
        # 刷新上一条
        self.idx -= 1
        while str(self.df.loc[self.idx]['个人主页']) == 'nan':
            self.idx -= 1
            if self.idx <= 0:
                break
        self.string.set(str(self.df.loc[self.idx]))
        self.driver.get(self.df.loc[self.idx]['个人主页'])
        self.idx_file.seek(0)
        self.idx_file.truncate()
        self.idx_file.write(str(self.idx))
        
        
    def __del__(self):
        a = 1
        #if hasattr(self, 'driver'):
        #    self.driver.close()
    '''    
win = Tk()  
win.title('专家专业领域预测程序')
app = App(win)  
win.geometry('1050x800')
win.mainloop() 


