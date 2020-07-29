import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from gensim.models.keyedvectors import KeyedVectors
#import word2vec
#from gensim.models import word2vec

torch.manual_seed(1)


class CNN_Text(nn.Module):
    
    def __init__(self, args, text_field):
        super(CNN_Text, self).__init__()
        self.args = args
        
        #new_model = word2vec.Word2Vec.load(args.word2vec_model)
        
        V = args.embed_num # 词汇个数，61382，统计得到
        D = args.embed_dim # 词向量维度，128，人为设定
        #V = text_field.vocab.vectors.shape[0] # 85362
        #D = text_field.vocab.vectors.shape[1] # 128
        #V = 34265
        #D = 128
        #print(V, D, type(V), type(D))
        
        C = args.class_num # 分类个数
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        
        if args.use_word2vec: # 使用预训练模型
            self.embed = nn.Embedding(V, D).from_pretrained(text_field.vocab.vectors) 
        else: # 随机初始化
            self.embed = nn.Embedding(V, D) 
        #self.embed = nn.Embedding(V, D).from_pretrained(text_field.vocab.vectors) # 使用预训练模型
        #self.embed.weight.data = torch.Tensor(new_model.vectors).clone()
        #self.embed.weight.data = torch.Tensor(wv_model.wv.vectors).clone()
        
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)  # 去掉第四维
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        #print(self.embed.weight.sum())
        #i = torch.tensor(1)
        #i = i.cuda()
        
        #print(dir(self.embed))
        #import os
        #os._exit(1)
        #print(x.shape[0], x.shape[1])
        
        x = self.embed(x).view(x.shape[0], x.shape[1], -1)  # (N, W, D)  N应该为batch，W为词的个数，D为词向量维度
        #print(x.shape) # [64, 44, 128]
        
        if self.args.static: # 静态了，就不允许反向传播时修改embed了
            x = Variable(x)  # 如果运行了这一行，embed不会被反向传播改变
        
        x = x.unsqueeze(1)  # (N, Ci, W, D) # Ci是输入通道个数，这里为1

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
        
        
class LSTM_Text(nn.Module):
    
    def __init__(self, args, text_field):
        super(LSTM_Text, self).__init__()
        self.args = args
        
        #V = args.embed_num # 词汇个数，61382，统计得到
        #D = args.embed_dim # 词向量维度，128，人为设定
        V = text_field.vocab.vectors.shape[0] # 85362
        D = text_field.vocab.vectors.shape[1] # 128

        C = args.class_num # 分类个数
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D) # 随机初始化
        self.embed = nn.Embedding(V, D).from_pretrained(text_field.vocab.vectors) # 使用预训练模型
        
        # LSTM
        # 如果num_layers = 1, 则dropout = 0。只有num_layers > 1, dropout > 0
        self.lstm = nn.LSTM( input_size=D, hidden_size=args.hidden_size, num_layers=args.lstm_num, dropout=0,
                            batch_first=True, bidirectional=True)
        
        self.bn2 = nn.BatchNorm1d(args.hidden_size*2)
        self.fc = nn.Linear(args.hidden_size*2, C)

    def forward(self, x):
        x = self.embed(x).view(x.shape[0], x.shape[1], -1)  # (N, W, D)  N应该为batch，W为词的个数，D为词向量维度
        #print(x.shape) # [64, 44, 128]
        
        if self.args.static: # 静态了，就不允许反向传播时修改embed了
            x = Variable(x)  # 如果运行了这一行，embed不会被方向传播改变
        
        #x = x.unsqueeze(1)  # (N, Ci, W, D) # Ci是输入通道个数，这里为1
        output, ht = self.lstm(x, None) # (N, C)
        print(output.shape, ht.shape)
        return output
        
        
        
        
        
        
