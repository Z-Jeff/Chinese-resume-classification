import gensim
from gensim.models import word2vec
import time 
import os

word2vec_path='./model/word2Vec.model'
segment_path='../data/train_text/'

train = 0
if train:  
    start_time = time.time()
    # 载入
    # [1]若只有一个文件，使用LineSentence读取文件
    # sentences = word2vec.LineSentence(segment_path)
    # [2]若存在多文件，使用PathLineSentences读取文件列表
    sentences = word2vec.PathLineSentences(segment_path)
 
    # 训练
    print('Training start...')
    model = word2vec.Word2Vec(sentences, hs=1, min_count=3, window=5, size=128)
    
    # 保存
    model.wv.save_word2vec_format(word2vec_path)
    
    # 提示
    print('Training end.')
    end_time = time.time()
    now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    log = 'Time: %s \nUse file: %s \nSave model: %s \nConsuming time: %d minute. \n\n' % (now_time, segment_path, word2vec_path, (end_time-start_time)/60.0)
    f = open('./log/train_word2vec.txt', 'a')
    f.write(log)
    print(log)
    
else:
    # 加载
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path)
    
    # 查看两个词之间的余弦相似性
    print(model.wv.similarity('电子', '计算机'))
    
    # 查看一个词的相似词
    print(model.wv.similar_by_word('计算机'))
    print(model.wv.similar_by_word('化学'))

