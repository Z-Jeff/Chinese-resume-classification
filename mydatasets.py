import re
import os
import random
import tarfile
import urllib
import pandas as pd
from torchtext import data
from torchtext.vocab import Vectors

class ResumeDataset(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    def __init__(self, text_field, label_field, excel_file, text_path, examples=None):
        fields = [('text', text_field), ('label', label_field)]
        
        if examples is None:
            examples = []
            df = pd.read_excel(excel_file)    
            
            # 遍历标注数据的每一行, 读取标签和文件id
            for i in range(len(df)):
                label = df.loc[i]['标签']
                # 如果没有标签，则跳过该条数据
                if not isinstance(label, str):
                    continue
                label = label.split('-')[-1]
                text_file = text_path + df.loc[i]['文件id'] + '.txt'
                with open(text_file, 'r') as f:
                    text = f.read()
                
                examples += [data.Example.fromlist([text, label], fields)]              
            
        super().__init__(examples, fields)

    @classmethod
    def splits(cls, excel_file, text_path, text_field, label_field, dev_ratio=.1, shuffle=True):
        examples = cls(text_field, label_field, excel_file, text_path).examples # 我觉得examples应该是父类data.Dataset的成员变量
        #print(len(examples)
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, excel_file, text_path, examples=examples[:dev_index]), 
                cls(text_field, label_field, excel_file, text_path, examples=examples[dev_index:])) # 相当于调用了MR类的__init__()方法
                        
def resume(excel_file, text_path, text_field, label_field, batch_size, wv_model=None, use_wv=False, **kargs):
    train_data, dev_data = ResumeDataset.splits(excel_file, text_path, text_field, label_field)
    
    #print(train_data, dir(train_data))
    #print(dir(train_data.examples[0]))
    #print(train_data.examples[0].text)
    #os._exit(1)
    
    #label_field.build_vocab(train_data, dev_data)
    #text_field.build_vocab([['i'], ['and'], ['you']]) # 要用两层嵌套列表才能导入单词，一层的话是导入字母
    if use_wv:
        # 使用预训练的word2vec模型初始化词典
        model_name = wv_model.split('/')[-1]
        path = '/'.join(wv_model.split('/')[:-1])
        vectors = Vectors(name=model_name, cache=path)
        #text_field.build_vocab([wv_model.wv.vocab.keys()]) 
        text_field.build_vocab([vectors.itos], vectors=vectors) # 导入训练好的word2vec模型
        #text_field.vocab.load_vectors(vectors)
    else:
        # 使用数据集初始化词典
        text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter
    
        
        
        
        
        
        
        
        
        
        
        
        
class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = 'rt-polaritydata'

    # 注释掉这个函数，不能正常运行
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None): # 删除掉**kwargs， 运行正常
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        # super(MR, self).__init__(examples, fields, **kwargs) # python2
        super().__init__(examples, fields) # python3 可以省略参数. super()是用来调用父类的一个方法, 这里是调用data.Dataset
        # # 删除掉**kwargs， 运行正常
        
    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path).examples # 我觉得应该是父类data.Dataset的成员变量
        #print(len(examples)
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]), 
                cls(text_field, label_field, examples=examples[dev_index:])) # 相当于调用了MR类的__init__()方法

def mr(text_field, label_field, batch_size, **kargs):
    train_data, dev_data = MR.splits(text_field, label_field)
    
    #print(train_data, dir(train_data))
    #print(dir(train_data.examples[0]))
    #print(train_data.examples[0].text)
    #os._exit(1)
    
    text_field.build_vocab(train_data, dev_data)
    #label_field.build_vocab(train_data, dev_data)
    #text_field.build_vocab([['i'], ['and'], ['you']]) # 要用两层嵌套列表才能导入单词，一层的话是导入字母
    #text_field.build_vocab([word2vec_model.vocab]) # 用这一行代码导入已有的vocab
    label_field.build_vocab(train_data, dev_data)
    
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter
    
    
                

