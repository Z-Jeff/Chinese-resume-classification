# Chinese-resume-classification
本项目目的：对高校教师的简历做分类，预测其专业领域。

所使用方法：Word2Vec预训练词向量模型 + TextCNN文本分类模型。

本项目的代码实现完全通过python实现，其中Word2Vec使用gensim包实现，TextCNN的pytorch实现参考https://github.com/Shawn1993/cnn-text-classification-pytorch。

项目介绍博客：https://blog.csdn.net/Jeff_zjf/article/details/107535329

# 环境安装
pip install gensim

pip install torch torchtext # 不同平台上安装脚本可能不同，需查询https://pytorch.org/

pip install word2vec

pip install xlrd>=0.9.0

pip install bs4

# 下载数据模型
百度云盘：https://pan.baidu.com/s/1Z88bmTh14eptERjmY7DmDg，提取码：4kwi

下载完成后，把 snapshot.zip 解压到 Chinese-resume-classification/ 下

把 data_.zip 解压到 Chinese-resume-classification/data/ 下

把 model.zip 解压到 Chinese-resume-classification/word2Vec/ 下


# 预测专业类别
直接通过简历链接，自动爬取文本，预测其专业类别。

可以以脚本形式运行：

python main.py -use-word2vec -predict-url https://baike.baidu.com/item/%E4%BD%95%E6%81%BA%E6%98%8E/22863446?fr=aladdin

这里的链接可以替换成你想要预测的简历链接。

或者使用GUI界面：

python demo.py

# 模型训练
python main.py -use-word2vec
