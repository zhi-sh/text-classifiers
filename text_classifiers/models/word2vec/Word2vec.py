# -*- coding: utf-8 -*-
# @DateTime :2021/2/25 下午5:43
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
from gensim.models import word2vec


class Word2vec:
    def __init__(self, dimension=300):
        self.dimension = dimension

    def train_word2vec(self, train_file_path, **kwargs):
        r'''训练好的词向量，存至训练文件所在的同目录下'''
        data_path, fullname = os.path.split(train_file_path)
        name, _ = os.path.split(fullname)
        embedding_path = os.path.join(data_path, f'token_vec.char')

        sentenses = word2vec.Text8Corpus(train_file_path)  # 加载分词语料
        model = word2vec.Word2Vec(sentenses, size=self.dimension, **kwargs)  # 训练skip-gram模型,默认window=5
        model.wv.save_word2vec_format(embedding_path, binary=False)
