# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午5:15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging, os, json
import numpy as np
from tqdm import tqdm
from torch import nn
from text_classifier.models import AbstractModel
from text_classifier.models.tokenizer import AbstractTokenizer, CharTokenizer

logger = logging.getLogger(__name__)


class WordEmbeddings(nn.Module, AbstractModel):
    r'''
    该模块必须基于一个已训练的词向量进行初始化，因为需要自动确认Embedding的维度。故请使用WordEmbeddings.from_text_file(w2v_file_path)来创建对象
    '''

    def __init__(self):
        super(WordEmbeddings, self).__init__()
        pass

    @staticmethod
    def from_text_file(w2v_file_path: str, max_vocab_size: int, update_embeddings: bool = False, w2v_separator: str = ' ', tokenizer=CharTokenizer()):
        r'''
        :param w2v_file_path: 预训练词向量文件路径
        :param max_vocab_size: 模型能接受的字词容量
        :param update_embeddings: 是否更新词向量，默认不更新
        :param w2v_separator: 词向量文件的分隔符
        :param tokenizer: 分词器，用于构建WordEmbeddings对象
        :return: WordEmbeddings对象
        '''

        assert os.path.exists(w2v_file_path)
        logger.info(f'generating WordEmbeddings object from word2vec embeddings file: {w2v_file_path}')

        vocab = []
        embeddings = []
        embeddings_dimension = None
        with open(w2v_file_path, encoding='utf-8') as fin:
            for line in tqdm(fin, desc='load word embeddings'):
                split = line.rstrip().split(w2v_separator)
                if len(split) < 5:  # 跳过某些w2v文件的第一行
                    continue

                # 添加<PAD>填充字符，默认index=0
                if embeddings_dimension is None:
                    embeddings_dimension = len(split) - 1
                    vocab.append('<PAD>')
                    embeddings.append(np.zeros(embeddings_dimension))

                # 解析每一行
                if len(split) - 1 != embeddings_dimension:
                    logger.error("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
                    continue

                word = split[0]
                vector = np.array([float(num) for num in split[1:]])
                vocab.append(word)
                embeddings.append(vector)

                # 判断是否超出模型容量
                if len(vocab) > max_vocab_size:
                    break

        embeddings = np.asarray(embeddings)
        tokenizer.set_vocab(vocab)
        return WordEmbeddings(tokenizer=tokenizer, embedding_weights=embeddings, update_embeddings=update_embeddings)
