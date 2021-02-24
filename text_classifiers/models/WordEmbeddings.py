# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午5:15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging, os
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from text_classifiers.tools import utils, tools
from text_classifiers.models import AbstractModel
from text_classifiers.models.tokenizer import AbstractTokenizer, CharTokenizer

logger = logging.getLogger(__name__)


class WordEmbeddings(nn.Module, AbstractModel):
    r'''
    该模块必须基于一个已训练的词向量进行初始化，因为需要自动确认Embedding的维度。故请使用WordEmbeddings.from_text_file(w2v_file_path)来创建对象
    '''

    def __init__(self, tokenizer: AbstractTokenizer, embedding_weights, update_embeddings: bool = False, max_seq_length: int = 1000000):
        super(WordEmbeddings, self).__init__()
        if isinstance(embedding_weights, list):
            embedding_weights = np.asarray(embedding_weights)
        if isinstance(embedding_weights, np.ndarray):
            embedding_weights = torch.from_numpy(embedding_weights)

        num_embeddings, embeddings_dimension = embedding_weights.size() if embedding_weights is not None else len(tokenizer.get_vocab()), 300
        self.emb_layer = nn.Embedding(num_embeddings, embeddings_dimension, padding_idx=tokenizer.get_padding_idx())
        if embedding_weights is not None:
            self.emb_layer.load_state_dict({'weight': embedding_weights})
        self.emb_layer.weight.requires_grad = update_embeddings

        self.embedding_dimension = embeddings_dimension
        self.tokenizer = tokenizer
        self.update_embeddings = update_embeddings
        self.max_seq_length = max_seq_length

    def forward(self, features):
        token_embeddings = self.emb_layer(features['input_ids'])
        cls_tokens = None
        features.update({'token_embeddings': token_embeddings, 'cls_token_embeddings': cls_tokens})
        return features

    def tokenize(self, texts: List[str]):
        tokenizerd_texts = [self.tokenizer.tokenize(text) for text in texts]
        sentence_lengths = [len(tokens) for tokens in tokenizerd_texts]
        max_len = max(sentence_lengths)

        # 填充至统一的最大长度
        input_ids = []
        attention_masks = []
        for tokens in tokenizerd_texts:
            padding = [0] * (max_len - len(tokens))
            input_ids.append(tokens + padding)
            attention_masks.append([1] * len(tokens) + padding)

        # 构造feature字典
        features = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'sentence_lengths': torch.tensor(sentence_lengths, dtype=torch.long)
        }

        return features

    @property
    def dimension(self) -> int:
        return self.embedding_dimension

    def get_config_dict(self):
        return {'tokenizer_class': utils.fullname(self.tokenizer), 'update_embeddings': self.update_embeddings, 'max_seq_length': self.max_seq_length}

    def save(self, output_path: str):
        saved_config_path = os.path.join(output_path, 'config.json')
        saved_model_path = os.path.join(output_path, 'pytorch_model.bin')

        tools.json_save(self.get_config_dict(), saved_config_path, indent=2)
        torch.save(self.state_dict(), saved_model_path)
        self.tokenizer.save(output_path)

    @staticmethod
    def load(input_path: str):
        saved_config_path = os.path.join(input_path, 'config.json')
        saved_model_path = os.path.join(input_path, 'pytorch_model.bin')

        config = tools.json_load(saved_config_path)
        tokenizer_class = config['tokenizer_class']
        tokenizer = tokenizer_class.load(input_path)
        weights = torch.load(saved_model_path, map_location=torch.device('cpu'))
        embedding_weights = weights['emb_layer.weight']
        model = WordEmbeddings(tokenizer=tokenizer, embedding_weights=embedding_weights, update_embeddings=config['update_embeddings'])
        return model

    @staticmethod
    def from_file(raw_text_file: str, w2v_file_path: str = None, max_vocab_size: int = 100000, update_embeddings: bool = False, w2v_separator: str = ' ', tokenizer=CharTokenizer()):
        r'''
        :param raw_text_file: 统计字典的原始语料文件
        :param w2v_file_path: 预训练词向量文件路径，如果不提供，则使用随机变量
        :param max_vocab_size: 模型能接受的字词容量
        :param update_embeddings: 是否更新词向量，默认不更新
        :param w2v_separator: 词向量文件的分隔符
        :param tokenizer: 分词器，用于构建WordEmbeddings对象
        :return: WordEmbeddings对象
        '''

        assert os.path.exists(raw_text_file)  # 确保原始语料存在
        vocab = tokenizer.vocab_from_file(raw_text_file, max_vocab_size)  # 经原始语料生成vocab
        tokenizer.set_vocab(vocab)

        if w2v_file_path is None:  # 没有提供预训练的词向量文件，使用随机变量
            return WordEmbeddings(tokenizer=tokenizer, embedding_weights=None, update_embeddings=True)
        else:
            logger.info(f'generating WordEmbeddings object from word2vec embeddings file: {w2v_file_path}')
