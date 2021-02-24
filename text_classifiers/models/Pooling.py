# -*- coding: utf-8 -*-
# @DateTime :2021/2/18 下午2:53
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os, json
from typing import List, Tuple, Dict, Union

import torch
from torch import nn, Tensor
from text_classifiers.tools import tools


class Pooling(nn.Module):
    r'''Performs pooling (max or mean) on the token embeddings.
    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: 词向量维度
    :param pooling_mode_cls_token: 使用[CLS]代表文本的语义
    :param pooling_mode_max_tokens: 使用 max-pooling 代表文本的语义
    :param pooling_mode_mean_tokens: 使用 mean-pooling 代表文本的语义
    :param pooling_mode_mean_sqrt_len_tokens: 使用 mean-pooling/text_length 代表文本的语义
    '''

    def __init__(self, word_embedding_dimension: int, pooling_mode_cls_token: bool = False, pooling_mode_max_tokens: bool = False, pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False):
        super(Pooling, self).__init__()
        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_max_tokens', 'pooling_mode_mean_tokens', 'pooling_mode_mean_sqrt_len_tokens']
        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        # Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set Padding token to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # if tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features.update({'text_embeddings': output_vector})
        return features

    @property
    def dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        saved_config_path = os.path.join(output_path, 'config.json')
        tools.json_save(self.get_config_dict(), saved_config_path, indent=2)

    @staticmethod
    def load(input_path: str):
        saved_config_path = os.path.join(input_path, 'config.json')
        config = tools.json_load(saved_config_path)
        return Pooling(**config)
