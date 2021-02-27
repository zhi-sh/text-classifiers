# -*- coding: utf-8 -*-
# @DateTime :2021/2/25 下午3:19
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import torch
from torch import nn
from text_classifiers.tools import tools


class Dropout(nn.Module):
    def __init__(self, dimension: int, p=0.2):
        super(Dropout, self).__init__()
        self.config_keys = ['dimension', 'p']
        self.embedding_dimension = dimension
        self.p = p

        self.dropout = nn.Dropout(p)

    def forward(self, features):
        text_embeddings = self.dropout(features['text_embeddings'])  # [bs, dimension] -> [bs, dimension]
        features.update({'text_embeddings': text_embeddings})
        return features

    @property
    def dimension(self):
        return self.embedding_dimension

    def save(self, output_path: str):
        saved_config_path = os.path.join(output_path, 'config.json')
        tools.json_save(self.get_config_dict(), saved_config_path, indent=2)

    @staticmethod
    def load(input_path: str):
        saved_config_path = os.path.join(input_path, 'config.json')
        config = tools.json_load(saved_config_path)
        return Dropout(**config)
