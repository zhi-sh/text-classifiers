# -*- coding: utf-8 -*-
# @DateTime :2021/2/25 下午3:31
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
from typing import List
import torch
from torch import nn
from text_classifiers.tools import tools
from text_classifiers.models import AbstractModel


class CNN1D(nn.Module, AbstractModel):
    r'''
        CNN1d特征提取器

        输入：[batch, seq_len, dimension]
        输出： [batch, seq_len, out_channels * len(kernel_sizes)]
    '''

    def __init__(self, embedding_dimension: int, out_channels: int = 256, kernel_sizes: List[int] = [1, 3, 5], embedding_name='text_embeddings'):
        super(CNN1D, self).__init__()
        self.config_keys = ['embedding_dimension', 'out_channels', 'kernel_sizes', 'embedding_name']
        self.embedding_dimension = embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.embedding_name = embedding_name

        self.convs = nn.ModuleList()

        in_channels = embedding_dimension
        for kernel_size in kernel_sizes:
            padding_size = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding_size)
            self.convs.append(conv)

    def forward(self, features):
        text_embeddings = features[self.embedding_name]  # [bs, len, dim]
        text_embeddings = text_embeddings.transpose(1, -1)  # [bs, dim, len]
        vectors = [conv(text_embeddings) for conv in self.convs]  # [bs, dim, len] * len(kernel_sizes)
        text_embeddings = torch.cat(vectors, dim=1).transpose(1, -1)  # [bs, len, out_channel*len(kernel_sizes)
        features.update({'text_embeddings': text_embeddings})
        return features

    @property
    def dimension(self):
        return self.out_channels * len(self.kernel_sizes)

    @staticmethod
    def load(input_path: str):
        saved_config_path = os.path.join(input_path, 'config.json')
        saved_model_path = os.path.join(input_path, 'pytorch_model.bin')

        config = tools.json_load(saved_config_path)
        model = CNN1D(**config)
        model.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
        return model
