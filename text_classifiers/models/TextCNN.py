# -*- coding: utf-8 -*-
# @DateTime :2021/4/14
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import torch
from torch import nn
from torch.nn import functional as F
from text_classifiers.basics import AbstractModel


class TextCNN(nn.Module, AbstractModel):
    r'''
        TextCNN
            1. 读入的数据 [bs, seq]
            2. 经embedding后变为[bs, seq, dim]
            3. 对第一维缩放，变为 [bs, 1, seq, dim]
            4. 经多组CNN提取特征  [bs, num, seq - ks, 1] * N , 并进行dim=3维压缩
            5. 对 dim=2 维进行最大池化，获取到 [bs, num] * N
            6. 线性变换输出 [bs, out]
    '''

    def __init__(self, vocab_size, output_size, embedding_dim=300, kernel_sizes=[2, 3, 4], num_kernels=100, dropout=0.1):
        super(TextCNN, self).__init__()
        self.config_keys = ['vocab_size', 'output_size', 'embedding_dim', 'kernel_sizes', 'num_kernels', 'dropout']
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.dropout = dropout

        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        # 扩展 channel 维度，用 高度=ks，宽度=embedding_dim的 kernel_size去扫描文本矩阵[bs, 1, seq, emb_dim] => [bs, num, seq - ks, 1]
        self.cnns = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=(ks, embedding_dim)) for ks in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=len(kernel_sizes) * num_kernels, out_features=output_size)

    def forward(self, x):
        r'''
            x: [bs, seq]
        '''
        x = self.embeddings(x)  # [bs, seq, emb]
        x = self.dropout(x)
        x = x.unsqueeze(dim=1)
        xs = [F.relu(conv(x)).squeeze() for conv in self.cnns]  # [[bs, num, seq - ks], * len(ks)]
        xs = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in xs]  # 对 dim=2 维进行最大池化，获取到 [[bs, num], * len(ks)]
        x = torch.cat(xs, dim=1)  # [bs, num * len(ks)]
        x = self.dropout(x)
        logit = self.fc(x)  # 映射输出
        return logit

    def get_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=1e-3)

    @staticmethod
    def load(input_path: str):
        config_path = os.path.join(input_path, 'config.json')
        model_path = os.path.join(input_path, 'pytorch_model.bin')

        config = TextCNN.json_load(config_path)
        model = TextCNN(*config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model
