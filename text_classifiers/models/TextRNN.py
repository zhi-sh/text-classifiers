# -*- coding: utf-8 -*-
# @DateTime :2021/4/15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import torch
from torch import nn
from text_classifiers.basics import AbstractModel


class TextRNN(nn.Module, AbstractModel):
    r'''
        TextRNN
            1. 输入数据 [bs, seq],
            2. 经embedding后变为[bs, seq, dim]
            3. 经lstm层变为 [bs, seq, hid * bidir], 后续只取序列最后时间步 [bs, hid * bidir]
            4. 经线性变换输出 [bs, out]
    '''

    def __init__(self, vocab_size, output_size, embedding_dim=300, hidden_size=256, num_layers=2, dropout=0.1, bidirectional=True):
        super(TextRNN, self).__init__()
        self.config_keys = ['vocab_size', 'output_size', 'embedding_dim', 'hidden_size', 'num_layers', 'bidirectional']
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        r'''
            x: [bs, seq]
        '''
        x = self.embedding(x)  # [bs, seq, emb]
        x = self.dropout(x)
        hiddens, _ = self.rnn(x)  # [bs, seq, hid * 2]
        x = hiddens[:, -1, :]  # [bs, hid * 2]
        logit = self.fc(x)  # [bs, out]
        return logit

    def get_optimizer(self, parameters):
        return torch.optim.Adam(parameters, lr=1e-3)

    @staticmethod
    def load(input_path: str):
        config_path = os.path.join(input_path, 'config.json')
        model_path = os.path.join(input_path, 'pytorch_model.bin')

        config = TextRNN.json_load(config_path)
        model = TextRNN(*config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model
