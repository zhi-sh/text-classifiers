# -*- coding: utf-8 -*-
# @DateTime :2021/2/24 下午3:18
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
from typing import Dict

import torch
from torch import nn, Tensor
from text_classifiers.models import AbstractModel
from text_classifiers.tools import tools, utils


class Dense(nn.Module, AbstractModel):
    r'''全连接网络'''

    def __init__(self, in_features: int, out_features: int, bias: bool = True, activation_function=nn.Tanh(), init_weight: Tensor = None, init_bias: Tensor = None):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)
        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)

    def forward(self, features: Dict[str, Tensor]):
        text_embeddings = self.activation_function(self.linear(features['text_embeddings']))
        features.update({'text_embeddings': text_embeddings})
        return features

    @property
    def dimension(self):
        return self.out_features

    def save(self, output_path):
        saved_config_path = os.path.join(output_path, 'config.json')
        saved_model_path = os.path.join(output_path, 'pytorch_model.bin')

        config = {'in_features': self.in_features, 'out_features': self.out_features, 'bias': self.bias, 'activation_function': utils.fullname(self.activation_function)}
        tools.json_save(config, saved_config_path, indent=2)
        torch.save(self.state_dict(), saved_model_path)

    @staticmethod
    def load(input_path):
        saved_config_path = os.path.join(input_path, 'config.json')
        saved_model_path = os.path.join(input_path, 'pytorch_model.bin')

        config = tools.json_load(saved_config_path)
        config['activation_function'] = utils.import_from_string(config['activation_function'])()
        model = Dense(**config)
        model.load_state_dict(torch.load(saved_model_path, map_location=torch.device('cpu')))
        return model
