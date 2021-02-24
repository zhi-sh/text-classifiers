# -*- coding: utf-8 -*-
# @DateTime :2021/2/24 下午3:15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from torch import nn, Tensor
from text_classifiers import TextClassifier


class SoftmaxLoss(nn.Module):
    def __init__(self, model: TextClassifier):
        super().__init__()
        self.model = model
        self.device = model.device
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, features: Tensor, labels: Tensor):
        output = self.model(features)['text_embeddings']  # 只有一个句子
        loss = self.loss_fct(output, labels.view(-1))
        return loss
