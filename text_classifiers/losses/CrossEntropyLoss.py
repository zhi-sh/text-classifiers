# -*- coding: utf-8 -*-
# @DateTime :2021/4/14
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, model):
        super(CrossEntropyLoss, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        x = batch['x']
        y = batch['y']
        logit = self.model(x)
        loss = self.criterion(logit, y)
        return loss
