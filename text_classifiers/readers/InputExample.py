# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午4:40
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from typing import List, Union


class InputExample:
    def __init__(self, guid: str = '', text: str = None, label: Union[int, float] = 0):
        r'''
        :param guid: 数据的标识
        :param text: [文本]
        :param label: 标签
        '''
        self.guid = guid
        self.text = text
        self.label = label

    def __str__(self):
        return f"<InputExample> label: {self.label}, text: {self.text}"
