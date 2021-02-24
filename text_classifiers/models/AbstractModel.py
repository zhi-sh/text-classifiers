# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午5:15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod


class AbstractModel(ABC):

    @abstractmethod
    def save(self, output_path: str):
        pass

    @staticmethod
    @abstractmethod
    def load(input_path: str):
        pass
