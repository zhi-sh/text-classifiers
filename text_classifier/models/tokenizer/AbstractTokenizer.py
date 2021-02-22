# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午5:36
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
from typing import Iterable, List
from abc import ABC, abstractmethod


class AbstractTokenizer(ABC):
    @abstractmethod
    def set_vocab(self, vocab: Iterable[str]):
        pass

    @abstractmethod
    def get_vocab(self, vocab: Iterable[str]):
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def save(self, output_path: str):
        pass

    @staticmethod
    @abstractmethod
    def load(input_path: str):
        pass
