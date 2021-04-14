# -*- coding: utf-8 -*-
# @DateTime :2021/4/14
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
from abc import ABC, abstractmethod
from typing import List, Union
from collections import Counter

import pandas as pd


class InputExample:

    def __init__(self, feats, label, guid: str = ''):
        self.guid = guid
        self.feats = feats
        self.label = label

    def __str__(self):
        return f"<InputExample> guid: {self.guid}, label: {self.label}, feats: {self.feats}"


class BaseTokenizer(ABC):
    PAD = '[PAD]'
    UNK = '[UNK]'
    _WORD = ['[PAD]', '[UNK]']  # 默认字符集

    @abstractmethod
    def tokenize(self, sentence):
        pass

    def cut_sentence_by_chars(self, sentence, stopwords=None):
        tokens = []
        for char in sentence:
            tokens.append(char)

        if stopwords is not None:
            tokens = [c for c in tokens if (c not in stopwords)]
        return tokens

    def build_chars_vocab_from_text_by_single(self, fpaths: Union[List[str], str], size: int = 100000, stopwords: Union[List[str], str] = None):
        # 1. 加载停用词表
        stop_set = self._stopwords_set(stopwords)

        # 2. 读取语料, 统计语料中的词频
        counter = Counter()
        if isinstance(fpaths, str):
            fpaths = [fpaths]

        for fpath in fpaths:
            assert os.path.isfile(fpath)
            with open(fpath, encoding='utf-8') as fr:
                for line in fr:
                    tokens = self.cut_sentence_by_chars(line.strip(), stop_set)
                    counter.update(tokens)

        # 3. 根据统计频次，计算出字典
        self._build_vocab(counter, size)

    def build_chars_vocab_from_dataframe(self, fpaths: Union[List[str], str], text_col: str, delimiter='\t', size: int = 100000, stopwords: Union[List[str], str] = None):
        # 1. 加载停用词表
        stop_set = self._stopwords_set(stopwords)

        # 2. 读取语料, 统计语料中的词频
        counter = Counter()
        if isinstance(fpaths, str):
            fpaths = [fpaths]

        for fpath in fpaths:
            assert os.path.isfile(fpath)
            df = pd.read_csv(fpath, delimiter=delimiter)
            for ix, r in df.iterrows():
                tokens = self.cut_sentence_by_chars(r[text_col].strip(), stop_set)
                counter.update(tokens)

        # 3. 根据统计频次，计算出字典
        self._build_vocab(counter, size)

    def _stopwords_set(self, stopwords):
        stop_set = None
        if isinstance(stopwords, list):
            stop_set = set(stopwords)
        elif isinstance(stopwords, str) and os.path.isfile(stopwords):
            with open(stopwords, encoding='utf-8') as fr:
                stop_set = set([w.strip() for w in fr.readlines()])
        self.stopwords = stop_set
        return stop_set

    def _build_vocab(self, counter, size):
        words = [c for c in self._WORD]
        for w, c in counter.most_common(size - len(self._WORD)):
            words.append(w)

        self.w2ix = {w: i for i, w in enumerate(words)}
        self.ix2w = {v: k for k, v in self.w2ix.items()}
        self.vocab_size = len(self.w2ix)


class CharTokenizer(BaseTokenizer):
    def tokenize(self, sentence):
        # 分词
        tokens = self.cut_sentence_by_chars(sentence, self.stopwords)

        # 转TOKEN_ID
        tokens = [self.w2ix.get(w, self.unk) for w in tokens]
        return tokens

    @property
    def unk(self):
        return self.w2ix.get(self.UNK)

    @property
    def pad(self):
        return self.w2ix.get(self.PAD)


class WhiteSpaceTokenizer(BaseTokenizer):
    def tokenize(self, sentence):
        pass
