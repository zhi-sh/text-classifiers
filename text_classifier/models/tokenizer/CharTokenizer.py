# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午5:39
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import collections, json, os
from typing import List, Iterable
from text_classifier.models.tokenizer import AbstractTokenizer


class CharTokenizer(AbstractTokenizer):
    r'''按字进行分词'''

    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = [], do_lower_case: bool = False):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def get_vocab(self, vocab: Iterable[str]):
        return self.vocab

    def tokenize(self, text: str) -> List[int]:
        if self.do_lower_case:
            text = text.lower()

        tokens_filtered = []
        for token in text:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:  # 确保每个token都有对应的词向量
                tokens_filtered.append(self.word2idx[token])
                continue
        return tokens_filtered

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'chartokenizer_config.json'), 'w', encoding='utf-8') as fout:
            json.dump({'vocab': list(self.word2idx.keys()), 'stop_words': list(self.stop_words), 'do_lower_case': self.do_lower_case}, fout, indent=2, ensure_ascii=False)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'chartokenizer_config.json'), encoding='utf-8') as fin:
            config = json.load(fin)
        return CharTokenizer(**config)
