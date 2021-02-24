# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午5:39
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import collections, json, os
from typing import List, Iterable
from text_classifiers.models.tokenizer import AbstractTokenizer


class CharTokenizer(AbstractTokenizer):
    r'''按字进行分词, 仅支持中文'''

    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = [], do_lower_case: bool = False):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def get_vocab(self):
        return self.vocab

    def get_padding_idx(self):
        return self.word2idx.get('<PAD>', None)

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

    @staticmethod
    def vocab_from_file(raw_text_file: str, max_vocab_size: int = 100000):
        r'''返回 字典列表'''
        counter = collections.Counter()

        # 读取每一行的字符
        with open(raw_text_file, encoding='utf-8') as fr:
            for line in fr:
                counter.update(list(line.rstrip()))

        vocab = ['<PAD>', '<UNK>']  # 确保<PAD>为第一个
        for w, c in counter.most_common(max_vocab_size - 2):
            vocab.append(w)

        return vocab
