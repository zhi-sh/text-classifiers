# -*- coding: utf-8 -*-
# @DateTime :2021/4/14
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
from text_classifiers.basics import InputExample


class DFDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.data = examples
        self.collate_fn = DFCollator(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_examples(fpath, xcol, ycol, delimiter='\t', max_examples=0):
        import pandas as pd
        df = pd.read_csv(fpath, delimiter=delimiter)
        examples = []
        for ix, r in df.iterrows():
            feats = r[xcol]
            label = r[ycol]
            examples.append(InputExample(feats=feats, label=label))

            if 0 < max_examples <= len(examples):
                return examples
        return examples


class DFCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        xs = []
        ys = []
        lengths = []
        for b in batch:
            tokens = self.tokenizer.tokenize(b.feats)
            xs.append(tokens)
            ys.append(b.label)
            lengths.append(len(tokens))

        # 将序列填充至同样长度
        max_len = max(lengths)
        for i, sz in enumerate(lengths):
            paddings = [self.tokenizer.pad] * (max_len - sz)
            xs[i] = xs[i] + paddings

        encoded = {
            'x': torch.tensor(xs).long(),
            'y': torch.tensor(ys).long(),
            'lengths': torch.tensor(lengths).long()
        }
        return encoded
