# -*- coding: utf-8 -*-
# @DateTime :2021/4/14
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import torch
from torch.utils.data import Dataset
from text_classifiers.basics import InputExample


class TextLabelPairDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.data = examples
        self.collate_fn = DFCollator(tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def get_df_examples(fpath, xcol, ycol=None, delimiter='\t', max_examples=0):
        import pandas as pd
        df = pd.read_csv(fpath, delimiter=delimiter)
        examples = []
        for ix, r in df.iterrows():
            feats = r[xcol]
            if ycol is not None:
                label = r[ycol]
            else:
                label = -1
            examples.append(InputExample(feats=feats, label=label))

            if 0 < max_examples <= len(examples):
                return examples
        return examples

    @staticmethod
    def get_text_examples(fpath, delimiter='\t', max_examples=0):
        examples = []
        with open(fpath, encoding='utf-8') as fr:
            for line in fr:
                pairs = line.strip().split(delimiter)
                if len(pairs) == 2:
                    feats = pairs[0]
                    label = int(pairs[1])
                    examples.append(InputExample(feats=feats, label=label))
                elif len(pairs) == 1:
                    feats = pairs[0]
                    label = -1
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
