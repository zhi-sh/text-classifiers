# -*- coding: utf-8 -*-
# @DateTime :2021/2/28 下午5:39
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os, json
import pandas as pd
from text_classifiers.readers import InputExample


class ReaderNews41:
    r'''针对具体数据集的读取器'''

    def __init__(self, dataset_folder: str):
        self.dataset_folder = dataset_folder

    def get_examples(self, filename, max_examples=0):
        r'''
        :param filename: 语料文件
        :param max_examples: 小批量采样条数
        :return: InputExample列表
        '''
        df = pd.read_csv(os.path.join(self.dataset_folder, filename))
        texts = df.headline.tolist()  # 评论内容，不同的数据集有特定的列
        labels = df.category.tolist()  # 标签，不同的数据集有特定的列

        examples = []
        idx = 0
        for text, label in zip(texts, labels):
            guid = str(idx)
            idx += 1
            examples.append(InputExample(guid=guid, text=text, label=self.map_label(label)))

            # 若设置max_examples，则读取固定数量的样本数据
            if 0 < max_examples <= len(examples):
                break

        return examples

    def get_labels(self):
        with open(f'{self.dataset_folder}/label_dict.json') as fr:
            dct = json.load(fr)
        return dct

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        if not isinstance(label, str):
            label = str(label)
        return self.get_labels()[label.strip().lower()]
