# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午4:50
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import pandas as pd
from text_classifiers.readers import InputExample


class ReaderWaiMai:
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
        texts = df.review.tolist()
        labels = df.label.tolist()

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

    @staticmethod
    def get_labels():
        return {'0': 0, '1': 1}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        if not isinstance(label, str):
            label = str(label)
        return self.get_labels()[label.strip().lower()]
