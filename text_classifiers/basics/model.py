# -*- coding: utf-8 -*-
# @DateTime :2021/4/3
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import json
from abc import ABC, abstractmethod

import torch


class AbstractModel(ABC):
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        config_path = os.path.join(output_path, 'config.json')
        model_path = os.path.join(output_path, 'pytorch_model.bin')

        self.json_save(self.get_config_dict(), config_path, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), model_path)

    @staticmethod
    @abstractmethod
    def load(input_path):
        pass

    @staticmethod
    def json_save(obj, fpath, **kwargs):
        r'''以JSON格式保存文件'''
        with open(fpath, 'w', encoding='utf-8') as fout:
            json.dump(obj, fout, **kwargs)

    @staticmethod
    def json_load(fpath):
        r'''加载JSON格式对象'''
        assert os.path.isfile(fpath)
        with open(fpath, 'r', encoding='utf-8') as fin:
            obj = json.load(fin)
        return obj
