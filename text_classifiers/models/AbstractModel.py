# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午5:15
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import os
import torch
from abc import ABC, abstractmethod
from text_classifiers.tools import tools


class AbstractModel(ABC):
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        saved_config_path = os.path.join(output_path, 'config.json')
        saved_model_path = os.path.join(output_path, 'pytorch_model.bin')

        tools.json_save(self.get_config_dict(), saved_config_path, indent=2)
        torch.save(self.state_dict(), saved_model_path)

    @staticmethod
    @abstractmethod
    def load(input_path: str):
        pass
