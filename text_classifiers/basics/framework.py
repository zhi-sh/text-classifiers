# -*- coding: utf-8 -*-
# @DateTime :2021/4/3
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import importlib
import math, json, os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Iterable, Tuple

import torch
import transformers
import numpy as np
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

config_demo = {
    # Framework 参数
    "device": None,  # 框架自己探测使用GPU or CPU
    "batch_size": 16,
    "weight_decay": 0.01,
    "scheduler_type": None,  # [WarmupLinear],
    "output_path": None,
}


class AbstractFramework(nn.Sequential, ABC):
    def __init__(self, model_path: str = None, modules: Iterable[nn.Module] = None, conf={}):
        self.conf = conf  # 全局配置字典

        # 如果参数是个路径，则加载各层模型
        if model_path is not None:
            print(f'load pretrained model from : {model_path}')
            with open(os.path.join(model_path, 'modules.json')) as fin:
                contained_modules = json.load(fin)
            modules = OrderedDict()
            for module_config in contained_modules:
                module_class = import_from_string(module_config['type'])
                module = module_class.load(os.path.join(model_path, module_config['path']))
                modules[module_config['name']] = module

        # 如果参数是模型列表
        if (modules is not None) and (not isinstance(modules, OrderedDict)):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)

        # 确认模型的基础设备
        device = self.conf.get('device')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._target_device = torch.device(device)

        # 模型迁移到对应的设备
        self.to(self._target_device)

        # 模型属性
        self.best_valid_loss = float('inf')
        self.round_counter = 0

    def fit(self,
            train_objective: Tuple[Dataset, nn.Module],  # 训练数据集，损失模型
            valid_dataset: Dataset = None,
            epochs: int = 1,
            patience: int = 5,  # 早停法，等待的轮次
            ):

        dataset, loss_model = train_objective
        loss_model.to(self._target_device)
        dataloader = self._dataloader(dataset=dataset, shuffle=True)

        num_steps = int(len(dataloader) * epochs)
        warmup_steps = math.ceil(num_steps * 0.1)  # 10% 的数据用以warmup
        weight_decay = self.conf.get('weight_decay', 0.01)
        scheduler = None
        if hasattr(self._first_module(), 'get_optimizer'):
            optimizer = self._first_module().get_optimizer(loss_model.parameters())
        else:
            param_optimizer = list(loss_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer_params = {'lr': 2e-5}
            optimizer = transformers.AdamW(optimizer_grouped_parameters, **optimizer_params)
            scheduler = self._get_scheduler(optimizer, scheduler=self.conf.get('scheduler_type'), warmup_steps=warmup_steps, t_total=num_steps)

        # 模型开始的基准评估
        self._eval_during_training(loss_model, valid_dataset, -1)
        progress_bar = tqdm(range(epochs), desc='Epoch ')
        for epoch in progress_bar:
            # 每轮先更改模型模式，并清空梯度
            loss_model.zero_grad()
            loss_model.train()
            for training_step, _batch in enumerate(dataloader):
                # 循环遍历每个数据集的dataloader
                loss_value = loss_model(_batch)
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_model.parameters(), self.conf.get('max_grad_norm', 1.0))
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                progress_bar.set_description(f"Epoch: {epoch} / step: {training_step} loss = {loss_value.item():.6f}")

            self._eval_during_training(loss_model, valid_dataset, epoch)
            if self.round_counter > patience:
                print(f"\nEarlyStopping is using, best valid loss: {self.best_valid_loss:.6f}")
                break

    def evaluation(self, dataset: Dataset, metrics=['accuracy']):
        self.eval()
        dataloader = self._dataloader(dataset)
        labels_list = []
        preds_list = []
        for data in dataloader:
            features, labels = data['x'], data['y']
            labels_list.append(labels.cpu().detach().numpy())
            with torch.no_grad():
                logits = self(features)
                preds = logits.argmax(dim=-1)
            preds_list.append(preds.cpu().detach().numpy())
        labels = np.asarray(labels_list)
        preds = np.asarray(preds_list)
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        scores = {}
        for mtype in metrics:
            scores.update(self._score(labels, preds, mtype))
        return scores

    def save(self):
        path = self.conf.get('output_path')
        if path is None:
            return
        os.makedirs(path, exist_ok=True)

        # 保存每一层模型
        contained_modules = []
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, str(idx) + "_" + type(module).__name__)
            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        # 保存模型列表配置文件
        with open(os.path.join(path, 'modules.json'), 'w') as fout:
            json.dump(contained_modules, fout, indent=2)

    def _eval_during_training(self, loss_model, dataset, epoch):
        if dataset is None:
            return float('inf')

        loss_model.eval()
        dataloader = self._dataloader(dataset=dataset)
        losses = []
        for data in dataloader:
            features, labels = data['x'], data['y']
            with torch.no_grad():
                loss_value = loss_model(features, labels)
                losses.append(loss_value.cpu().detach().numpy())
        losses = np.asarray(losses)
        valid_loss = losses.mean()

        if valid_loss < self.best_valid_loss:
            print(f'\nEpoch {epoch}: Best valid loss updating ({self.best_valid_loss:.6f} -> {valid_loss:.6f})')
            self.best_valid_loss = valid_loss
            self.round_counter = 0
            self.save()
            print(f"model saved...")
        else:
            self.round_counter += 1

    def _dataloader(self, dataset: Dataset, shuffle=False):
        batch_size = self.conf.get('batch_size', 16)
        collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else self.collate_fn
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        return dataloader

    def _score(self, y_true, y_pred, mtype):
        if mtype == 'accuracy':
            return {'accuracy': metrics.accuracy_score(y_true, y_pred)}
        elif mtype == 'f1_score':
            return {'f1_score': metrics.f1_score(y_true, y_pred)}
        elif mtype == 'auc':
            from sklearn.metrics import roc_auc_score
            return {'auc': metrics.roc_auc_score(y_true, y_pred)}
        elif mtype == 'precision':
            return {'precision': metrics.precision_score(y_true, y_pred)}
        elif mtype == 'recall':
            return {'recall': metrics.recall_score(y_true, y_pred)}

    def _first_module(self):
        return self._modules[next(iter(self._modules))]

    def _last_module(self):
        return self._modules[next(reversed(self._modules))]

    @abstractmethod
    def collate_fn(self, batch):
        pass

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        r'''获取学习率策略'''
        if scheduler is None:
            return None


# ----------------------------- 公用函数 ------------------------------
def import_from_string(dotted_path):
    r'''根据点分的字符串路径，导入对应的模块'''
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path."
        raise ImportError(msg)

    module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f"Module {module_path} does not define a {class_name} attribute/class."
        raise ImportError(msg)
