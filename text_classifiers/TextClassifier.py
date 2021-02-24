# -*- coding: utf-8 -*-
# @DateTime :2021/2/20 下午4:39
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging, json, os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple, Type

import torch
import transformers
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import trange
from text_classifiers import __version__
from text_classifiers.tools import tools, utils
from text_classifiers.evaluations import AbstractEvaluation

logger = logging.getLogger(__name__)


class TextClassifier(nn.Sequential):
    def __init__(self, model_path: str = None, modules: Iterable[nn.Module] = None, device: str = None):
        # 加载已训练的模型
        if model_path is not None:
            logger.info(f'load Image Classifier model from : {model_path}')
            # 确认版本一致
            with open(os.path.join(model_path, 'config.json')) as fin:
                config = json.load(fin)
                if config['__version__'] != __version__:
                    logger.error(f'version of image classifier should be same!')

            # 依次加载各层模型
            with open(os.path.join(model_path, 'modules.json')) as fin:
                contained_modules = json.load(fin)

            modules = OrderedDict()
            for module_config in contained_modules:
                module_class = utils.import_from_string(module_config['type'])
                module = module_class.load(os.path.join(model_path, module_config['path']))
                modules[module_config['name']] = module

        if (modules is not None) and (not isinstance(modules, OrderedDict)):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._target_device = torch.device(device)

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: AbstractEvaluation = None,
            epochs=1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[torch.optim.Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5, 'eps': 1e-6, 'correct_bias': False},
            weight_decay: float = 0.01,
            max_grad_norm: float = 1.0,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            ):
        self.to(self._target_device)
        self.best_score = -999999

        tools.ensure_path_exist(output_path)

        # dataloader列表，并为每个dataloader添加collate_fn
        dataloaders = [dataloader for dataloader, _ in train_objectives]
        for dataloader in dataloaders:
            dataloader.collate_fn = self.batching_collate  # 文本转为Batch的策略

        # 获取所有包含模型模型损失类
        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self._target_device)

        # 计算所有数据集，一轮训练最小的步数
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int(steps_per_epoch * epochs)

        # 不同的训练对，提供不同的优化器和学习率策略
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer_obj = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer_obj, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)
            optimizers.append(optimizer_obj)
            schedulers.append(scheduler_obj)

        # 封装各个dataloader为iter
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        num_train_objectives = len(train_objectives)
        # 模型开始的基准评估
        self._eval_during_training(evaluator, output_path, save_best_model, -1, -1)
        for epoch in trange(epochs, desc='Epoch'):
            training_steps = 0  # 训练统计，用以计算何时评估模型

            # 每轮先更改模型模式，并清空梯度
            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc='Iteration', smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    # 循环遍历每个数据集的dataloader
                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data

                    loss_value = loss_model(features, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                training_steps += 1
                if (evaluation_steps > 0) and (training_steps % evaluation_steps == 0):
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

            # 每Epoch最后评估一次模型
            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1)

    def evaluate(self, evaluator: AbstractEvaluation, output_path: str = None):
        r'''评估模型'''
        tools.ensure_path_exist(output_path)
        return evaluator(self, output_path)

    def save(self, path: str):
        r'''保存模型'''
        if path is None:
            return

        tools.ensure_path_exist(path)
        logger.info(f'save model to {path}')

        contained_modules = []
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(path, f'{str(idx)}_{type(module).__name__}')
            tools.ensure_path_exist(model_path)
            module.save(model_path)
            contained_modules.append({'idx': idx, 'name': name, 'path': os.path.basename(model_path), 'type': type(module).__module__})

        tools.json_save(contained_modules, os.path.join(path, 'modules.json'), indent=2)
        tools.json_save({'__version__': __version__}, os.path.join(path, 'config.json'), indent=2)

    def batching_collate(self, batch):
        r'''将一个batch的文本转为Tensor'''
        texts = []
        labels = []

        for example in batch:
            texts.append(example.text)
            labels.append(example.label)

        labels = torch.tensor(labels).to(self._target_device)

        features = self.tokenize(texts)  # 调用子模型，表示tokenize每个文本
        utils.batch_to_device(features, self._target_device)

        return features, labels

    def tokenize(self, text: str):
        return self._first_module().tokenize(text)

    # --------------------------------------------- 模型属性 ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            # TODO nn.DataParaParallel compatibility
            return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # --------------------------------------------- 内部函数 ----------------------------------------------------------
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps):
        if evaluator is not None:  # 如果存在评估实体，则进行评估模型的过程
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if score >= self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        r'''获取学习率策略'''
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _first_module(self):
        r'''returns the first moudle of this sequential embedder'''
        return self._modules[next(iter(self._modules))]
