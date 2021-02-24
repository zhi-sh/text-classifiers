# -*- coding: utf-8 -*-
# @DateTime :2021/2/24 上午11:13
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import importlib
import torch


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


def fullname(o):
    r'''获取对象的类全称'''
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # 避免返回内建方法
    else:
        return module + '.' + o.__class__.__name__


def batch_to_device(batch, target_device: torch.device):
    """将batch放到对应的设备上"""
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
