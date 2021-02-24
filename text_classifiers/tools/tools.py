# -*- coding: utf-8 -*-
# @DateTime :2021/2/24 上午11:18
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import json, os

import os, pickle


# --------------------------- 文件目录相关操作 -----------------------------------
def ensure_path_exist(path):
    r'''确保目录存在。如果不存在，则创建'''
    if path is not None:
        os.makedirs(path, exist_ok=True)


def gather_files_by_ext(path, ext, files: list = None):
    r'''递归获取目录下特定类型的所有文件'''
    if files is None:
        files = []
    file_list = os.listdir(path)  # 首先遍历当前目录下所有文件及文件夹
    for file in file_list:  # 遍历当前目录下所有文件
        cur_path = os.path.join(path, file)  # 当前文件的全路径
        if os.path.isdir(cur_path):
            # 如果是子目录，则递归遍历子目录
            gather_files_by_ext(cur_path, ext=ext, files=files)
        else:
            if cur_path.endswith(ext):  # 如果文件以指定类型结尾，则加入列表
                files.append(cur_path)

    return files


def pkl_save(obj, fpath):
    r'''对象序列化过程'''
    with open(fpath, 'wb') as fw:
        pickle.dump(obj, fw)


def pkl_load(fpath):
    r'''对象反序列化过程'''
    assert os.path.isfile(fpath)
    with open(fpath, 'rb') as fr:
        obj = pickle.load(fr)
    return obj


def json_save(obj, fpath, **kwargs):
    r'''以JSON格式保存文件'''
    with open(fpath, 'w', encoding='utf-8') as fout:
        json.dump(obj, fout, **kwargs)


def json_load(fpath):
    r'''加载JSON格式对象'''
    assert os.path.isfile(fpath)
    with open(fpath, 'r', encoding='utf-8') as fin:
        obj = json.load(fin)
    return obj
