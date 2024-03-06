import random

import torch
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from utils.utils import save_model, get_regresssion_metrics, get_metrics, LossAnomalyDetector
import torch.nn.functional as F
from sklearn.metrics import classification_report, matthews_corrcoef

hyper_params = [8, 8, 6, 8, 6, 8, 8, 10, 9, 9, 8, 5, 5, 8, 6, 10, 9, 8, 8, 5, 5, 6, 5, 4, 7, 5, 5, 6, 5, 4,
                7, 6, 5, 3,
                6, 8, 5, 5, 5, 7, 8, 7, 7, 6, 5, 7, 7, 8, 7, 6, 5, 7, 8, 8, 10, 5, 5, 7, 5, 4, 6, 5, 5, 7,
                5, 6, 5, 5,
                5, 8, 5, 6, 6, 5, 6, 10, 8, 8, 8, 4, 5, 10, 8, 9, 9, 5, 5, 10, 10, 10, 10, 5, 5, 10, 12,
                12, 10, 7, 5,
                11, 11, 12, 11, 7, 5, 12, 11, 12, 12, 6, 5, 8, 8, 9, 8, 8, 8, 12, 10, 12, 12, 7, 5, 5, 5,
                8, 8, 7, 5, 8,
                10, 8, 10, 5, 5, 6, 8, 9, 5, 3, 5, 10, 8, 8, 11, 5, 5, 9, 8, 8, 11, 5, 5, 10, 8, 9, 12, 5,
                5, 8, 8, 9, 8, 5, 5]


# for i, dim in enumerate(hyper_params):
#     print(i)
#     print(dim)
#
# a = ['1', '1', '1', '1']
#
# for x in a:
#     assert x == a[0]

# z = torch.tensor([[0.413, 0.3474, 0.5557],
#                   [0.1632, 0.2163, 0.6436]])
# z = z.argmax(dim=-1)
# # print(z)
# # 预测值
# predictions = torch.tensor([-0.9, 0.1])
# # 真实标签
# targets = torch.tensor([1.0, 0.0])
#
# # 计算损失
loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_fn2 = torch.nn.BCELoss()
# loss = loss_fn(predictions, targets)
# index = torch.randperm(10)
# print(loss)
# print(index)
def custom_one_hot(labels, num_classes):
    # 初始化一个全零矩阵，行数为标签数量，列数为类别数量
    one_hot_labels = np.zeros((len(labels), num_classes), dtype=np.float32)

    # 对于每个标签，将对应位置置为1
    for i, label in enumerate(labels):
        if label == 0:
            one_hot_labels[i][0] = 1.0
        elif label == 1:
            one_hot_labels[i][1] = 1.0
        # 对于其他类别，保持全零

    return one_hot_labels


# 测试数据
# labels = [0, 1, 2, 1, 0]
# num_classes = 2  # 类别数量
#
# # 转换标签为类似one-hot形式
# one_hot_labels = custom_one_hot(labels, num_classes)
# print(one_hot_labels)
import torch

import torch


import torch

def random_select_ratio_vectors(tensor, ratio):
    # 获取张量中的向量数量
    num_vectors = tensor.size(1)

    # 计算需要选择的向量数量
    num_select = int(num_vectors * ratio)

    # 随机选择索引
    selected_indices = torch.randperm(num_vectors)[:num_select]

    # 根据索引选择向量
    selected_vectors = tensor[:, selected_indices]

    return selected_vectors

# # 测试数据
# tensor = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])  # 2行N列的张量
# ratio = 0.5  # 选择50%的向量
#
# # 随机选择向量
# selected_vectors = random_select_ratio_vectors(tensor, ratio)
# print(selected_vectors)
list=[1,2]
print(3 in list)
print(1 in list)
print(random.randint(0,2))