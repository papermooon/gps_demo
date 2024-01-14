import hashlib
import os
import os.path as osp
import pickle
import shutil
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from loguru import logger
from utils.graph_utils import log_loaded_dataset
from transform import RRWPTransform
import matplotlib.pyplot as plt

# merge_data = pd.read_csv('./feat_label.csv')
# node_feature = merge_data.drop(["class", "txId", "Times"], axis=1)
#
# print(node_feature)

# 离散化、重新构建子图
hyper_params = [0, 0, 8, 8, 6, 8, 6, 8, 8, 10, 9, 9, 8, 5, 5, 8, 6, 10, 9, 8, 8, 5, 5, 6, 5, 4, 7, 5, 5, 6, 5, 4, 7, 6,
                5, 3,
                6, 8, 5, 5, 5, 7, 8, 7, 7, 6, 5, 7, 7, 8, 7, 6, 5, 7, 8, 8, 10, 5, 5, 7, 5, 4, 6, 5, 5, 7, 5, 6, 5, 5,
                5, 8, 5, 6, 6, 5, 6, 10, 8, 8, 8, 4, 5, 10, 8, 9, 9, 5, 5, 10, 10, 10, 10, 5, 5, 10, 12, 12, 10, 7, 5,
                11, 11, 12, 11, 7, 5, 12, 11, 12, 12, 6, 5, 8, 8, 9, 8, 8, 8, 12, 10, 12, 12, 7, 5, 5, 5, 8, 8, 7, 5, 8,
                10, 8, 10, 5, 5, 6, 8, 9, 5, 3, 5, 10, 8, 8, 11, 5, 5, 9, 8, 8, 11, 5, 5, 10, 8, 9, 12, 5, 5, 8, 8, 9,
                8, 5, 5]
node_feature = pd.read_csv('./feat_label.csv')
for i in range(2, 167):
    data = node_feature[str(i)]
    # node_feature[str(i)].hist()
    # plt.show()
    data_reshape = data.values.reshape((data.shape[0], 1))
    estimator = KMeans(n_clusters=hyper_params[i], random_state=0, n_init='auto')
    res = estimator.fit_predict(data_reshape)
    node_feature[str(i)] = res
node_feature.to_csv('digit_fit_label.csv', index=False)
# print(node_feature)
# print(node_feature.max())
# print(node_feature.min())

# node_feature = pd.read_csv('./digit_fit_label.csv')
# for i in range(2, 167):
#     data = node_feature[str(i)]
#     node_feature[str(i)].hist()
#     plt.show()
# print(node_feature)


# 决策树离散化和k-means

# 2、聚类算法
# for i in range(2, 167):
#     data = node_feature[str(i)]
#     data_len = data.shape[0]
#     data_reshape = data.values.reshape((data.shape[0], 1))
#
#     X = range(2, 20)
#     SSE = []  # 存放每次结果的误差平方和
#
#     for k in range(2, 20):
#         estimator = KMeans(n_clusters=k, random_state=0, n_init='auto')
#         estimator.fit_predict(data_reshape)
#         # 轮廓系数# test.append(silhouette_score(data_reshape, estimator.labels_,sample_size=10000, metric='euclidean'))
#         SSE.append(estimator.inertia_ / data_len)  # estimator.inertia_获取聚类准则的总和
#     plt.xlabel('k-feat-'+str(i))
#     plt.xticks(range(1, 25))
#     plt.ylabel('SSE')
#     plt.plot(X, SSE, 'o-')
#     plt.savefig('./feat_work/feat'+str(i)+'.png')
#     plt.show()
#
# # 如果给出n个坐标点，则可以使用n-1项多项式进行拟合。
# coefficients = np.polyfit(list(X), SSE, 7)
# # 定义多项式函数
# def polynomial_function(x):
#     return np.polyval(coefficients, x)
# # 计算一阶导数和二阶导数
# derivative_1 = np.polyder(coefficients)
# derivative_2 = np.polyder(derivative_1)
# def curvature(x):
#     numerator = np.abs(np.polyval(derivative_2, x))
#     denominator = (1 + np.polyval(derivative_1, x) ** 2) ** (3 / 2)
#     return numerator / denominator
# # 在某一段 x 取值范围内计算曲率
# curvature_values = curvature(np.arange(2.0, 9.0, 0.0001))
# print(curvature_values)
# plt.plot(np.arange(2.0, 9.0, 0.0001), curvature_values, color='g', label='Fit')
# plt.plot(list(X), np.polyval(coefficients, list(X)), color='r', label='Fit')
