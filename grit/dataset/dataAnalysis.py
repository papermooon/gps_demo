import hashlib
import os
import os.path as osp
import pickle
import shutil
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from loguru import logger
from utils.graph_utils import log_loaded_dataset
from transform import RRWPTransform

# 把标签和节点特征拼在一起
merge_data = pd.read_csv('./digit_fit_label.csv')
raw_egdes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
# 查看连边是否有不同的时间片
# ct = 0
# for tup in zip(raw_egdes['txId1'], raw_egdes['txId2']):
#     tx1 = merge_data[merge_data['txId'] == tup[0]]
#     tx2 = merge_data[merge_data['txId'] == tup[1]]
#     if tx1.Times.tolist()[0] != tx2.Times.tolist()[0]:
#         ct += 1
# print(ct)

# 按时间片分成49张子图
# for i in range(1, 50):
#     filename = 'sub_' + str(i) + '.csv'
#     tmp = merge_data[merge_data['Times'] == i]
#     tmp.to_csv('./subs/' + filename, index=False)

# print(merge_data.columns.tolist())
# print(merge_data[merge_data['Times'] == 1])
# 统计不同时间戳的节点数量
# for i in range(1, 50):
#     print(i)
#     tmp = merge_data[merge_data['Times'] == i]
#     print(len(tmp))

# 统计不同子图的交易比例
for i in range(1, 50):
    filename = 'sub_' + str(i) + '.csv'
    merge_data = pd.read_csv('./subs/' + filename)
    # class_message = merge_data['class']
    # class_message.hist()
    # plt.show()
    label0 = len(merge_data[merge_data['class'] == 0])
    label1 = len(merge_data[merge_data['class'] == 1])
    label2 = len(merge_data[merge_data['class'] == 2])
    print(i, '-----------', 'label0:', label0, 'label1:', label1, 'label2:', label2)
    print(i, '非法/合法----------', label1 / label0 * 100, "%")

# # 按txId排序
# merge_data = merge_data.sort_values('txId').reset_index(drop=True)
# nodes = merge_data['txId'].values
#
# # 重写各节点id，重写连边
# map_id = {j: i for i, j in enumerate(nodes)}
# raw_egdes.txId1 = raw_egdes.txId1.map(map_id)
# raw_egdes.txId2 = raw_egdes.txId2.map(map_id)
#
# # # 划分数据集，未知数据只能放test_set
# known_ids = merge_data['class'].loc[merge_data['class'] != 2].index
# unknown_ids = merge_data['class'].loc[merge_data['class'] == 2].index
#
# # 存储每个节点的特征，形状是[num_nodes, num_node_features]，一般是float tensor
# # 保留时序先node_feature = merge_data.drop(["class", "txId", "Times"], axis=1)
# node_feature = merge_data.drop(["class", "txId"], axis=1)
# data_x = torch.tensor(np.array(node_feature.values), dtype=torch.float)
#
# # 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；
# node_label = merge_data['class']
# data_y = torch.tensor(node_label, dtype=torch.long)
#
# # 用于存储节点之间的边，形状是[2, num_edges]，一般是long tensor。
# edge_index = torch.tensor(np.array(raw_egdes.values), dtype=torch.long).T
#
# data = Data(x=data_x, edge_index=edge_index, y=data_y)
# data_list = [data]
