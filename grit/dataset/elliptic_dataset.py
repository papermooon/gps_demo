import hashlib
import os.path as osp
import pickle
import shutil
from typing import Union, List, Tuple

import pandas as pd
import torch
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm
from loguru import logger

from utils.graph_utils import log_loaded_dataset


# from .transform import RRWPTransform


class EllipticFunctionalDataset(InMemoryDataset):
    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ['elliptic_txs_features.csv', 'elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv']

    @property
    def processed_file_names(self):
        return 'elliptic_data_processed.pt'

    def process(self):
        raw_node = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
        raw_class = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
        raw_egdes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')

        raw_node.rename(columns={0: 'txId', 1: 'Times'}, inplace=True)
        class_verify = raw_class[raw_class['class'] != "unknown"]
        merge_data = raw_node.merge(class_verify, left_on="txId", right_on="txId")
        nodes = merge_data[0].values
        print(nodes)

        merge_data = merge_data.sort_values(0).reset_index(drop=True)

        nodes = merge_data[0].values
        print(nodes)

        # data = Data()


if __name__ == '__main__':
    raw_node = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
    raw_class = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    raw_egdes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')

    raw_node.rename(columns={0: 'txId', 1: 'Times'}, inplace=True)
    # 节点标签重写
    class_verify = raw_class.replace({'class': {'unknown': 2, '2': 0, '1': 1}})
    # 把标签和节点特征拼在一起
    merge_data = raw_node.merge(class_verify, left_on="txId", right_on="txId")
    # 按txId排序
    merge_data = merge_data.sort_values('txId').reset_index(drop=True)
    nodes = merge_data['txId'].values

    # 重写各节点id，重写连边
    map_id = {j: i for i, j in enumerate(nodes)}
    raw_egdes.txId1 = raw_egdes.txId1.map(map_id)
    raw_egdes.txId2 = raw_egdes.txId2.map(map_id)

    print(merge_data.head())

    labels = df_merge['class']  ##标签数据提取
    node_features = df_merge.drop([0, 1, 'txId'], axis=1)
    classify_id = node_features['class'].loc[node_features['class'] != 2].index  ##分类的数据标签，因为数据中包含未知数据，未知数据是用来测试的
    unclassify_id = node_features['class'].loc[node_features['class'] == 2].index  ##未知数据标签
    llic_classify_id = node_features['class'].loc[node_features['class'] == 0].index  ##在分类数据标签的基础上包含非法交易和正常交易数据，把他们分出来
    illic_classify_id = node_features['class'].loc[node_features['class'] == 1].index
    weights = torch.ones(edge_list.shape[0], dtype=torch.double)  ##边的权重随机初始化为1

    ## edge_index转化为 [2,E]形状的tensor 类型为torch.long
    edge_index = np.array(edge_list.values).T
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    node_features.drop(['class'], axis=1, inplace=True)
    node_features = torch.tensor(np.array(node_features.values), dtype=torch.float)
    train_idx, valid_idx = train_test_split(classify_id, test_size=0.2)
    ## 下面建立数据集
    data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                      y=torch.tensor(labels, dtype=torch.float))
    data_train.train_idx = train_idx
    data_train.valid_idx = valid_idx
    data_train.test_idx = unclassify_id
