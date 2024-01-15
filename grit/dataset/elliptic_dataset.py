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
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from loguru import logger
from utils.graph_utils import log_loaded_dataset
from transform import RRWPTransform


class EllipticFunctionalDataset(InMemoryDataset):
    def __init__(self, root='./', filepath='./elliptic_bitcoin_dataset', use_edge_attr=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.root = root
        self.filepath = filepath
        self.filenames = os.listdir(filepath)
        self.use_edge_attr = use_edge_attr
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # self.train_data, self.val_data, self.test_data = torch.load(self.processed_paths[1])

    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ['elliptic_txs_features.csv', 'elliptic_txs_classes.csv', 'elliptic_txs_edgelist.csv']

    @property
    def processed_file_names(self):
        return ['elliptic_data_processed.pt']

    def process(self):
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

        # # 划分数据集，未知数据只能放test_set
        known_ids = merge_data['class'].loc[merge_data['class'] != 2].index
        unknown_ids = merge_data['class'].loc[merge_data['class'] == 2].index

        # 存储每个节点的特征，形状是[num_nodes, num_node_features]，一般是float tensor
        # 保留时序先node_feature = merge_data.drop(["class", "txId", "Times"], axis=1)
        node_feature = merge_data.drop(["class", "txId"], axis=1)
        data_x = torch.tensor(np.array(node_feature.values), dtype=torch.float)

        # 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；
        node_label = merge_data['class']
        data_y = torch.tensor(node_label, dtype=torch.long)

        # 用于存储节点之间的边，形状是[2, num_edges]，一般是long tensor。
        edge_index = torch.tensor(np.array(raw_egdes.values), dtype=torch.long).T

        data = Data(x=data_x, edge_index=edge_index, y=data_y)
        data_list = [data]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
        # torch.save((train_data, val_data, test_data), self.processed_paths[1])


def create_dataset_elliptic(config):
    pre_transform = RRWPTransform(**config.pos_enc_rrwp)
    dataset = EllipticFunctionalDataset(pre_transform=pre_transform)

    # # 划分数据集，未知数据只能放test——set
    # known_ids=
    # unknown_ids=

    split_idx = dataset.get_idx_split()
    dataset.split_idxs = [split_idx[s] for s in ['train', 'val', 'test']]
    train_dataset, val_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['val']], dataset[
        split_idx['test']]

    torch.set_num_threads(config.num_workers)
    val_dataset = [x for x in val_dataset]  # Fixed for valid after enumeration
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # pre_transform = RRWPTransform(ksteps=24)
    # dataset = EllipticFunctionalDataset(pre_transform=pre_transform)
    dataset = EllipticFunctionalDataset()
    data = dataset.data
    print(data.num_nodes)
    print(data.num_edges)
    print(data.num_node_features)
    print(data.has_isolated_nodes())
    print(data.is_directed())

    # print(data.edge_index)
    # print(data.edge_index.shape)
    # print(dataset.slices)
    # print(data.y)
    # raw_node = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
    # raw_class = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
    # raw_egdes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
    # raw_node.rename(columns={0: 'txId', 1: 'Times'}, inplace=True)
    #
    # # 节点标签重写
    # class_verify = raw_class.replace({'class': {'unknown': 2, '2': 0, '1': 1}})
    #
    # # 把标签和节点特征拼在一起
    # merge_data = raw_node.merge(class_verify, left_on="txId", right_on="txId")
    #
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
    # print(merge_data.head())
    # print(node_feature.head())
    # print(known_ids)
    # print(unknown_ids)
    #
    # # 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；
    # node_label = merge_data['class']
    # data_y = torch.tensor(node_label, dtype=torch.long)
    #
    # # 用于存储节点之间的边，形状是[2, num_edges]，一般是long tensor。
    # edge_index = torch.tensor(np.array(raw_egdes.values), dtype=torch.long).T
