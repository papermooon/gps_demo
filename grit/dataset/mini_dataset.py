import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_sparse import SparseTensor

from grit.dataset.transform import RRWPTransform, SimilarTransform
import argparse
from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model
import torch.nn.functional as F


# 本文件下root是./ 在finetune时应改为./dataset
class miniDataset(InMemoryDataset):
    def __init__(self, root='./dataset', use_edge_attr=True, transform=None,
                 pre_transform=None, pre_filter=None):
        self.root = root
        self.use_edge_attr = use_edge_attr
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def raw_file_names(self):
        filelist = []
        for i in range(1, 50):
            file = 'graph_data' + str(i) + '.pt'
            filelist.append(file)
        return filelist

    @property
    def processed_file_names(self):
        return ['mini_graphs.pt']

    def process(self):
        data_list = []
        for i in range(1, 50):
            file = 'graph_data' + str(i) + '.pt'
            loaded_data = torch.load('./subs/' + file)
            data_list += loaded_data
        print(data_list)
        print(len(data_list))
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


def create_miniDataset(config):
    # pre_transform = RRWPTransform(**config.pos_enc_rrwp)
    pre_transform = SimilarTransform(**config.pos_enc_rrwp)
    dataset = miniDataset(pre_transform=pre_transform)
    dataset = dataset.shuffle()
    data_len = len(dataset)
    # print(dataset[0])
    # t1 = torch.tensor([[[1, 2, 0], [0, 0, 3]],
    #                    [[0, 0, 0], [4, 5, 0]]])
    # print(t1)
    # print(t1.shape)
    # abs_pe = t1.diagonal().transpose(0, 1)
    # rel_pe = SparseTensor.from_dense(t1, has_value=True)
    # rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
    # print(rel_pe_idx)
    # print(rel_pe_val)
    # print(rel_pe_val.shape)
    # print(abs_pe)
    # # 统计一下连通分量
    # for G in dataset:
    #     print(G)
    #     G = nx.from_edgelist(G.edge_index.t().numpy())
    #     # 计算连通分量
    #     connected_components = nx.connected_components(G)
    #     for i, component in enumerate(connected_components):
    #         print(f"Connected Component {i + 1}: {list(component)}")

    train_dataset = dataset[:int(data_len * 0.7)]
    val_dataset = dataset[int(data_len * 0.7):int(data_len * 0.9)]
    test_dataset = dataset[int(data_len * 0.9):]
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    # print(train_dataset[1])
    import torch_geometric.transforms as T

    split_train = T.RandomNodeSplit(split="train_rest", num_val=0., num_test=0.)
    split_val = T.RandomNodeSplit(num_val=1.0)
    split_test = T.RandomNodeSplit(num_test=1.0, num_val=0.)

    tmp_train = []
    for data in train_dataset:
        data = split_train(data)
        data.split = 'train'
        tmp_train.append(data)
    train_dataset = tmp_train
    # mask = '{}_mask'.format(train_dataset[0].split)
    # print(train_dataset[0][mask])
    # print(train_dataset[0].x)
    # print(train_dataset[0].x[train_dataset[0][mask]])
    # print(train_dataset[0].x['train_mask'])

    tmp_test = []
    for data in test_dataset:
        data = split_test(data)
        data.split = 'test'
        tmp_test.append(data)
    test_dataset = tmp_test

    tmp_val = []
    for data in val_dataset:
        data = split_val(data)
        data.split = 'val'
        tmp_val.append(data)
    val_dataset = tmp_val

    # from torch_geometric.loader import DataLoader
    # train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=config.num_workers,
    #                               pin_memory=True)
    # mixup_ratio = 0.5
    #
    # for it in train_dataloader:
    #     data = it[0]
    #     b=data.binary_label
    #     l=data.y
    #     print(b[0:20])
    #     print(l[0:20])
    #     return

    return train_dataset, val_dataset, test_dataset




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../task_finetune.yaml')
    args = parser.parse_args()
    config = parse_config(args.config)
    create_miniDataset(config.data)
