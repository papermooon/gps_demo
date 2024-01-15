import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from grit.dataset.transform import RRWPTransform
import argparse
from utils.utils import parse_config, set_random_seed, log_GPU_info, load_model


class EllipticFunctionalDataset(InMemoryDataset):
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
            file = 'sub_' + str(i) + '.csv'
            filelist.append(file)
        return filelist

    @property
    def processed_file_names(self):
        return ['sub_graphs.pt']

    def process(self):
        data_list = []
        raw_edges = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')
        for i in range(1, 50):
            file = 'sub_' + str(i) + '.csv'
            sub_graph = pd.read_csv('./subs/' + file)

            # 按txId排序
            sub_graph = sub_graph.sort_values('txId').reset_index(drop=True)
            nodes = sub_graph['txId'].values

            index_list = []
            for node in nodes:
                index_list += raw_edges[raw_edges['txId1'] == node].index.tolist()

            index_list = pd.Index(index_list)
            sub_edges = raw_edges.loc[index_list]
            # 重写各节点id，重写连边
            map_id = {j: i for i, j in enumerate(nodes)}
            sub_edges.txId1 = sub_edges.txId1.map(map_id)
            sub_edges.txId2 = sub_edges.txId2.map(map_id)

            node_feature = sub_graph.drop(["class", "txId", "Times"], axis=1)
            data_x = torch.tensor(np.array(node_feature.values), dtype=torch.long)

            node_label = sub_graph['class']
            data_y = torch.tensor(node_label, dtype=torch.long)

            edge_index = torch.tensor(np.array(sub_edges.values), dtype=torch.long).T

            data = Data(x=data_x, edge_index=edge_index, y=data_y)
            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])


def create_dataset_elliptic(config):
    pre_transform = RRWPTransform(**config.pos_enc_rrwp)
    dataset = EllipticFunctionalDataset(pre_transform=pre_transform)
    dataset = dataset.shuffle()
    train_dataset = dataset[:34]
    val_dataset = dataset[34:39]
    test_dataset = dataset[39:]
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    # print(train_dataset[1])
    import torch_geometric.transforms as T

    split_train = T.RandomNodeSplit(split="train_rest", num_val=0., num_test=0.)
    split_val = T.RandomNodeSplit(num_val=1.0)
    split_test = T.RandomNodeSplit(split="test_rest")

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

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../task_finetune.yaml')
    args = parser.parse_args()
    config = parse_config(args.config)
    create_dataset_elliptic(config.data)
