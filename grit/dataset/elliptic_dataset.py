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
from .transform import RRWPTransform

raw_node = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_features.csv', header=None)
raw_class = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_classes.csv')
raw_egdes = pd.read_csv('./elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv')



class EllipticFunctionalDataset(InMemoryDataset):
    def download(self):
        pass

    @property
    def raw_file_names(self):
        return ['elliptic_txs_features.csv','elliptic_txs_classes.csv','elliptic_txs_edgelist.csv']

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
