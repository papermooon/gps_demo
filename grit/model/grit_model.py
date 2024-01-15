import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config, BatchNorm1dNode, MLP)

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.graphgym.models.head import GNNGraphHead, GNNNodeHead
from .rrwp_pe import RRWPLinearNodeEncoder, RRWPLinearEdgeEncoder
from .grit_layer import GritTransformerLayer


class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(NodeEncoder, self).__init__()

        self.hyper_params = [8, 8, 6, 8, 6, 8, 8, 10, 9, 9, 8, 5, 5, 8, 6, 10, 9, 8, 8, 5, 5, 6, 5, 4, 7, 5, 5, 6, 5, 4,
                             7, 6, 5, 3,
                             6, 8, 5, 5, 5, 7, 8, 7, 7, 6, 5, 7, 7, 8, 7, 6, 5, 7, 8, 8, 10, 5, 5, 7, 5, 4, 6, 5, 5, 7,
                             5, 6, 5, 5,
                             5, 8, 5, 6, 6, 5, 6, 10, 8, 8, 8, 4, 5, 10, 8, 9, 9, 5, 5, 10, 10, 10, 10, 5, 5, 10, 12,
                             12, 10, 7, 5,
                             11, 11, 12, 11, 7, 5, 12, 11, 12, 12, 6, 5, 8, 8, 9, 8, 8, 8, 12, 10, 12, 12, 7, 5, 5, 5,
                             8, 8, 7, 5, 8,
                             10, 8, 10, 5, 5, 6, 8, 9, 5, 3, 5, 10, 8, 8, 11, 5, 5, 9, 8, 8, 11, 5, 5, 10, 8, 9, 12, 5,
                             5, 8, 8, 9, 8, 5, 5]

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(self.hyper_params):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class DIY_NodeHead(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def _apply_index(self, batch):
        for words in batch.split:
            assert words == batch.split[0]
        mask = '{}_mask'.format(batch.split[0])
        return batch.x[batch[mask]], \
            batch.y[batch[mask]]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # self.node_encoder = AtomEncoder(hidden_size)
        self.node_encoder = NodeEncoder(hidden_size)
        self.edge_encoder = BondEncoder(hidden_size)

    def forward(self, batch):
        # print(batch)
        # print("处理batch：", batch.x.shape)
        # for i in range(batch.x.shape[1]):
        #     print(batch.x[:, i])
        # print("处理attr：", batch.edge_attr.shape)
        # for i in range(batch.edge_attr.shape[1]):
        #     print(batch.edge_attr[:, i])
        batch.x = self.node_encoder(batch.x)
        # batch.edge_attr = self.edge_encoder(batch.edge_attr)
        return batch


class GritTransformer(torch.nn.Module):
    ''' The proposed GritTransformer (Graph Inductive Bias Transformer) '''

    def __init__(self, dim_out,
                 hidden_size=96, ksteps=17, layers_pre_mp=0, n_layers=4, n_heads=4,
                 dropout=0.0, attn_dropout=0.5, ):
        super().__init__()
        self.encoder = FeatureEncoder(hidden_size)
        self.rrwp_abs_encoder = RRWPLinearNodeEncoder(ksteps, hidden_size)
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(ksteps, hidden_size, pad_to_full_graph=True,
                                                      add_node_attr_as_self_loop=False, fill_value=0.)
        self.layers_pre_mp = layers_pre_mp
        if layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(hidden_size, hidden_size, layers_pre_mp)

        layers = [GritTransformerLayer(in_dim=hidden_size, out_dim=hidden_size, num_heads=n_heads,
                                       dropout=dropout, attn_dropout=attn_dropout)
                  for _ in range(n_layers)]
        self.layers = torch.nn.Sequential(*layers)

        # self.post_mp = GNNGraphHead(dim_in=hidden_size, dim_out=dim_out)
        # self.post_mp = GNNNodeHead(dim_in=hidden_size, dim_out=dim_out)
        self.post_mp = DIY_NodeHead(dim_in=hidden_size, dim_out=dim_out)

    def forward(self, batch):
        batch = self.get_embd(batch)
        return self.post_mp(batch)

    def get_embd(self, batch):
        batch = self.encoder(batch)
        batch = self.rrwp_abs_encoder(batch)
        batch = self.rrwp_rel_encoder(batch)
        if self.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        return self.layers(batch)
