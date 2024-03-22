from typing import Callable

import torch
import torch.nn as nn
from torch.nn import Linear, Parameter, GRU, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch.nn.init import kaiming_uniform_

__all__ = [
    'MPNNLayer'
]

class MPNNLayer(MessagePassing):
    """Implements the MPNN Layer from Justin Gilmer et al. [1].

    References
    ----------
    [1] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    """
    def __init__(self, edge_in_feats=64, node_in_feats=64, node_out_feats = 64):
        super().__init__(aggr='sum')  # "Add" aggregation
        # shape (emb_dim, emb_dim)
        self.num_edges = edge_in_feats
        self.node_in_dim = node_in_feats
        self.node_out_dim = node_out_feats

        self.mlp = torch.nn.Sequential(
            Linear(in_features=edge_in_feats, out_features=node_in_feats*node_out_feats),
            ReLU(),
            Linear(in_features=node_in_feats*node_out_feats, out_features=node_in_feats*node_out_feats),
            ReLU(),
            Linear(in_features=node_in_feats*node_out_feats, out_features=node_in_feats*node_out_feats)
        )

        self.lin0 = Linear(edge_in_feats, node_in_feats)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin0.reset_parameters()

    def forward(self, x, edge_index, edge_attr, gru):

        nodes_out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_attr, gru=gru)

        return nodes_out

    def message(self, x_j, edge_attr):
        A = self.mlp(edge_attr)
        A = A.view(-1, self.node_in_dim, self.node_out_dim)
        print(A.shape)
        #print(x.shape)
        print(x_j.shape)
        mm = torch.matmul(x_j.unsqueeze(1), A).squeeze(1)
        return mm

    def update(self, aggr_out, x, gru):

        return gru(x.unsqueeze(0), aggr_out).squeeze(0)