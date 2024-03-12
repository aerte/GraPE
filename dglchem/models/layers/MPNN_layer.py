import torch
import torch.nn as nn
from torch.nn import Linear, Parameter, GRU
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
    def __init__(self, edge_in_feats=64, node_in_feats=64):
        super().__init__(aggr='sum')  # "Add" aggregation
        # shape (emb_dim, emb_dim)
        self.lin0 = Linear(edge_in_feats, node_in_feats)

        self.gru = nn.GRU(node_in_feats, node_in_feats)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin0.reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        nodes_out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_attr)

        return nodes_out

    def message(self, x, edge_attr):
        mm = self.lin0(edge_attr)*x
        return mm

    def update(self, aggr_out, x):

        return self.gru(x, aggr_out)