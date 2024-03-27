from typing import Optional

import torch

import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing


__all__ = [
    'DMPNN_t'
]

class DMPNN_t(MessagePassing):
    """Implements the Molecular Graph Convolutional Layer (sometimes called Weave) from Steven Kearnes at al. [1]
    as described in [2].

    References
    ----------
    [1] Steven Kearnes at al., Molecular graph convolutions: moving beyond fingerprints, http://dx.doi.org/10.1007/s10822-016-9938-8
    [2] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    """
    def __init__(self, node_in_feats: int, hidden_nodes: int = 64, is_final: bool = False):
        super().__init__(aggr='mean')  # "Add" aggregation
        # -> It has to have the edge dimension to be concat. with the message (sum of edges)
        self.linM = Linear(node_in_feats, node_in_feats)
        self.is_final = is_final

        print('init')


        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.linM.reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        print('x shape: ', x.shape)

        nodes_out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_attr)
        print('nodes out shape: ', nodes_out.shape)

        return nodes_out

    def message(self, x):
        print('message\n----------')
        print(f'current node: {x}')
        print(f'current node shape: {x.shape}')
        if self.is_final:
            return x
        else:
            return x

    def update(self, aggr_out, x):
        print('update\n-----------')
        print(f'current node: {x}')
        print(f'current node shape: {x.shape}')
        #print('message node out: ', aggr_out)
        #print(x.shape)
        # Skip connection does not work
        return x