# Weave layer

import torch

import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch.nn.init import kaiming_uniform_


__all__ = [
    'MGConvLayer'
]

class MGConvLayer(MessagePassing):
    """Implements the Molecular Graph Convolutional Layer (sometimes called Weave) from Steven Kearnes at al. [1]
    as described in [2].

    References
    ----------
    [1] Steven Kearnes at al., Molecular graph convolutions: moving beyond fingerprints, http://dx.doi.org/10.1007/s10822-016-9938-8
    [2] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    """
    def __init__(self, node_in_feats: int, edge_in_feats: int, hidden_nodes: int = 64, hidden_edges: int = 64):
        super().__init__(aggr='sum')  # "Add" aggregation
        # -> It has to have the edge dimension to be concat. with the message (sum of edges)
        self.lin0 = Linear(node_in_feats, edge_in_feats)
        # -> edge_in x2
        self.lin1 = Linear(edge_in_feats*2, hidden_nodes)
        # -> Depends on the number of edges in
        #self.lin2 = Parameter(torch.empty(hidden_edges, edge_in_feats))
        self.lin2 = Linear(edge_in_feats, hidden_edges)
        # -> only works for bidirectional edges (num_nodes = num_edges) ### NEEDS CLARIFICATION
        self.lin3 = Linear(node_in_feats*2, edge_in_feats)
        # shape (emb_dim*2 + edge_dim, edge_dim) For now I have embedding the edges into the emb dim
        self.lin4 = Linear(hidden_edges+edge_in_feats, hidden_edges)

        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        #kaiming_uniform_(self.lin2)
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, x, edge_index, edge_attr):

        #print('x shape: ', x.shape)

        nodes_out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_attr)
        #print('nodes out shape: ', nodes_out.shape)
        edges_out = self.edge_updater(edge_index=edge_index, x=x, edge_attr = edge_attr)


        return nodes_out, edges_out

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out, x):
        #print('message node out: ', aggr_out)
        #print(x.shape)
        # Skip connection does not work
        inner = torch.cat( (self.relu(self.lin0(x)), aggr_out), dim=1)

        return self.relu(self.lin1(inner))

    def edge_update(self, x_i, x_j, edge_attr):

        #print(self.lin2.shape)
        #print(x_i)

        cat_1 = self.lin2(edge_attr)
        #print('cat 1 shape',cat_1.shape)
        #print('xv shape', x_i.shape)
        #print('xw shape', x_j.shape)
        #print('cat 2 shape', torch.cat((x_i, x_j)).shape)
        cat_2 = self.lin3(torch.cat((x_i, x_j),dim=1))
        cat_3 = torch.cat((self.relu(cat_1), self.relu(cat_2)), dim=1)

        return self.relu(self.lin4(cat_3))