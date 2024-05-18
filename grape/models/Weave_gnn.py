# Inspired by https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

# Weave Model

import torch


import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.data import Data, DataLoader

__all__ = [
    'Weave',
    'MGConv'
]

class Weave(MessagePassing):
    """Implements the Molecular Graph Convolutional Layer (sometimes called Weave) from Steven Kearnes at al. [1]
    as described in [2].

    ----

    References

    [1] Steven Kearnes at al., Molecular graph convolutions: moving beyond fingerprints, http://dx.doi.org/10.1007/s10822-016-9938-8

    [2] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    -----

    Parameters
    ------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 64
    edge_hidden_dim: int
        The dimension of the hidden edge features. Default: 64

    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, node_hidden_dim: int = 64, edge_hidden_dim: int = 64):
        super().__init__(aggr='sum')

        self.lin0 = Linear(node_in_dim, edge_in_dim)
        self.lin1 = Linear(edge_in_dim * 2, node_hidden_dim)
        self.lin2 = Linear(edge_in_dim, edge_hidden_dim)
        self.lin3 = Linear(node_in_dim * 2, edge_in_dim)
        self.lin4 = Linear(edge_hidden_dim + edge_in_dim, edge_hidden_dim)

        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> tuple[Tensor,Tensor]:
        """Returns the new node and edge representations computed by the Weave model.

        Parameters
        ------------
        x: Tensor
            The node representation of a singular graph or batch of graphs.
        edge_index: Adj
            The adjacency matrix of x.
        edge_attr: Tensor
            The edge representation of a singular graph or batch of graphs

        Returns
        ---------
        (nodes_out, edges_out)
            The updates node and edge representation.

        """

        nodes_out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_attr)
        edges_out = self.edge_updater(edge_index=edge_index, x=x, edge_attr = edge_attr)
        return nodes_out, edges_out

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out, x):
        inner = torch.cat( (self.relu(self.lin0(x)), aggr_out), dim=1)

        return self.relu(self.lin1(inner))

    def edge_update(self, x_i, x_j, edge_attr):

        cat_1 = self.lin2(edge_attr)
        cat_2 = self.lin3(torch.cat((x_i, x_j),dim=1))
        cat_3 = torch.cat((self.relu(cat_1), self.relu(cat_2)), dim=1)

        return self.relu(self.lin4(cat_3))


class MGConv(nn.Module):
    """Implements the Molecular Graph Convolutional Layer (sometimes called Weave) from Steven Kearnes at al. [1]
    as described in [2].

    ----

    References

    [1] Steven Kearnes at al., Molecular graph convolutions: moving beyond fingerprints, http://dx.doi.org/10.1007/s10822-016-9938-8

    [2] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    ----

    Parameters
    ------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 64
    edge_hidden_dim: int
        The dimension of the hidden edge features. Default: 64
    num_layers: int
        The number of Weave layers that will be used. Default: 4
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, node_hidden_dim: int = 64,
                 edge_hidden_dim: int = 64, num_layers: int = 4):

        super().__init__()

        self.gnn_layers = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer==0:
                self.gnn_layers.append(Weave(node_in_dim=node_in_dim,
                                             edge_in_dim=edge_in_dim,
                                             node_hidden_dim=node_hidden_dim,
                                             edge_hidden_dim=edge_hidden_dim))
            else:
                self.gnn_layers.append(Weave(node_in_dim=node_hidden_dim,
                                             edge_in_dim=edge_hidden_dim,
                                             node_hidden_dim=node_hidden_dim,
                                             edge_hidden_dim=edge_hidden_dim))

    def forward(self, data: Data or DataLoader, only_nodes: bool = True) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        data: Data or DataLoader
            A singular graph Data object or a batch of graphs in the form of a DataLoader object.
        only_nodes: bool
            Decides if just the final node representations are returned. Default: True

        Returns
        -------
        (h, edge_attr)
            The final representations of the nodes and edges respectively.

        """
        edge_attr = data.edge_attr
        h = data.x

        for layer in self.gnn_layers:
            h, edge_attr = layer(h, data.edge_index, edge_attr)

        if only_nodes:
            return h
        else:
            return h, edge_attr
