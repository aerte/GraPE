# Inspired by https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

import torch

import torch.nn as nn
from torch import Tensor
from dglchem.models.layers import Weave
from torch_geometric.data import Data, DataLoader

__all__ = [
    'MGConv'
]


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
