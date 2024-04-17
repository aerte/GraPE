# Inspired by https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

import torch

import torch.nn as nn
from torch.nn import Linear, GRU, ReLU

from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import Set2Set

__all__ = [
    'MPNN',
    'MPNN_Model'
]

class MPNN(nn.Module):
    """Implements the MPNN network described by Justin Gilmer et al. in [1]. It uses a simple, 3 layered MLP as the
    free neural network that projects the edge features. The MPNN layer itself is the native pytorch geometric
    implementation (NNConv).

    ----

    References

    [1] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    ----

    Parameters
    ----------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 64
    edge_hidden_dim: int
        The dimension of the hidden edge features. Default: 64
    num_layers: int
        The number of MPNN layers that will be used. Default: 4
    num_gru_layers: int
        The number of GRU layers that will be used for each layer. Default: 4

    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, node_hidden_dim: int = 64,
                 num_layers: int = 1, num_gru_layers: int = 1):

        super().__init__()

        self.gru = GRU(input_size=node_hidden_dim, hidden_size=node_hidden_dim, num_layers=num_gru_layers,
                       batch_first=False)

        self.mlp_in = torch.nn.Sequential(
            Linear(in_features=node_in_dim, out_features=node_hidden_dim),
            ReLU(),
            Linear(in_features=node_hidden_dim, out_features=node_hidden_dim),
            ReLU(),
            Linear(in_features=node_hidden_dim, out_features=node_hidden_dim)
        )

        self.mlp = torch.nn.Sequential(
            Linear(in_features=edge_in_dim, out_features=node_hidden_dim),
            ReLU(),
            Linear(in_features=node_hidden_dim, out_features=node_hidden_dim),
            ReLU(),
            Linear(in_features=node_hidden_dim, out_features=node_hidden_dim * node_hidden_dim)
        )

        # Stack of MPNN layers
        self.gnn_layers = torch.nn.ModuleList()
        for layer in range(num_layers):
             self.gnn_layers.append(NNConv(in_channels=node_hidden_dim, out_channels=node_hidden_dim, nn=self.mlp))

    def forward(self, data):
        """
        Parameters
        ------------
        data: Data or DataLoader
            A singular graph Data object or a batch of graphs in the form of a DataLoader object.

        Returns
        ---------
        h: Tensor
            Returns the final hidden node representation of shape (num nodes, nodes_hidden_features).
        """

        h = self.mlp_in(data.x)
        hidden_gru  = None

        for layer in self.gnn_layers:
            m_v = layer(h, data.edge_index, data.edge_attr)
            if hidden_gru is None:
                h, hidden_gru = self.gru(m_v.unsqueeze(0))
            else:
                h, hidden_gru = self.gru(m_v.unsqueeze(0), hidden_gru)
            h = h.squeeze(0)

        return h

class MPNN_Model(nn.Module):
    """Implements the complete MPNN model described by Justin Gilmer et al. in [1]. It uses a simple, 2 layered MLP as the
    free neural network that projects the edge features. The MPNN layer itself is the native pytorch geometric
    implementation (NNConv), the readout layer is set2set and the output model is another MLP.

    ----

    References

    [1] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    ----

    Parameters
    ----------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 64
    edge_hidden_dim: int
        The dimension of the hidden edge features. Default: 64
    num_layers: int
        The number of MPNN layers that will be used. Default: 4
    num_gru_layers: int
        The number of GRU layers that will be used for each layer. Default: 4

    """

    def __init__(self, node_in_dim: int, edge_in_dim: int, message_nn: nn.Module = None,
                 node_hidden_dim: int = 64, num_layers: int = 1, num_gru_layers: int = 1,
                 set2set_layers: int = 1):

        super().__init__()

        if message_nn is None:
            message_nn = MPNN(node_in_dim, edge_in_dim, node_hidden_dim, num_layers, num_gru_layers)
        self.message = message_nn

        self.read_out = Set2Set(in_channels=node_hidden_dim, processing_steps=set2set_layers)

        self.mlp_out = torch.nn.Sequential(
            Linear(in_features=node_hidden_dim*2, out_features=node_hidden_dim),
            ReLU(),
            Linear(in_features=node_hidden_dim, out_features=node_hidden_dim),
            ReLU(),
            Linear(in_features=node_hidden_dim, out_features=1)
        )




    def forward(self, data):
        """
        Parameters
        ------------
        data: Data or DataLoader
            A singular graph Data object or a batch of graphs in the form of a DataLoader object.

        Returns
        ---------
        h: Tensor
            Returns the final hidden node representation of shape (num nodes, nodes_hidden_features).
        """

        h_t = self.message(data)
        h = self.read_out(h_t, dim_size=data.batch_size)
        y = self.mlp_out(h)

        return y.flatten()



