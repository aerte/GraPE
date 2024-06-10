# Inspired by https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

from typing import Union
import torch
import torch.nn as nn
from torch.nn import Linear, GRU, ReLU

from torch_geometric.nn import NNConv
from torch_geometric.nn.aggr import Set2Set

__all__ = [
    'MPNNEncoder',
    'MPNN'
]

class MPNNEncoder(nn.Module):
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

        self.relu = ReLU()
        self.gru = GRU(input_size=node_hidden_dim, hidden_size=node_hidden_dim, num_layers=1,
                       batch_first=False)

        self.mlp_in = torch.nn.Sequential(
            Linear(in_features=node_in_dim, out_features=node_hidden_dim),
            ReLU()
        )

        self.mlp = torch.nn.Sequential(
            Linear(in_features=edge_in_dim, out_features=node_hidden_dim),
            ReLU(),
            Linear(in_features=node_hidden_dim, out_features=node_hidden_dim * node_hidden_dim)
        )

        # Stack of MPNN layers
        self.gnn_layers = torch.nn.ModuleList()
        for layer in range(num_layers):
             self.gnn_layers.append(NNConv(in_channels=node_hidden_dim, out_channels=node_hidden_dim, nn=self.mlp,
                                           aggr='mean'))

        self.reset_parameters()

    def reset_parameters(self):
        from grape_chem.utils import reset_weights
        reset_weights(self.mlp_in)
        reset_weights(self.mlp)
        reset_weights(self.gru)
        reset_weights(self.gnn_layers)

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
        hidden_gru = h.unsqueeze(0)

        for layer in self.gnn_layers:
            m_v = self.relu(layer(h, data.edge_index, data.edge_attr))
            h, hidden_gru = self.gru(m_v.unsqueeze(0), hidden_gru)
            h = h.squeeze(0)

        return h

class MPNN(nn.Module):
    """Implements the complete MPNN model described by Justin Gilmer et al. in [1]. It uses a simple, 2 layered MLP as
    the free neural network that projects the edge features. The MPNN layer itself is the native pytorch geometric
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
    num_layers: int
        The number of MPNN layers that will be used. Default: 4
    num_gru_layers: int
        The number of GRU layers that will be used for each layer. Default: 4
    set2set_steps: int
        The number of set2set pooling steps/iterations that will be used. Default: 1
    mlp_out_hidden: int or list
            The number of hidden features should a regressor (3 layer MLP) be added to the end.
             Alternatively, a list of ints can be passed that will be used for an MLP. The
             weights are then used in the same order as given. Default: 512.
    rep_dropout: float
        The probability of dropping a node from the embedding representation. Default: 0.0.
    num_global_feats: int
        The number of global features that are passed to the model. Default:0

    """

    def __init__(self, node_in_dim: int, edge_in_dim: int, message_nn: nn.Module = None,
                 node_hidden_dim: int = 64, num_layers: int = 1, num_gru_layers: int = 1,
                 set2set_steps: int = 1, mlp_out_hidden:Union[int, list]=512,
                 rep_dropout: float = 0.0, num_global_feats:int = 0):

        super().__init__()

        if message_nn is None:
            message_nn = MPNNEncoder(node_in_dim, edge_in_dim, node_hidden_dim, num_layers, num_gru_layers)
        self.message = message_nn

        self.read_out = Set2Set(in_channels=node_hidden_dim, processing_steps=set2set_steps)
        self.rep_dropout = nn.Dropout(rep_dropout)
        self.num_global_feats = num_global_feats

        if isinstance(mlp_out_hidden, int):
            self.mlp_out = nn.Sequential(
                nn.Linear(node_hidden_dim *2 + self.num_global_feats, mlp_out_hidden),
                nn.ReLU(),
                nn.Linear(mlp_out_hidden, mlp_out_hidden // 2),
                nn.ReLU(),
                nn.Linear(mlp_out_hidden // 2, 1)
            )

        else:
            self.mlp_out = []
            self.mlp_out.append(nn.Linear(node_hidden_dim * 2 + self.num_global_feats, mlp_out_hidden[0]))
            for i in range(len(mlp_out_hidden)):
                self.mlp_out.append(nn.ReLU())
                if i == len(mlp_out_hidden) - 1:
                    self.mlp_out.append(nn.Linear(mlp_out_hidden[i], 1))
                else:
                    self.mlp_out.append(nn.Linear(mlp_out_hidden[i], mlp_out_hidden[i + 1]))
            self.mlp_out = nn.Sequential(*self.mlp_out)

        self.reset_parameters()

    def reset_parameters(self):
        from grape_chem.utils import reset_weights
        reset_weights(self.mlp_out)
        reset_weights(self.read_out)
        reset_weights(self.message)

    def forward(self, data, return_lats:bool=False):
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
        h = self.read_out(h_t, data.batch)

        if return_lats:
            return h

        h = self.rep_dropout(h)

        if self.num_global_feats > 0:
            h = torch.concat((h, data.global_feats[:,None]), dim=1)

        y = self.mlp_out(h)

        return y.view(-1)



