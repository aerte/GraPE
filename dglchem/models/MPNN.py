# Inspired by https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

import torch

import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from dglchem.models.layers import MPNNLayer

__all__ = [
    'MPNNModel'
]

class MPNNModel(nn.Module):
    def __init__(self, num_layers=4, node_dim=11, edge_dim=4, node_hidden_dim = 64, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(node_dim, node_hidden_dim)

        # Stack of MPNN layers
        self.gnn_layers = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.gnn_layers.append(MPNNLayer(node_in_feats=node_hidden_dim,
                                                   edge_in_feats=edge_dim))
            else:
                self.gnn_layers.append(MPNNLayer(node_in_feats=node_dim,
                                                   edge_in_feats=edge_dim))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(node_hidden_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x)  # (n, d_n) -> (n, d)

        # Add skip connection here
        for conv in self.gnn_layers:
            h = h + conv(h, data.edge_index, data.edge_attr)  # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch)  # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph)  # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)

