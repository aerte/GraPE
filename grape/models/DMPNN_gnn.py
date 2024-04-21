# Inspired by https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

from typing import Any

import torch


import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size
from .Weave_gnn import Weave
from torch_geometric.nn.pool import global_mean_pool

__all__ = [
    'DMPNN_layer',
    'DMPNN'
]


class DMPNN_layer(MessagePassing):
    """

    """
    def __init__(self,h_0=None, update_nn=None, hidden_size:int = 64, is_final: bool = False):
        super().__init__(aggr='sum') # "Add" aggregation
        # -> It has to have the edge dimension to be concat. with the message (sum of edges)
        if is_final:
            self.linA = Linear(hidden_size, hidden_size)
        self.linM = Linear(hidden_size, hidden_size)

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

    def edge_updater(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> Tensor:
        print('edge updater')
        print('--------------')

    # def message(self, x_i, x_j, edge_attr):
    #     print('message\n----------')
    #     print(f'current node: {x_i}')
    #     print(f'current node shape: {x_j}')
    #
    #     return x_i
    #
    # def update(self, aggr_out, x):
    #     print('update\n-----------')
    #     print(f'current aggr: {aggr_out}')
    #     print(f'current node shape: {aggr_out.shape}')
    #     #print('message node out: ', aggr_out)
    #     #print(x.shape)
    #     # Skip connection does not work
    #     return aggr_out

class DMPNN(nn.Module):
    def __init__(self, num_layers=4, node_dim=11, edge_dim=4, node_hidden_dim = 64,
                 edge_hidden_dim = 64, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        self.gnn_layers = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer==0:
                self.gnn_layers.append(Weave(node_in_feats=node_dim,
                                                   edge_in_feats=edge_dim,
                                                   hidden_nodes=node_hidden_dim,
                                                   hidden_edges=edge_hidden_dim))
            else:
                self.gnn_layers.append(Weave(node_in_feats=node_hidden_dim,
                                                   edge_in_feats=edge_hidden_dim,
                                                   hidden_nodes=node_hidden_dim,
                                                   hidden_edges=edge_hidden_dim))


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
        edge_attr = data.edge_attr

        not_first_it = False
        #i = 0

        h = data.x

        for layer in self.gnn_layers:

            if not_first_it:
                h_temp, edge_attr = layer(h, data.edge_index, edge_attr)
                # Skip connection
                #h = h + h_temp
                #i+=1
                #print(i)
            else:
                h, edge_attr = layer(h, data.edge_index, edge_attr)
                #print('Edge shape after one iteration: ', edge_attr.shape)
                #print('Node shape after one iteration: ', h.shape)
                not_first_it = True
                #i+=1
                #print(i)

            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch)  # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph)  # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)