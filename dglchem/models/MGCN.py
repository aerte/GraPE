# Implements the Molecular Graph Convolutional Layer from Steven Kearnes et alhttp://dx.doi.org/10.1007/s10822-016-9938-8

# Inspired by https://github.com/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

import torch

import torch.nn as nn
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch.nn.init import kaiming_uniform_
from torch_geometric.nn import global_mean_pool

__all__ = [
    'MGConv',
    'MGConvLayer'
]


class MGConv(nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, node_dim=11, edge_dim=4, out_dim=1):
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
        self.lin_in = Linear(node_dim, emb_dim)
        print(self.lin_in)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MGConvLayer(emb_dim, edge_dim))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x)  # (n, d_n) -> (n, d)

        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr)  # (n, d) -> (n, d)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch)  # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph)  # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)

class MGConvLayer(MessagePassing):
    """Implements teh Molecular Graph Convolutional Layer from Steven Kearnes at al. [1] as described in [2].

    References
    ----------
    [1] Steven Kearnes at al., Molecular graph convolutions: moving beyond fingerprints, http://dx.doi.org/10.1007/s10822-016-9938-8
    [2] Justin Gilmer et al., Neural Message Passing for Quantum Chemistry, http://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

    """
    def __init__(self, emb_dim, edge_dim):
        super().__init__(aggr='sum')  # "Add" aggregation
        # shape (emb_dim, emb_dim)
        self.lin0 = Linear(emb_dim, edge_dim)
        # shape (emb_dim + edge_dim, emb_dim)
        self.lin1 = Linear(edge_dim, emb_dim)
        # shape (emb_dim, emb_dim)
        self.lin2 = Parameter(torch.empty(emb_dim,edge_dim))
        # shape (emb_dim*2, emb_dim)
        self.lin3 = Linear(emb_dim*2, emb_dim)
        # shape (emb_dim*2 + edge_dim, edge_dim) For now I have embedding the edges into the emb dim
        self.lin4 = Linear(emb_dim+edge_dim, emb_dim)

        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        kaiming_uniform_(self.lin2)
        self.lin3.reset_parameters()
        self.lin4.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)

        # Step 3: Compute normalization.
        #row, col = edge_index
        #deg = degree(col, x.size(0), dtype=x.dtype)
        #deg_inv_sqrt = deg.pow(-0.5)
        #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        nodes_out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_attr)
        edges_out = self.edge_updater(edge_index=edge_index, x=x, edge_attr = edge_attr)

        # Step 6: Apply a final bias vector.
        #out = out + self.bias

        return nodes_out, edges_out

    def message(self, edge_attr):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return edge_attr

    #def aggregate(self, inputs):
    #    print(inputs)
    #    print(torch.sum(inputs))
    #    return torch.sum(inputs)

    def update(self, aggr_out, x):
        print(aggr_out)
        inner = torch.cat( (self.relu(self.lin0(x)), aggr_out))

        return self.relu(self.lin1(inner))

    def edge_update(self, x_i, x_j, edge_attr):

        print(self.lin2.shape)

        cat_1 = torch.cat((self.lin2, edge_attr))
        cat_2 = self.lin3(torch.cat((x_i, x_j)))
        cat_3 = torch.cat((self.relu(cat_1), self.relu(cat_2)))

        return self.relu(self.lin4(cat_3))
