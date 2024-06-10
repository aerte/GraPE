# D-MPNN implementation

from typing import Union
import torch
import torch.nn as nn
import torch.utils.data
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_sum

__all__ = [
    'DMPNNEncoder',
    'DMPNN'
]



class DMPNNEncoder(nn.Module):
    """An implementation of the D-MPNN Model by Yang et al. [1,2], their original code is at
    https://github.com/chemprop/chemprop. This code is written by Ichigaku Takigawa, 2022, under the MIT License,
    and adapted by us. His original code can be found at: https://github.com/itakigawa/pyg_chemprop

    Notes
    -------
    This implementation requires the package torch-scatter to run efficiently. See
    https://github.com/itakigawa/pyg_chemprop/blob/main/pyg_chemprop_naive.py for an implementation without.


    -------

    References

    [1] Yang et al (2019). Analyzing Learned Molecular Representations for Property Prediction. JCIM, 59(8),
    3370–3388. https://doi.org/10.1021/acs.jcim.9b00237

    [2] Yang et al (2019). Correction to Analyzing Learned Molecular Representations for Property Prediction.
    JCIM, 59(12), 5304–5305. https://doi.org/10.1021/acs.jcim.9b01076

    --------

    Parameters
    ----------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 64
    depth: int
        The number of D-MPNN layers that will be used. Default: 4
    dropout: float
        The dropout probability used in a dropout layer after the message update. Default: 0.15
    pool: bool
        Whether to use a pooling layer before returning the output. Default: True

    """

    def __init__(self, node_in_dim:int, edge_in_dim:int, node_hidden_dim:int=64, depth:int=3, dropout:float=0.15,
                 bias:bool=True, pool:bool=True):
        super(DMPNNEncoder, self).__init__()
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_in_dim + edge_in_dim, node_hidden_dim, bias=bias)
        self.W2 = nn.Linear(node_hidden_dim, node_hidden_dim, bias=bias)
        self.W3 = nn.Linear(node_in_dim + node_hidden_dim, node_hidden_dim, bias=bias)
        self.depth = depth
        self.dropout_layer = nn.Dropout(dropout)
        self.pool = pool

    @staticmethod
    def directed_mp(message, edge_index, revedge_index):
        m = scatter_sum(message, edge_index[1], dim=0)
        m_all = m[edge_index[0]]
        m_rev = message[revedge_index]
        return m_all - m_rev

    @staticmethod
    def aggregate_at_nodes(num_nodes, message, edge_index):
        m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
        return m[torch.arange(num_nodes)]

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = self.directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = self.aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        ### Dropout layer
        node_attr = self.dropout_layer(node_attr)

        # readout: pyg global pooling
        if self.pool:
            return global_mean_pool(node_attr, batch)
        else:
            return node_attr


class DMPNN(torch.nn.Module):
    """D-MPNN [1,2] model from graph to prediction. It extends the DMPNNEncoder with a 3 layered MLP for prediction.

    -------

    References

    [1] Yang et al. (2019). Analyzing Learned Molecular Representations for Property Prediction. JCIM, 59(8),
    3370–3388. https://doi.org/10.1021/acs.jcim.9b00237

    [2] Yang et al. (2019). Correction to Analyzing Learned Molecular Representations for Property Prediction.
    JCIM, 59(12), 5304–5305. https://doi.org/10.1021/acs.jcim.9b01076

    --------

    Parameters
    ----------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 64
    depth: int
        The number of D-MPNN layers that will be used. Default: 4
    dropout: float
        The dropout probability used in a dropout layer after the message update. Default: 0.15
    mlp_out_hidden: int or list
            The number of hidden features should a regressor (3 layer MLP) be added to the end.
             Alternatively, a list of ints can be passed that will be used for an MLP. The
             weights are then used in the same order as given. Default: 512.
    rep_dropout: float
        The probability of dropping a node from the embedding representation. Default: 0.0.
    num_global_feats: int
        The number of global features that are passed to the model. Default:0

    """
    def __init__(self, node_in_dim:int, edge_in_dim:int,node_hidden_dim:int=64, depth=3, dropout=0.15,
                        mlp_out_hidden:Union[int, list]=512, rep_dropout:float=0.0, num_global_feats:int=0):
        super(DMPNN, self).__init__()

        self.hidden_size = node_hidden_dim
        self.node_dim = node_in_dim
        self.edge_dim = edge_in_dim
        self.depth = depth

        self.rep_dropout = nn.Dropout(rep_dropout)
        self.num_global_feats = num_global_feats

        self.encoder = DMPNNEncoder(node_in_dim = node_in_dim,
                                    edge_in_dim=edge_in_dim,
                                    depth=depth,
                                    dropout=dropout,
                                    node_hidden_dim=node_hidden_dim,
                                    pool=True)


        if isinstance(mlp_out_hidden, int):
            self.mlp_out = nn.Sequential(
                nn.Linear(self.hidden_size+self.num_global_feats, mlp_out_hidden),
                nn.ReLU(),
                nn.Linear(mlp_out_hidden, mlp_out_hidden // 2),
                nn.ReLU(),
                nn.Linear(mlp_out_hidden // 2, 1)
            )
        else:
            self.mlp_out = []
            self.mlp_out.append(nn.Linear(self.hidden_size+self.num_global_feats, mlp_out_hidden[0]))
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
        reset_weights(self.encoder)

    def forward(self, data, return_lats:bool=False):

        z = self.encoder(data)

        if return_lats:
            return z

        z = self.rep_dropout(z)

        ### Check if global graphs is present for each graph
        if self.num_global_feats > 0:
            z = torch.concat((z, data.global_feats[:, None]), dim=1)

        out = self.mlp_out(z)
        return out.view(-1)



