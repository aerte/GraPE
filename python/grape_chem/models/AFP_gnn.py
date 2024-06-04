# AFP model
from typing import Union
from torch_geometric.nn import AttentiveFP
from torch.nn import Module
from torch import nn
import torch

__all__ = ['AFP']


class AFP(Module):
    """ A shorthand for the Attentive FP model introduced in [1] and implemented in Pytorch-Geometric [2]. In addition
    to the AFP model, a regression MLP can be added for performance.

    Notes
    ------
    The model consists of two graph attention blocks, 'node' and 'graph'-wise, indicated by the
    ``num_layers_atom`` and ``num_layers_mol`` inputs respectively. Refer to [1] or the summary at the bottom
    for more details.

    Notes
    -------
    If you wish to add global features, then the AFP will return the latent graph representations concatenated
    with the flattened global features, but **will not automatically add an MLP to the end.** This means that
    a separate model has to be built to accommodate regression or classification.


    References:

    [1] Xiong, Z., Wang, D., Liu, X., Zhong, F., Wan, X., Li, X., Li, Z., Luo, X., Chen, K., Jiang, H., & Zheng, M.
    (2020). "Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism"
    https://doi.org/10.1021/acs.jmedchem.9b00959

    [2] https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/attentive_fp.html#AttentiveFP

    [3] Veličković, P., Cucurull, G., Casanova, A., Romero, A., LiĂ ̨, P., & Bengio, Y. (2018). "Graph attention
    networks", https://arxiv.org/abs/1710.10903

    ------

    Parameters
    node_in_dim : int
        The number of node features that are passed to the model.
    edge_in_dim : int
        The number of edge features that are passed to the model.
    out_dim : int or None
        An optional parameter to specify the output dimension of the model. This could be used in the context of
        using the AFP model as a building block in something like the FraGAT model.
    hidden_dim : int
        An optional parameter to specify the hidden dimension of the AFP layer. Defaults: 128.
    num_layers_atom : int
        Number of 'atom'-wise layers in the AFP block. See [1] or the summary below for more details. Default: 3.
    num_layers_mol : int
        Number of 'graph'-wise layers in the AFP block. See [1] or the summary below for more details. Default: 3.
    dropout : float
        The dropout probability for a dropout layer at the end of the AFP block. Default: 0.0.
    regressor: bool
        Decides is a regressor MLP is added to the end. Default: True.
    mlp_out_hidden: int or list
        The number of hidden features should a regressor (3 layer MLP) be added to the end.
         Alternatively, a list of ints can be passed that will be used for an MLP. The
         weights are then used in the same order as given. Default: 512.
    rep_dropout: float
        The probability of dropping a node from the embedding representation. Default: 0.0.
    num_global_feats: int
        The number of global features that are passed to the model. Default:0

    ______

    Short Summary
    --------------
    The input graphs are first embedded, then the graphs are put
    through graph attention [3] layers 'node'-wise, which means that nodes are given individual attention.
    The outputs of those layers are then connected to a virtual node, a representation of the whole 'graph', and put
    through another set of graph attention layers, this time 'graph' or 'molecule'-wise, as only the virtual node is
    given attention.


    """

    def __init__(self, node_in_dim: int, edge_in_dim:int, out_dim:int=None, hidden_dim:int=128,
                 num_layers_atom:int = 3, num_layers_mol:int = 3, dropout:float=0.0,
                 regressor:bool=True, mlp_out_hidden:Union[int,list]=512, rep_dropout:float=0.0,
                 num_global_feats:int = 0):
        super(AFP, self).__init__()

        self.regressor = regressor
        self.num_global_feats = num_global_feats

        self.rep_dropout = nn.Dropout(rep_dropout)

        if out_dim is None or out_dim == 1 or self.regressor:
            self.regressor = True
            if isinstance(mlp_out_hidden, int):
                out_dim = mlp_out_hidden
            else:
                out_dim = mlp_out_hidden[0]


        self.AFP_layers = AttentiveFP(
            in_channels=node_in_dim,
            edge_dim= edge_in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=num_layers_atom,
            num_timesteps=num_layers_mol,
            dropout=dropout,
        )

        if self.regressor:
            if isinstance(mlp_out_hidden, int):
                self.mlp_out = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(mlp_out_hidden + self.num_global_feats, mlp_out_hidden//2),
                    nn.ReLU(),
                    nn.Linear(mlp_out_hidden//2, 1)
                )
            else:
                self.mlp_out = []
                for i in range(len(mlp_out_hidden)):
                    self.mlp_out.append(nn.ReLU())
                    if i == len(mlp_out_hidden)-1:
                        # Added condition if there are global features but only one MLP layer
                        hidden_temp = mlp_out_hidden[i]+self.num_global_feats if i == 0 else mlp_out_hidden[i]
                        self.mlp_out.append(nn.Linear(hidden_temp, 1))
                    elif i == 0:
                        self.mlp_out.append(nn.Linear(mlp_out_hidden[i]+self.num_global_feats, mlp_out_hidden[i + 1]))
                    else:
                        self.mlp_out.append(nn.Linear(mlp_out_hidden[i], mlp_out_hidden[i+1]))
                self.mlp_out = nn.Sequential(*self.mlp_out)
        else:
            self.mlp_out = lambda x: x

    def reset_parameters(self):
        from grape.utils import reset_weights
        self.AFP_layers.reset_parameters()
        if self.regressor:
            reset_weights(self.mlp_out)


    def forward(self, data, return_lats:bool=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        out = self.AFP_layers(x, edge_index, edge_attr, batch)

        if return_lats:
            return out

        # Dropout
        out = self.rep_dropout(out)

        ### Check if global graphs is present for each graph
        if self.num_global_feats > 0:
            out = torch.concat((out, data.global_feats[:,None]), dim=1)


        if self.regressor:
            out = self.mlp_out(out)
        return out.view(-1)