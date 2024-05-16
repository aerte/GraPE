# AFP model
from typing import Union
from torch_geometric.nn import AttentiveFP
from torch.nn import Module
from torch import nn
from grape.utils import reset_weights


class AFP(Module):
    """ A shorthand for the Attentive FP model introduced in [1] and implemented in Pytorch-Geometric [2]. In addition
    to the AFP model, a regression MLP can be added for performance.

    Notes
    ------
    The model consists of two graph attention blocks, 'node' and 'graph'-wise, indicated by the
    ``num_layers_atom`` and ``num_layers_mol`` inputs respectively. Refer to [1] or the summary at the bottom
    for more details.


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
                 regressor:bool=True, mlp_out_hidden:Union[int,list]=512):
        super(AFP, self).__init__()

        self.regressor = regressor

        if out_dim is None or out_dim == 1 or regressor:
            self.regressor = True
            if isinstance(mlp_out_hidden, int):
                out_dim = self.mlp_out_hidden
            else:
                out_dim = mlp_out_hidden[0]

        print(out_dim)

        self.AFP_layers = AttentiveFP(
            in_channels=node_in_dim,
            edge_dim= edge_in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=num_layers_atom,
            num_timesteps=num_layers_mol,
            dropout=dropout,
        )

        if regressor:
            if isinstance(mlp_out_hidden, int):
                self.mlp_out = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(mlp_out_hidden, mlp_out_hidden//2),
                    nn.ReLU(),
                    nn.Linear(mlp_out_hidden//2, 1)
                )
            else:
                self.mlp_out = []
                for i in range(len(mlp_out_hidden)):
                    self.mlp_out.append(nn.ReLU())
                    if i == len(mlp_out_hidden)-1:
                        self.mlp_out.append(nn.Linear(mlp_out_hidden[i], 1))
                    else:
                        self.mlp_out.append(nn.Linear(mlp_out_hidden[i], mlp_out_hidden[i+1]))
                self.mlp_out = nn.Sequential(*self.mlp_out)
        else:
            self.mlp_out = lambda x: x

    def reset_parameters(self):
        self.AFP_layers.reset_parameters()
        if self.regressor:
            reset_weights(self.mlp_out)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        out = self.AFP_layers(x, edge_index, edge_attr, batch)
        if self.regressor:
            out = self.mlp_out(out)
        return out.view(-1)