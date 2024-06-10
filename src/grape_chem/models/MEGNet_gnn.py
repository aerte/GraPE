# MEGNet
# Implementation inspired by https://github.com/deepchem/deepchem/blob/28195eb49b9962ecc81d47eb87a82dbafc36c5b2/deepchem/models/torch_models/layers.py#L1063
#

from typing import Union
import torch


import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Set2Set
from torch_geometric.typing import Adj
from torch_scatter import scatter


__all__ = [
    'MEGNet_block',
    'MEGNet'
]


class MEGNet_block(nn.Module):
    """An implementation of the MatErials Graph Network (MEGNet) block by Chen et al. [1]. The MEGNet block
    operates on three different types of input graphs: nodes, edges and global variables. The global
    variables are *graph*-wise, for example, the melting point of a molecule or any other graph level information.

    Note that each element-wise update function is a three-depth MLP, so not many blocks are needed for good
    performance.

    References
    -----------
    [1] Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. (2019). "Graph networks as a universal machine learning
    framework for molecules and crystals". Chemistry of Materials, 31 (9), 3564â3572.
    https://doi.org/10.1021/acs.chemmater.9b01294

    ------

    Parameters
    ----------------
    node_in_dim: int
        The number of input node features.
    edge_in_dim: int
        The number of input edge features.
    global_in_dim:int
        The number of global input features.
    node_hidden_dim: int
        The dimension of the hidden node features. Default: 32
    edge_hidden_dim: int
        The dimension of the hidden edge features. Default: 32


    """


    def __init__(self, node_in_dim: int, edge_in_dim:int, global_in_dim:int, node_hidden_dim: int, edge_hidden_dim: int,
                 device = torch.device("cpu")):
        super(MEGNet_block, self).__init__()

        self.device = device

        self.update_net_edge = nn.Sequential(
            nn.Linear(node_in_dim * 2 + edge_in_dim + global_in_dim, edge_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim, bias=True),
        )

        self.update_net_node = nn.Sequential(
            nn.Linear(node_in_dim + edge_hidden_dim + global_in_dim, node_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(node_hidden_dim, node_hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(node_hidden_dim, node_hidden_dim, bias=True),
        )

        self.update_net_global = nn.Sequential(
            nn.Linear(node_hidden_dim + edge_hidden_dim + global_in_dim, global_in_dim, bias=True),
            nn.Softplus(),
            nn.Linear(global_in_dim, global_in_dim, bias=True),
            nn.Softplus(),
            nn.Linear(global_in_dim, global_in_dim, bias=True),
        )

    # def reset_parameters(self) -> None:
    #     from grape_chem.utils import reset_weights
    #     reset_weights(self.update_net_edge)
    #     reset_weights(self.update_net_node)
    #     reset_weights(self.update_net_global)

    def update_edge_feats(self, edge_index: Adj, node_feats, edge_feats, global_feats,
                          batch) -> Tensor:
        src_index, dst_index = edge_index

        out = torch.cat((node_feats[src_index], node_feats[dst_index],
                         edge_feats, global_feats[batch]), dim=1)

        out = self.update_net_edge(out)

        return out


    def update_node_feats(self, edge_index: Adj, node_feats, edge_feats, global_feats, batch):
        src_index, dst_index = edge_index
        # Compute mean edge features for each node by dst_index (each node
        # receives information from edges which have that node as its destination,
        # hence the computation uses dst_index to aggregate information)
        edge_features_mean_by_node = scatter(src=edge_feats,
                                             index=dst_index,
                                             dim=0,
                                             reduce='mean')
        out = torch.cat(
            (node_feats, edge_features_mean_by_node, global_feats[batch]),dim=1)

        out = self.update_net_node(out)

        return out

    def update_global_features(self, node_feats, edge_feats,
                                global_feats, node_batch_map,
                                edge_batch_map):

        edge_features_mean = scatter(src=edge_feats,
                                     index=edge_batch_map,
                                     dim=0,
                                     reduce='mean')

        node_features_mean = scatter(src=node_feats,
                                     index=node_batch_map,
                                     dim=0,
                                     reduce='mean')

        out = torch.cat((edge_features_mean, node_features_mean, global_feats), dim=1)

        out = self.update_net_global(out)

        return out


    def forward(self, edge_index, x, edge_attr, global_feats, batch=None):

        if batch is None:
            batch = x.new_zeros(x.size(0),dtype=torch.int64).to(self.device)

        edge_batch_map = batch[edge_index[0]]
        h_e = self.update_edge_feats(node_feats=x, edge_index=edge_index, edge_feats=edge_attr,
                                     global_feats=global_feats, batch=edge_batch_map)

        h_n = self.update_node_feats(node_feats=x, edge_index=edge_index, edge_feats=h_e,
                                     global_feats=global_feats, batch=batch)

        h_u = self.update_global_features(node_feats=h_n, node_batch_map=batch,
                                          edge_batch_map=edge_batch_map, edge_feats=h_e, global_feats=global_feats)



        return h_e, h_n, h_u



class MEGNet(nn.Module):
    """An implementation of the **MatErials Graph Network (MEGNet)** block by Chen et al. [1]. The MEGNet block
        operates on three different types of input graphs: nodes, edges and global variables. The global
        variables are *graph*-wise, for example, the melting point of a molecule or any other graph level information.

        Note that each element-wise update function is a three-depth MLP, so not many blocks are needed for good
        performance.

        Notes
        ------
        While the documentation specifies that the model needs to be feed the global features,
        if *None* is passed to the forward function, a random feature vector of the correct size will be
        generated.


        References
        -----------
        [1] Chen, C., Ye, W., Zuo, Y., Zheng, C., & Ong, S. P. (2019). "Graph networks as a universal machine learning
        framework for molecules and crystals". Chemistry of Materials, 31 (9), 3564â3572.
        https://doi.org/10.1021/acs.chemmater.9b01294

        ------

        Parameters
        ----------------
        node_in_dim: int
            The number of input node features.
        edge_in_dim: int
            The number of input edge features.
        global_in_dim:int
            The number of global input features.
        node_hidden_dim: int
            The dimension of the hidden node features. Default: 32
        edge_hidden_dim: int
            The dimension of the hidden edge features. Default: 32
        global_hidden_dim: int
            The dimension of the hidden global features. Default: 32
        depth: int
            The number of consecutive MEGNet blocks to be used. Default: 3
        mlp_out_hidden: int or list
            The number of hidden features should a regressor (3 layer MLP) be added to the end.
             Alternatively, a list of ints can be passed that will be used for an MLP. The
             weights are then used in the same order as given. Default: 512.
        rep_dropout: float
            The probability of dropping a node from the embedding representation. Default: 0.0.
        device: torch.device
            The device on which the code is run. Default: 'cpu'
    """


    def __init__(self, node_in_dim: int, edge_in_dim: int, global_in_dim: int=32, node_hidden_dim: int=64,
                 edge_hidden_dim: int=64, global_hidden_dim:int=32, depth:int=2, mlp_out_hidden:Union[int, list]=512,
                 rep_dropout:float=0.0 ,device = torch.device("cpu")):
        super(MEGNet, self).__init__()

        self.device = device
        self.depth = depth
        self.global_in_dim = global_in_dim

        self.embed_nodes = nn.Linear(node_in_dim, node_hidden_dim)
        self.embed_edges = nn.Linear(edge_in_dim, edge_hidden_dim)
        self.embed_global = nn.Linear(global_in_dim, global_hidden_dim)

        self.rep_dropout = nn.Dropout(rep_dropout)

        self.dense_layers_nodes = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_hidden_dim, node_hidden_dim*2),
                nn.ReLU(),
                nn.Linear(node_hidden_dim*2, node_hidden_dim),
            ) for _ in range(depth)])

        self.dense_layers_edges = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_hidden_dim, edge_hidden_dim * 2),
                nn.Linear(edge_hidden_dim * 2, edge_hidden_dim),
            ) for _ in range(depth)])

        self.dense_layers_global = nn.ModuleList([
            nn.Sequential(
                nn.Linear(global_hidden_dim, global_hidden_dim * 2),
                nn.Linear(global_hidden_dim * 2, global_hidden_dim),
            ) for _ in range(depth)])

        if isinstance(mlp_out_hidden, int):
            self.mlp_out = nn.Sequential(
                nn.Linear(node_hidden_dim * 2 + edge_hidden_dim * 2 + global_hidden_dim, mlp_out_hidden),
                nn.ReLU(),
                nn.Linear(mlp_out_hidden, mlp_out_hidden // 2),
                nn.ReLU(),
                nn.Linear(mlp_out_hidden // 2, 1)
            )
        else:
            self.mlp_out = []
            self.mlp_out.append(nn.Linear(node_hidden_dim * 2 + edge_hidden_dim * 2 + global_hidden_dim,
                                          mlp_out_hidden[0]))
            for i in range(len(mlp_out_hidden)):
                self.mlp_out.append(nn.ReLU())
                if i == len(mlp_out_hidden) - 1:
                    self.mlp_out.append(nn.Linear(mlp_out_hidden[i], 1))
                else:
                    self.mlp_out.append(nn.Linear(mlp_out_hidden[i], mlp_out_hidden[i + 1]))
            self.mlp_out = nn.Sequential(*self.mlp_out)

        self.read_out_nodes = Set2Set(node_hidden_dim, processing_steps=3)
        self.read_out_edges = Set2Set(edge_hidden_dim, processing_steps=3)

        self.blocks = nn.ModuleList([
            MEGNet_block(node_in_dim=node_hidden_dim,
                         edge_in_dim=edge_hidden_dim,
                         global_in_dim=global_hidden_dim,
                         node_hidden_dim=node_hidden_dim,
                         edge_hidden_dim=edge_hidden_dim,
                         device=self.device)
            for _ in range(depth)
        ])

    def reset_parameters(self) -> None:
        from grape_chem.utils import reset_weights
        reset_weights(self.read_out_nodes)
        reset_weights(self.read_out_edges)
        reset_weights(self.mlp_out)
        reset_weights(self.dense_layers_nodes)
        for i in range(self.depth):
            reset_weights(self.dense_layers_edges[i])
            reset_weights(self.dense_layers_nodes[i])
            reset_weights(self.dense_layers_global[i])
            reset_weights(self.blocks[i])


    def forward(self, data, return_lats:bool = False) -> Tensor:

        x, edge_index, edge_attr, global_feats = data.x, data.edge_index, data.edge_attr, data.global_feats

        h_n = self.embed_nodes(x)
        h_e = self.embed_edges(edge_attr)
        h_u = self.embed_global(global_feats[:,None])

        for i in range(self.depth):
            h_n = self.dense_layers_nodes[i](h_n)
            h_e = self.dense_layers_edges[i](h_e)
            h_u = self.dense_layers_global[i](h_u)


            h_e, h_n, h_u = self.blocks[i](edge_index, h_n, h_e, h_u, data.batch)


        src_index, dst_index = edge_index
        h_n = self.read_out_nodes(h_n, data.batch)
        h_e = self.read_out_edges(h_e, data.batch[src_index])

        if return_lats:
            return h_n

        out = torch.concat((h_n, h_e, h_u), dim=1)

        out = self.rep_dropout(out)

        return self.mlp_out(out).view(-1)









