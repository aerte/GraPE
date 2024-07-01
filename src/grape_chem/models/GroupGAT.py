import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.data import Batch

from torch_geometric.nn import AttentiveFP

class SingleHeadOriginLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        # Adjust net_params as necessary for the PyG model
        self.AttentiveEmbedding = AttentiveFP(net_params['in_channels'],
                                              net_params['hidden_channels'],
                                              net_params['out_channels'],
                                              net_params['edge_dim'],
                                              net_params['num_layers'],
                                              net_params['num_timesteps'],
                                              net_params['dropout'])

    def forward(self, data):
        # Assume data contains .x, .edge_index, .edge_attr
        node_features = self.AttentiveEmbedding(data.x, data.edge_index, data.edge_attr)
        graph_features = global_add_pool(node_features, data.batch)  # or global_mean_pool
        return graph_features


class OriginChannel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['in_channels'], bias=True),
            nn.BatchNorm1d(net_params['in_channels']),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['edge_dim'], bias=True),
            nn.BatchNorm1d(net_params['edge_dim']),
            nn.LeakyReLU()
        )
        self.origin_heads = nn.ModuleList([SingleHeadOriginLayer(net_params) for _ in range(net_params['num_heads'])])
        self.origin_attend = nn.Sequential(
            nn.Linear(net_params['num_heads'] * net_params['out_channels'], net_params['out_channels'], bias=True),
            nn.BatchNorm1d(net_params['out_channels']),
            nn.ReLU()
        )

    def reset_parameters(self):
        # Reset parameters in all sub-modules
        for layer in self.embedding_node_lin:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.embedding_edge_lin:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.origin_attend:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for head in self.origin_heads:
            head.reset_parameters()

    def forward(self, batch):
        batch.x = self.embedding_node_lin(batch.x.float())
        batch.edge_attr = self.embedding_edge_lin(batch.edge_attr.float())
        origin_heads_out = [head(batch) for head in self.origin_heads]
        graph_origin = self.origin_attend(torch.cat(origin_heads_out, dim=-1))
        return graph_origin
    
class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AttentiveEmbedding = AttentiveFP(net_params['in_channels'],
                                              net_params['hidden_channels'],
                                              net_params['out_channels'],
                                              net_params['edge_dim'],
                                              net_params['num_layers'],
                                              net_params['num_timesteps'],
                                              net_params['dropout'])

    def forward(self, data):
        node_features = self.AttentiveEmbedding(data.x, data.edge_index, data.edge_attr)
        graph_features = global_add_pool(node_features, data.batch)
        return graph_features

class FragmentChannel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['in_channels'], bias=True),
            nn.BatchNorm1d(net_params['in_channels']),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['edge_dim'], bias=True),
            nn.BatchNorm1d(net_params['edge_dim']),
            nn.LeakyReLU()
        )
        self.fragment_heads = nn.ModuleList([SingleHeadFragmentLayer(net_params) for _ in range(net_params['num_heads'])])
        self.frag_attend = nn.Sequential(
            nn.Linear(net_params['num_heads'] * net_params['out_channels'], net_params['out_channels'], bias=True),
            nn.BatchNorm1d(net_params['out_channels']),
            nn.ReLU()
        )

    def forward(self, batch):
        batch.x = self.embedding_node_lin(batch.x.float())
        batch.edge_attr = self.embedding_edge_lin(batch.edge_attr.float())
        frag_heads_out = [frag_block(batch) for frag_block in self.fragment_heads]
        graph_frag = self.frag_attend(torch.cat(frag_heads_out, dim=-1))
        return graph_frag
    
class SingleHeadJunctionLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.project_motif = nn.Linear(net_params['L2_hidden_dim'] + net_params['L3_hidden_dim'], net_params['L3_hidden_dim'], bias=True)
        self.AttentiveEmbedding = AttentiveFP(net_params['L3_hidden_dim'],
                                              net_params['L3_hidden_channels'],
                                              net_params['L3_out_channels'],
                                              net_params['edge_dim'],
                                              net_params['L3_layers'],
                                              net_params['L3_timesteps'],
                                              net_params['L3_dropout'])

    def forward(self, data):
        data.x = self.project_motif(data.x)
        node_features = self.AttentiveEmbedding(data.x, data.edge_index, data.edge_attr)
        graph_features, attention_weights = global_add_pool(node_features, data.batch), None  # Adjust if you get attention weights
        return graph_features, attention_weights

class JT_Channel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_frag_lin = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['L3_hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['L3_hidden_dim']),
            nn.LeakyReLU()
        )
        self.junction_heads = nn.ModuleList([SingleHeadJunctionLayer(net_params) for _ in range(net_params['num_heads'])])

    def forward(self, batch):
        batch.x = self.embedding_frag_lin(batch.x)
        junction_graph_heads_out = []
        junction_attention_heads_out = []
        for single_head in self.junction_heads:
            single_head_new_graph, single_head_attention_weight = single_head(batch)
            junction_graph_heads_out.append(single_head_new_graph)
            junction_attention_heads_out.append(single_head_attention_weight)
        super_new_graph = torch.relu(torch.mean(torch.stack(junction_graph_heads_out, dim=1), dim=1))
        super_attention_weight = torch.mean(torch.stack(junction_attention_heads_out, dim=1), dim=1)
        return super_new_graph, super_attention_weight
    
class GCGAT_v4pro(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.origin_module = OriginChannel(net_params)
        self.frag_module = FragmentChannel(net_params)
        self.junction_module = JT_Channel(net_params)

        # assuming net_params includes dimensions for concatenated outputs (does it?)
        concat_dim = net_params['L1_hidden_dim'] + net_params['L2_hidden_dim'] + net_params['L3_hidden_dim']
        self.linear_predict1 = nn.Sequential(
            nn.Dropout(net_params['final_dropout']),
            nn.Linear(concat_dim, int(concat_dim / 2), bias=True),
            nn.LeakyReLU(negative_slope=1e-7),
            nn.BatchNorm1d(int(concat_dim / 2)),
        )

        self.linear_predict2 = nn.Sequential()
        mid_dim = int(concat_dim / 2)
        for _ in range(net_params['MLP_layers'] - 1):
            self.linear_predict2.append(nn.Linear(mid_dim, mid_dim, bias=True))
            self.linear_predict2.append(nn.LeakyReLU(negative_slope=1e-7))
        self.linear_predict2.append(nn.Linear(mid_dim, 1, bias=True))
        self.linear_predict2.append(nn.LeakyReLU(negative_slope=1e-7))

    def forward(self, origin_data, frag_data, junction_data):
        # Approach:
        # 1. extract graph-level features from different channels
        graph_origin = self.origin_module(origin_data)
        graph_frag = self.frag_module(frag_data)
        super_new_graph, super_attention_weight = self.junction_module(junction_data)

        # 2. concat the output from different channels
        concat_features = torch.cat([graph_origin, graph_frag, super_new_graph], dim=-1)
        descriptors = self.linear_predict1(concat_features)
        output = self.linear_predict2(descriptors)
        return output

    def reset_parameters(self):
        self.origin_module.reset_parameters()
        self.frag_module.reset_parameters()
        self.junction_module.reset_parameters()
        for layer in self.linear_predict1:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.linear_predict2:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()