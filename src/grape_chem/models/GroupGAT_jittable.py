import torch
import torch.nn as nn
from typing import List, Tuple

from grape_chem.models.AFP_gnn_jittable import AFP_jittable as AFP  # Assuming AFP is TorchScript compatible

__all__ = ['GCGAT_v4pro_jit']

class SingleHeadOriginLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AttentiveEmbedding = AFP(
            node_in_dim=net_params["L1_hidden_dim"],
            edge_in_dim=net_params["L1_hidden_dim"],
            hidden_dim=net_params['hidden_dim'],
            num_layers_atom=net_params['L1_layers_atom'],
            num_layers_mol=net_params['L1_layers_mol'],
            dropout=net_params['dropout'],
            out_dim=net_params['L1_hidden_dim'],
            regressor=False,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.AttentiveEmbedding(x, edge_index, edge_attr, batch, return_lats = True,)

class OriginChannel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(net_params['num_atom_type'], net_params['L1_hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['L1_hidden_dim']),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['L1_hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['L1_hidden_dim']),
            nn.LeakyReLU()
        )
        self.origin_heads = nn.ModuleList([SingleHeadOriginLayer(net_params) for _ in range(net_params['num_heads'])])
        self.origin_attend = nn.Sequential(
            nn.Linear(net_params['num_heads'] * net_params['L1_hidden_dim'], net_params['L1_hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['L1_hidden_dim']),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, batch):
        embedded_x = self.embedding_node_lin(x)
        embedded_edge_attr = self.embedding_edge_lin(edge_attr)
        origin_heads_out = []
        for head in self.origin_heads:
            out = head(embedded_x, edge_index, embedded_edge_attr, batch)
            origin_heads_out.append(out)
        origin_heads_out_cat = torch.cat(origin_heads_out, dim=-1)
        graph_origin = self.origin_attend(origin_heads_out_cat)
        return graph_origin

class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AttentiveEmbedding = AFP(node_in_dim=net_params["node_in_dim"],
                                      edge_in_dim=net_params["edge_in_dim"],
                                        hidden_dim=net_params['hidden_dim'],
                                        num_layers_atom=net_params['L2_layers_atom'],
                                        num_layers_mol=net_params['L2_layers_mol'],
                                        dropout=net_params['dropout'],
                                        out_dim=net_params['L2_hidden_dim'],
                                        regressor=False,
                                    )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.AttentiveEmbedding(x, edge_index, edge_attr, batch, return_lats=True)

class FragmentChannel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_node_lin = nn.Sequential(
            nn.Linear(net_params["num_atom_type"], net_params["L2_hidden_dim"], bias=True),
            nn.BatchNorm1d(net_params["L2_hidden_dim"]),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params["L2_hidden_dim"], bias=True),
            nn.BatchNorm1d(net_params["L2_hidden_dim"]),
            nn.LeakyReLU()
        )
        self.fragment_heads = nn.ModuleList([SingleHeadFragmentLayer(net_params) for _ in range(net_params['num_heads'])])
        self.frag_attend = nn.Sequential(
            nn.Linear(net_params['num_heads'] * net_params['L2_hidden_dim'], net_params['L2_hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['L2_hidden_dim']),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr, batch):
        #breakpoint()
        frag_heads_out = []
        for frag_block in self.fragment_heads:
            out = frag_block(x, edge_index, edge_attr, batch)
            frag_heads_out.append(out)
        frag_heads_out_cat = torch.cat(frag_heads_out, dim=-1)
        #breakpoint()
        graph_frag = self.frag_attend(frag_heads_out_cat)
        return graph_frag

class SingleHeadJunctionLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.project_motif = nn.Linear(net_params['L2_hidden_dim'] + net_params['L3_hidden_dim'], net_params['L3_hidden_dim'], bias=True)
        self.AttentiveEmbedding = AFP(
            node_in_dim=net_params["L3_hidden_dim"],
            edge_in_dim=net_params["L3_hidden_dim"],
            hidden_dim=net_params["hidden_dim"],
            num_layers_atom=net_params['L3_layers_atom'],
            num_layers_mol=net_params['L3_layers_mol'],
            dropout=net_params['dropout'],
            out_dim=net_params["L3_hidden_dim"],
            regressor=False,
        )

    def forward(self, x, edge_index, edge_attr, batch,):
        x_projected = self.project_motif(x)
        motif_graph_features = self.AttentiveEmbedding(x_projected, edge_index, edge_attr, batch, return_lats = True)
        return motif_graph_features

class JT_Channel(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.embedding_frag_lin = nn.Sequential(
            nn.Linear(net_params['frag_dim'], net_params['L3_hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['L3_hidden_dim']),
            nn.LeakyReLU()
        )
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(net_params['num_bond_type'], net_params['L3_hidden_dim'], bias=True),
            nn.BatchNorm1d(net_params['L3_hidden_dim']),
            nn.LeakyReLU()
        )
        self.junction_heads = nn.ModuleList([SingleHeadJunctionLayer(net_params) for _ in range(net_params['num_heads'])])

    def forward(self, x, edge_index, edge_attr, batch, motif_nodes):
        motif_nodes_embedded = self.embedding_frag_lin(motif_nodes)
        edge_attr_embedded = self.embedding_edge_lin(edge_attr)
        x = torch.cat([x, motif_nodes_embedded], dim=-1)
        junction_graph_heads_out = []
        for single_head in self.junction_heads:
            single_head_new_graph = single_head(x, edge_index, edge_attr_embedded, batch)
            junction_graph_heads_out.append(single_head_new_graph)
        super_new_graph = torch.relu(torch.stack(junction_graph_heads_out, dim=0).mean(dim=0))
        return super_new_graph

class GCGAT_v4pro_jit(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.origin_module = OriginChannel(net_params)
        self.frag_module = FragmentChannel(net_params)
        self.junction_module = JT_Channel(net_params)

        #self.output_dim = net_params['output_dim'] #TODO: should default to 1
        self.output_dim = net_params['output_dim'] if 'output_dim' in net_params else 1
        self.frag_res_dim = net_params['L2_hidden_dim']
        concat_dim = net_params['L1_hidden_dim'] + net_params['L2_hidden_dim'] + net_params['L3_hidden_dim']

        self.use_global_features = net_params.get('global_features', False)
        if self.use_global_features:
            self.global_feats_dim = net_params.get('num_global_feats', 1)
            concat_dim += self.global_feats_dim

        self.linear_predict1 = nn.Sequential(
            nn.Dropout(net_params['final_dropout']),
            nn.Linear(concat_dim, int(concat_dim / 2), bias=True),
            nn.LeakyReLU(negative_slope=0.001),
            nn.BatchNorm1d(int(concat_dim / 2)),
        )

        self.linear_predict2 = nn.Sequential()
        mid_dim = int(concat_dim / 2)
        for _ in range(net_params['MLP_layers'] - 1):
            self.linear_predict2.add_module('linear', nn.Linear(mid_dim, mid_dim, bias=True))
            self.linear_predict2.add_module('leaky_relu', nn.LeakyReLU(negative_slope=0.001))
        self.linear_predict2.add_module('output_linear', nn.Linear(mid_dim, self.output_dim, bias=True))

    def forward(
        self,
        data_x: torch.Tensor,
        data_edge_index: torch.Tensor,
        data_edge_attr: torch.Tensor,
        data_batch: torch.Tensor,
        frag_x: torch.Tensor,
        frag_edge_index: torch.Tensor,
        frag_edge_attr: torch.Tensor,
        frag_batch: torch.Tensor,
        junction_x: torch.Tensor,
        junction_edge_index: torch.Tensor,
        junction_edge_attr: torch.Tensor,
        junction_batch: torch.Tensor,
        motif_nodes: torch.Tensor,
        global_feats: torch.Tensor,
    ) -> torch.Tensor:
        device = data_x.device  # Use the device of the input data

        # Origin Module
        graph_origin = self.origin_module(data_x, data_edge_index, data_edge_attr, data_batch)

        # Fragment Module
        graph_frag = self.frag_module(frag_x, frag_edge_index, frag_edge_attr, frag_batch)

        # Aggregate fragment results
        num_mols = int(data_batch.max().item()) + 1
        frag_res = torch.zeros((num_mols, self.frag_res_dim), device=device) #vector that will contain the sums of the frags embedding of each mol
        index = junction_batch.unsqueeze(1).expand(-1, self.frag_res_dim)
        frag_res = frag_res.scatter_add_(0, index, graph_frag)

        motif_nodes = junction_x
        junction_x = graph_frag
        # Junction Module
        super_new_graph = self.junction_module(
            junction_x, junction_edge_index, junction_edge_attr, junction_batch, motif_nodes
        )

        # Concatenate features
        concat_features = torch.cat([graph_origin, frag_res, super_new_graph], dim=-1)

        if self.use_global_features and global_feats.numel() > 0:
            global_feats = global_feats.unsqueeze(-1)
            concat_features = torch.cat([concat_features, global_feats], dim=-1)

        descriptors = self.linear_predict1(concat_features)
        output = self.linear_predict2(descriptors)

        return output