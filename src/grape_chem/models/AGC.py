import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch

from grape_chem.models import AFP

__all__ = ['AGC']
class SingleHeadOriginLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        from grape_chem.models import AFP
        self.AttentiveEmbedding = AFP(node_in_dim=net_params["L1_hidden_dim"],
                                      edge_in_dim=net_params["L1_hidden_dim"],
                                        hidden_dim=net_params['hidden_dim'],
                                        num_layers_atom=net_params['L1_layers_atom'],
                                        num_layers_mol=net_params['L1_layers_mol'],
                                        dropout=net_params['dropout'],
                                        out_dim=net_params['L1_hidden_dim'],
                                        regressor=False
                                    )

    def forward(self, data):
        d = Data(data.x, data.edge_index, data.edge_attr, data.batch)
        d.batch = data.batch  # batch needs to be explicitly passed because AFP expects it
        origin_graph_features = self.AttentiveEmbedding(d, return_lats=True)
        return origin_graph_features

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

    def forward(self, data):
        embedded_x = self.embedding_node_lin(data.x)
        embedded_edge_attr = self.embedding_edge_lin(data.edge_attr)
        origin_data = Data(embedded_x, data.edge_index, embedded_edge_attr,)
        origin_data.batch = data.batch
        origin_heads_out = [head(origin_data) for head in self.origin_heads]
        graph_origin = self.origin_attend(torch.cat(origin_heads_out, dim=-1))
        return graph_origin

class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        from grape_chem.models import AFP
        self.AttentiveEmbedding = AFP(node_in_dim=net_params["node_in_dim"],
                                      edge_in_dim=net_params["edge_in_dim"],
                                        hidden_dim=net_params['hidden_dim'],
                                        num_layers_atom=net_params['L2_layers_atom'],
                                        num_layers_mol=net_params['L2_layers_mol'],
                                        dropout=net_params['dropout'],
                                        out_dim=net_params['L2_hidden_dim'],
                                        regressor=False
                                    )

    def forward(self, data):
        d = Data(data.x, data.edge_index, data.edge_attr, data.batch)
        d.batch = data.batch
        frag_features = self.AttentiveEmbedding(d, return_lats=True)
        return frag_features

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

    def forward(self, batch):
        frag_heads_out = [frag_block(batch) for frag_block in self.fragment_heads]
        graph_frag = self.frag_attend(torch.cat(frag_heads_out, dim=-1))
        return graph_frag

class SingleHeadJunctionLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        from grape_chem.models import AFP
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
                                return_super_nodes=True
                            )
    def forward(self, data):
        data.x = self.project_motif(data.x)
        d = Data(data.x, data.edge_index, data.edge_attr, data.batch)
        d.batch = data.batch
        motif_graph_features, alphas  = self.AttentiveEmbedding(d, return_lats=True)
        return motif_graph_features, alphas

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

    def forward(self, batch, motif_nodes):
        motif_nodes = self.embedding_frag_lin(motif_nodes)
        batch.edge_attr = self.embedding_edge_lin(batch.edge_attr)
        batch.x = torch.cat([batch.x, motif_nodes], dim=-1)
        junction_graph_heads_out = []
        junction_attention_heads_out = []
        for single_head in self.junction_heads:
            single_head_new_graph, single_head_attention_weight  = single_head(batch)
            junction_graph_heads_out.append(single_head_new_graph)
            junction_attention_heads_out.append(single_head_attention_weight[1])
        super_new_graph = torch.relu(torch.mean(torch.stack(junction_graph_heads_out, dim=1), dim=1))
        super_attention_weight = torch.mean(torch.stack(junction_attention_heads_out, dim=1), dim=1)
        return super_new_graph, super_attention_weight

class AGC(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.origin_module = OriginChannel(net_params)  # Include the OriginChannel
        self.frag_module = FragmentChannel(net_params)
        self.junction_module = JT_Channel(net_params)

        self.frag_res_dim = net_params['L2_hidden_dim']  # Needed to sum and concat the frag graph embeddings
        concat_dim = net_params['L1_hidden_dim'] + net_params['L2_hidden_dim'] + net_params['L3_hidden_dim']
        self.use_global_feats = net_params.get('use_global_features', False)
        if self.use_global_feats:
            concat_dim += 1
        self.linear_predict1 = nn.Sequential(
            nn.Dropout(net_params['final_dropout']),
            nn.Linear(concat_dim, int(concat_dim / 2), bias=True),
            nn.LeakyReLU(negative_slope=0.001),
            nn.BatchNorm1d(int(concat_dim / 2)),
        )

        self.linear_predict2 = nn.Sequential()
        mid_dim = int(concat_dim / 2)
        for _ in range(net_params['MLP_layers'] - 1):
            self.linear_predict2.append(nn.Linear(mid_dim, mid_dim, bias=True))
            self.linear_predict2.append(nn.LeakyReLU(negative_slope=0.001))
        self.linear_predict2.append(nn.Linear(mid_dim, 1, bias=True))

    def forward(self, data, get_attention=False, get_descriptors=False, global_feature=None):
        device = self.parameters().__next__().device
        data = data.to(device)

        # Origin module
        graph_origin = self.origin_module(data)

        # Fragment module
        frag_data, junction_data = Batch.from_data_list(data.frag_graphs), Batch.from_data_list(data.motif_graphs)
        graph_frag = self.frag_module(frag_data)

        # Junction module
        motif_nodes = junction_data.x
        junction_data.x = graph_frag.clone()
        super_new_graph, super_attention_weight = self.junction_module(junction_data, motif_nodes)

        # Sum features for motif graph
        num_mols = len(junction_data.batch.unique(dim=0))
        frag_res = torch.zeros(num_mols, self.frag_res_dim, device=device)
        index = junction_data.batch.unsqueeze(1).expand(-1, self.frag_res_dim)
        frag_res = frag_res.scatter_add_(0, index, graph_frag)

        # Concatenate outputs from all channels
        concat_features = torch.cat([graph_origin, frag_res, super_new_graph], dim=-1)

        if self.use_global_feats:
            if hasattr(data, 'global_feats') and data.global_feats is not None:
                global_feats = data.global_feats.to(concat_features.device).unsqueeze(-1)
                concat_features = torch.cat([concat_features, global_feats], dim=-1)

        descriptors = self.linear_predict1(concat_features)
        output = self.linear_predict2(descriptors)

        results = [output]
        if get_attention:
            results.append(super_attention_weight)
        if get_descriptors:
            results.append(descriptors)

        return tuple(results) if len(results) > 1 else results[0]

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
