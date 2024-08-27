import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch

from grape_chem.models import AFP

__all__ = ['GCGAT_v4pro']
class SingleHeadOriginLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AttentiveEmbedding = AFP(node_in_dim=net_params["node_in_dim"],
                                      edge_in_dim=net_params["edge_in_dim"],
                                        hidden_dim=net_params['hidden_dim'],
                                        num_layers_atom=net_params['num_layers_atom'],
                                        num_layers_mol=net_params['num_layers_mol'], #why is it called timesteps (old codebase)?
                                        dropout=net_params['dropout'],
                                        out_dim=net_params['L1_hidden_dim'],
                                        regressor=False
                                    )

    def forward(self, data):
        d = Data(data.x, data.edge_index, data.edge_attr, data.batch)
        d.batch = data.batch #batch needs to be explicitely passed because AFP expects it. Known issue/bad QoL
        origin_graph_features = self.AttentiveEmbedding(d, return_lats = True)
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

    def reset_parameters(self):
        # Reset parameters in all sub-modules
        #TODO: clean-up
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

    def forward(self, data): #TODO: perhaps batch should be a data instance instead
        origin_data = Data(data.x, data.edge_index, data.edge_attr,)
        origin_data.batch = data.batch
        #(embedding -> hidden_dim is handled in the call to AFP)
        origin_heads_out = [head(origin_data) for head in self.origin_heads]
        graph_origin = self.origin_attend(torch.cat(origin_heads_out, dim=-1))
        return graph_origin
    
class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AttentiveEmbedding = AFP(node_in_dim=net_params["node_in_dim"],
                                      edge_in_dim=net_params["edge_in_dim"],
                                        hidden_dim=net_params['hidden_dim'],
                                        num_layers_atom=net_params['num_layers_atom'],
                                        num_layers_mol=net_params['num_layers_mol'], #why is it called timesteps (old codebase)?
                                        dropout=net_params['dropout'],
                                        out_dim=net_params['L2_hidden_dim'],
                                        regressor=False
                                    )

    def forward(self, data):
        breakpoint()
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
        #batch.x = self.embedding_node_lin(batch.x.float())
        #batch.edge_attr = self.embedding_edge_lin(batch.edge_attr.float())
        frag_heads_out = [frag_block(batch) for frag_block in self.fragment_heads]
        graph_frag = self.frag_attend(torch.cat(frag_heads_out, dim=-1))
        return graph_frag
    
class SingleHeadJunctionLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.project_motif = nn.Linear(net_params['L2_hidden_dim'] + net_params['L3_hidden_dim'], net_params['L3_hidden_dim'], bias=True)
        self.AttentiveEmbedding = AFP(node_in_dim=net_params["node_in_dim"],
                                edge_in_dim=net_params["edge_in_dim"],
                                hidden_dim=net_params['hidden_dim'],
                                num_layers_atom=net_params['num_layers_atom'],
                                num_layers_mol=net_params['num_layers_mol'], #used to be called num_timesteps in old codebase
                                dropout=net_params['dropout'],
                                regressor=False
                            ) #I hope these actually match

    def forward(self, data):
        data.x = self.project_motif(data.x)
        d = Data(data.x, data.edge_index, data.edge_attr, data.batch)
        d.batch = data.batch
        motif_graph_features = self.AttentiveEmbedding(d)
        #graph_features, attention_weights = global_add_pool(node_features, data.batch), None  # Adjust if you get attention weights
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

    def forward(self, batch):
        """
        TODO: still doesn't match exactly with dgl version
        """
        batch.x = self.embedding_frag_lin(batch.x)
        batch.edge_attr = self.embedding_edge_lin(batch.edge_attr)
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

    def forward(self, data, get_attention=False, get_descriptors=False):
        # Approach:
        # 1. extract graph-level features from different channels
        device = self.parameters().__next__().device
        data = data.to(device) #TODO: investigate why second call to this is required
        origin_data = Data(data.x, data.edge_index, data.edge_attr,)
        origin_data.batch = data.batch
        breakpoint()
        frag_data, junction_data = Batch.from_data_list(data.frag_graphs[0]), Batch.from_data_list(data.motif_graphs)
        graph_origin = self.origin_module(origin_data)
        graph_frag = self.frag_module(frag_data)
        super_new_graph, super_attention_weight = self.junction_module(junction_data)


        # if descriptors: sum node features for motif graph (akin to dgl.sum_nodes)
        motifs_series = global_add_pool(graph_frag.x, graph_frag.batch) if get_attention else torch.zeros((graph_frag.x.size(0), 0), device=graph_frag.x.device)

        # 2. concat the output from different channels
        concat_features = torch.cat([graph_origin, graph_frag, super_new_graph, motifs_series], dim=-1)
        descriptors = self.linear_predict1(concat_features)
        output = self.linear_predict2(descriptors)
        
        results = [output]
        if get_attention:
            # TODO: depending on how attention weights are returned (per-node? per fragment?) 
            # we can either directly use them or we'll need to aggregate or de-aggregate them
            # attention_list_array = []
            # unique_graph_ids = torch.unique(data.batch, sorted=True)
            # for graph_id in unique_graph_ids:
            #     mask = (data.batch == graph_id)
            #     graph_attention = super_attention_weight[mask]
            #     attention_list_array.append(graph_attention.detach().cpu().numpy())
            # results.append(attention_list_array)
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