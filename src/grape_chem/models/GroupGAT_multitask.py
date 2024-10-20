import torch
import torch.nn as nn
from grape_chem.models.AFP_gnn_jittable import AFP_jittable as AFP  # Assuming AFP is TorchScript compatible


__all__ = ['GroupGAT_ICP']
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
        return self.AttentiveEmbedding(x, edge_index, edge_attr, batch, True)

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
        origin_heads_out = [head(embedded_x, edge_index, embedded_edge_attr, batch) for head in self.origin_heads]
        origin_heads_out_cat = torch.cat(origin_heads_out, dim=-1)
        graph_origin = self.origin_attend(origin_heads_out_cat)
        return graph_origin

class SingleHeadFragmentLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.AttentiveEmbedding = AFP(
            node_in_dim=net_params["node_in_dim"],
            edge_in_dim=net_params["edge_in_dim"],
            hidden_dim=net_params['hidden_dim'],
            num_layers_atom=net_params['L2_layers_atom'],
            num_layers_mol=net_params['L2_layers_mol'],
            dropout=net_params['dropout'],
            out_dim=net_params['L2_hidden_dim'],
            regressor=False,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.AttentiveEmbedding(x, edge_index, edge_attr, batch, True,)

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
        frag_heads_out = [frag_block(x, edge_index, edge_attr, batch) for frag_block in self.fragment_heads]
        frag_heads_out_cat = torch.cat(frag_heads_out, dim=-1)
        graph_frag = self.frag_attend(frag_heads_out_cat)
        return graph_frag

class SingleHeadJunctionLayer(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.project_motif = nn.Linear(
            net_params['L2_hidden_dim'] + net_params['L3_hidden_dim'],
            net_params['L3_hidden_dim'],
            bias=True
        )
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

    def forward(self, x, edge_index, edge_attr, batch):
        x_projected = self.project_motif(x)
        motif_graph_features = self.AttentiveEmbedding(x_projected, edge_index, edge_attr, batch, True)
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
        self.junction_heads = nn.ModuleList(
            [SingleHeadJunctionLayer(net_params) for _ in range(net_params['num_heads'])]
        )

    def forward(self, x, edge_index, edge_attr, batch, motif_nodes):
        motif_nodes_embedded = self.embedding_frag_lin(motif_nodes)
        edge_attr_embedded = self.embedding_edge_lin(edge_attr)
        x = torch.cat([x, motif_nodes_embedded], dim=-1)
        junction_graph_heads_out = [
            head(x, edge_index, edge_attr_embedded, batch) for head in self.junction_heads
        ]
        super_new_graph = torch.relu(torch.stack(junction_graph_heads_out, dim=0).mean(dim=0))
        return super_new_graph

class GroupGAT_ICP(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        self.origin_module = OriginChannel(net_params)
        self.frag_module = FragmentChannel(net_params)
        self.junction_module = JT_Channel(net_params)

        self.output_dim = 5  # Outputting 5 parameters per molecule

        self.frag_res_dim = net_params['L2_hidden_dim']
        concat_dim = net_params['L1_hidden_dim'] + net_params['L2_hidden_dim'] + net_params['L3_hidden_dim']

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

        self.C_p_layer = CpLayer()

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
        device = data_x.device

        # Origin Module
        graph_origin = self.origin_module(data_x, data_edge_index, data_edge_attr, data_batch)

        # Fragment Module
        graph_frag = self.frag_module(frag_x, frag_edge_index, frag_edge_attr, frag_batch)

        # Aggregate fragment results
        num_mols = int(data_batch.max().item()) + 1
        frag_res = torch.zeros((num_mols, self.frag_res_dim), device=device)
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

        descriptors = self.linear_predict1(concat_features)
        output_params = self.linear_predict2(descriptors)  # Shape: [num_mols, num coeffs]
        #print(f"Output params stats - min: {output_params.min()}, max: {output_params.max()}, mean: {output_params.mean()}")

        B, C, D, E, F = output_params[:, 0], output_params[:, 1], output_params[:, 2], output_params[:, 3], output_params[:, 4]
        T = global_feats.squeeze()  # Assuming global_feats is shape [num_mols, 1]

        # print(f"B stats - min: {B.min()}, max: {B.max()}, mean: {B.mean()}")
        # print(f"C stats - min: {C.min()}, max: {C.max()}, mean: {C.mean()}")
        # print(f"D stats - min: {D.min()}, max: {D.max()}, mean: {D.mean()}")
        # print(f"E stats - min: {E.min()}, max: {E.max()}, mean: {E.mean()}")
        # print(f"F stats - min: {F.min()}, max: {F.max()}, mean: {F.mean()}")

        return self.C_p_layer(B, C, D, E, F, T)

    

class CpLayer(nn.Module):
    def __init__(self):
        super(CpLayer, self).__init__()

    def forward(self, B, C, D, E, F, T):
        """
        Computes Cp using the provided equation:
        Cp = B + C * ((D / T) / sinh(D / T)) ** 2 + E * ((F / T) / cosh(F / T)) ** 2
        """
        epsilon = 1e-7 # to avoid 0 division
        T = T + epsilon

        D_over_T = torch.clamp(D / T, min=-1000, max=1000)
        F_over_T = torch.clamp(F / T, min=-1000, max=1000)

        # print(f"D_over_T stats - min: {D_over_T.min()}, max: {D_over_T.max()}, mean: {D_over_T.mean()}")
        # print(f"F_over_T stats - min: {F_over_T.min()}, max: {F_over_T.max()}, mean: {F_over_T.mean()}")

        sinh_term = torch.sinh(D_over_T)
        cosh_term = torch.cosh(F_over_T)
        # print(f"sinh_term stats - min: {sinh_term.min()}, max: {sinh_term.max()}, mean: {sinh_term.mean()}")
        # print(f"cosh_term stats - min: {cosh_term.min()}, max: {cosh_term.max()}, mean: {cosh_term.mean()}")

        Cp = B + C * ((D_over_T / sinh_term) ** 2) + E * ((F_over_T / cosh_term) ** 2)
        # Ensure T doesn't contain zero
        
        # Cp_o = B + C * ((D / T) / torch.sinh(D / T)) ** 2 + E * (
        #         (F / T) / torch.cosh(F / T)) ** 2
        return Cp


#     # print(f"B stats - min: {B.min()}, max: {B.max()}, mean: {B.mean()}")
# # print(f"C stats - min: {C.min()}, max: {C.max()}, mean: {C.mean()}")
# # print(f"D stats - min: {D.min()}, max: {D.max()}, mean: {D.mean()}")
# # print(f"E stats - min: {E.min()}, max: {E.max()}, mean: {E.mean()}")
# # print(f"F stats - min: {F.min()}, max: {F.max()}, mean: {F.mean()}")
# # Compute Cp as per the equation:
# # Cp = B + C * ((D / T) / sinh(D / T)) ** 2 + E * ((F / T) / cosh(F / T)) ** 2

# D_over_T = D / T
# F_over_T = F / T

# # print(f"D_over_T stats - min: {D_over_T.min()}, max: {D_over_T.max()}, mean: {D_over_T.mean()}")
# # print(f"F_over_T stats - min: {F_over_T.min()}, max: {F_over_T.max()}, mean: {F_over_T.mean()}")

# # to avoid 0 division
# epsilon = 1e-7

# sinh_term = torch.sinh(D_over_T.clamp(min=epsilon))+epsilon
# cosh_term = torch.cosh(F_over_T.clamp(min=epsilon))+epsilon

# # print(f"sinh_term stats - min: {sinh_term.min()}, max: {sinh_term.max()}, mean: {sinh_term.mean()}")
# # print(f"cosh_term stats - min: {cosh_term.min()}, max: {cosh_term.max()}, mean: {cosh_term.mean()}")

# #breakpoint()
# Cp = B + C * ((D_over_T / sinh_term) ** 2) + E * ((F_over_T / cosh_term) ** 2)

# # Cp = self.B + self.C * ((self.D / x) / (torch.sinh(self.D / x)+epsilon)) ** 2 + self.E * (
# #         (self.F / x) / (torch.cosh(self.F / x)+epsilon)) ** 2

# return Cp  # Shape: [num_mols]