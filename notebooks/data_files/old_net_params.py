import torch

epochs = 1000
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.00126
weight_decay = 1e-4
mlp_layers = 2
atom_layers = 3
mol_layers = 3

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# num_global_feats is the dimension of global features per observation
mlp = None
net_params = {
              "device": device, #shouldn't be passed in in this way, but best we have for now  
              "num_atom_type": 44, # == node_in_dim TODO: check matches with featurizer or read from featurizer
              "num_bond_type": 12, # == edge_in_dim
              "dropout": 0.0,
              "MLP_layers":mlp_layers,
              "frag_dim": frag_dim,
              "final_dropout": 0.119,
            # for origins:
              "num_heads": 1,
            # for AFP:
              "node_in_dim": 44, 
              "edge_in_dim": 12, 
              "num_global_feats":1, 
              "hidden_dim": hidden_dim, #Important: check matches with `L1_hidden_dim`
              "mlp_out_hidden": mlp, 
              "num_layers_atom": atom_layers, 
              "num_layers_mol": mol_layers,
            # for channels:
              "L1_layers_atom": 4, #L1_layers
              "L1_layers_mol": 1,  #L1_depth
              "L1_dropout": 0.142,

              "L2_layers_atom": 2, #L2_layers
              "L2_layers_mol": 3,  #2_depth
              "L2_dropout": 0.255,

              "L3_layers_atom": 1, #L3_layers
              "L3_layers_mol": 4,  #L3_depth
              "L3_dropout": 0.026,

              "L1_hidden_dim": 247,
              "L2_hidden_dim": 141,
              "L3_hidden_dim": 47,
              }