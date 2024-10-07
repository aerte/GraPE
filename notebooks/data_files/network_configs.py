# a list of configs of networks, for reproduceability purposes
import torch 
from grape_chem.utils import return_hidden_layers

epochs = 600
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.00126
weight_decay = 0.003250012
mlp_layers = 4
atom_layers = 3
mol_layers = 3

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

mlp = return_hidden_layers(mlp_layers)

GroupGAT_pka = {
    'training params': {
        "epochs" : 600,
        "batch_size" : 700,
        "patience" : 30,
        "hidden_dim" : 47,
        "learning_rate" : 0.00126,
        "weight_decay" : 0.003250012,
        "mlp_layers" : 4,
        "atom_layers" : 3,
        "mol_layers" : 3,
    },
    'net_params' : {
              "device": device, #shouldn't be passed in in this way, but best we have for now  
              "num_atom_type": 39, # == node_in_dim TODO: check matches with featurizer or read from featurizer
              "num_bond_type": 12, # == edge_in_dim
              "dropout": 0.0,
              "MLP_layers":mlp_layers,
              "frag_dim": None, #get from fragmentation instance
              "final_dropout": 0.257507,
            # for origin:
              "num_heads": 1,
            # for AFP:
              "node_in_dim": 39, 
              "edge_in_dim": 12, 
              "num_global_feats":1, 
              "hidden_dim": hidden_dim, #Important: check matches with `L1_hidden_dim`
              "mlp_out_hidden": mlp, 
              "num_layers_atom": atom_layers, 
              "num_layers_mol": mol_layers,
            # for channels:
              "L1_layers_atom": 3, #L1_layers
              "L1_layers_mol": 3,  #L1_depth
              "L1_dropout": 0.370796,

              "L2_layers_atom": 3, #L2_layers
              "L2_layers_mol": 2,  #2_depth
              "L2_dropout": 0.056907,

              "L3_layers_atom": 1, #L3_layers
              "L3_layers_mol": 4,  #L3_depth
              "L3_dropout": 0.137254,

              "L1_hidden_dim": 125,
              "L2_hidden_dim": 155,
              "L3_hidden_dim": 64,
              }
}