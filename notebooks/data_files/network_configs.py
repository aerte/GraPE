# a list of configs of networks,
#  for reproduceability purposes
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
              "device": device,#shouldn't be passed in in this way but best we have for now  
              "num_atom_type": 39,# == node_in_dim TODO: check matches with featurizer or read from featurizer
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

GroupGAT_Vc = {
"""
this is coincidentally the same config that acheived good-ish performance
on the Solvation dataset with only the first layer (L1/origin) of the
GroupGAT model.
"""
    'training params': {
        "epochs":1000,
        "batch_size":700,
        "patience":30,
        "hidden_dim":47,
        "learning_rate":0.001054627,
        "weight_decay":1e-4,
        "mlp_layers":2,
        "atom_layers":3,
        "mol_layers":3,
    },
    'net_params': {
                  "device": device, 
                  "num_atom_type": 44, # == node_in_dim
                  "num_bond_type": 12, # == edge_in_dim
                  "dropout": 0.0,
                  "MLP_layers":mlp_layers,
                  "frag_dim": None, #get from fragmentation instance
                  "final_dropout": 0.119,
                  "global_features": True,
                # for origins:
                  "num_heads": 1,
                # for AFP:
                  "node_in_dim": 44, 
                  "edge_in_dim": 12, 
                  "num_global_feats":1, 
                  "hidden_dim": hidden_dim, #check matches with `L1_hidden_dim`
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
}

ICP_coupled_multitask_groupGAT = {
    'training params': {
        "epochs" : 12000,
      "batch_size" : 700,
      "patience" : 30,
      "hidden_dim" : 47,
      "learning_rate" : 0.005,
      "weight_decay" : 1e-6,
      "mlp_layers" : 2,
      "atom_layers" : 3,
      "mol_layers" : 3,
    },
    'seed': 14102024,
    'net_params': {
        "device": device,
        "num_atom_type": 44,
        "num_bond_type": 12,
        "dropout": 0.0,
        "MLP_layers": mlp_layers,
        "frag_dim": None, #probably 219
        "final_dropout": 0.119,
        "use_global_features": True,
        "num_heads": 1,
        "node_in_dim": 44,
        "edge_in_dim": 12,
        "num_global_feats": 1,
        "hidden_dim": hidden_dim,
        "mlp_out_hidden": mlp,
        "num_layers_atom": atom_layers,
        "num_layers_mol": mol_layers,
        "L1_layers_atom": 4,
        "L1_layers_mol": 1,
        "L1_dropout": 0.142,
        "L2_layers_atom": 2,
        "L2_layers_mol": 3,
        "L2_dropout": 0.255,
        "L3_layers_atom": 1,
        "L3_layers_mol": 4,
        "L3_dropout": 0.026,
        "L1_hidden_dim": 247,
        "L2_hidden_dim": 141,
        "L3_hidden_dim": 47,
        "num_prediction": 5, #was not used like this in the actual implementation, but this would have been the value
    }
}

GroupGAT_pKA_after_1st_hyperparam_search = {'L1_dropout': 0.428770053646,
 'L1_hidden_dim': 51,
 'L1_layers_atom': 5,
 'L1_layers_mol': 1,
 'MLP_layers': 2,
 'depth': 2,
 'dropout': 0.0317320110957,
 'gnn_hidden_dim': 108,
 'hidden_dim': 114,
 'initial_lr': 0.0036979616644,
 'lr_reduction_factor': 0.7552366725079,
 'mlp_layers': 1,
 'num_layers_atom': 3,
 'num_layers_mol': 2,
 'weight_decay': 0.0051247177026
 }


new_pka_hyperparam_opt = {
 'L1_dropout': 0.1723501091646,
 'L1_hidden_dim': 223,
 'L1_layers_atom': 4,
 'L1_layers_mol': 4,
 'L2_dropout': 0.1900047779751,
 'L2_hidden_dim': 202,
 'L2_layers_atom': 5,
 'L2_layers_mol': 3,
 'L3_dropout': 0.00054874106,
 'L3_hidden_dim': 227,
 'L3_layers_atom': 4,
 'L3_layers_mol': 2,
 'MLP_layers': 3,
 'depth': 4,
 'dropout': 0.0523608921771,
 'final_dropout': 0.1720950490985,
 'hidden_dim': 378,
 'initial_lr': 0.0016848138285,
 'lr_reduction_factor': 0.8409780590571,
 'mlp_layers': 1,
 'num_heads': 1,
 'num_layers_atom': 1,
 'num_layers_mol': 5,
 'weight_decay': 0.0003324230157}

params={
'L1_dropout': 0.2618828014953,
'L1_hidden_dim': 232,
'L1_layers_atom': 3,
'L1_layers_mol': 2,

'L2_dropout': 0.2884782246237,
'L2_hidden_dim': 204,
'L2_layers_atom': 3,
'L2_layers_mol': 3,

'L3_dropout': 0.0540956622725,
'L3_hidden_dim': 159,
'L3_layers_atom': 2,
'L3_layers_mol': 5,

'MLP_layers': 2,
'depth': 4,
'dropout': 0.1005681539104,
'final_dropout': 0.2347400515029,
'hidden_dim': 272,
'initial_lr': 0.0005717428089,
'lr_reduction_factor': 0.9637191579013,
'mlp_layers': 1,
'num_heads': 1,
'num_layers_atom': 3,
'num_layers_mol': 4,
'weight_decay': 0.0008670852451
}

hyperparam_opt_ICP_pddgroupgat_params = {
    'L1_dropout': 0.2785096050756,
    'L1_hidden_dim': 91,
    'L1_layers_atom': 4,
    'L1_layers_mol': 5,
    'L2_dropout': 0.2994343573948,
    'L2_hidden_dim': 69,
    'L2_layers_atom': 2,
    'L2_layers_mol': 2,
    'L3_dropout': 0.2111227117327,
    'L3_hidden_dim': 191,
    'L3_layers_atom': 1,
    'L3_layers_mol': 2,
    'MLP_layers': 2,
    'dropout': 0.1153677540364,
    'final_dropout': 0.0392564772347,
    'hidden_dim': 468,
    'initial_lr': 3.60530383e-05,
    'lr_reduction_factor': 0.9593370235448,
    'num_layers_atom': 5,
    'num_layers_mol': 3,
    'weight_decay': 7.81484164e-05
}

groupgat_pdd_params={
'L1_dropout': 0.1459542359848,
'L1_hidden_dim': 249,
'L1_layers_atom': 2,
'L1_layers_mol': 4,
'L2_dropout': 0.0798744447151,
'L2_hidden_dim': 227,
'L2_layers_atom': 2,
'L2_layers_mol': 2,
'L3_dropout': 0.0634387027139,
'L3_hidden_dim': 76,
'L3_layers_atom': 1,
'L3_layers_mol': 3,
'MLP_layers': 1,
'dropout': 0.2340773447748,
'final_dropout': 0.482811492885,
'hidden_dim': 319,
'initial_lr': 0.0014516461238,
'lr_reduction_factor': 0.5874594731827,
'num_heads': 3,
'num_layers_atom': 2,
'num_layers_mol': 2,
'weight_decay': 0.0065225760795}