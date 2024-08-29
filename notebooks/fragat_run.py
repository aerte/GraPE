from grape_chem.models import AFP
from grape_chem.models import GroupGAT
from grape_chem.utils import DataSet, train_model, EarlyStopping, split_data, test_model, pred_metric, return_hidden_layers, set_seed, JT_SubGraph, FragmentGraphDataSet
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd

## Install GraPE with: pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

def standardize(x, mean, std):
    return (x - mean) / std

##########################################################################################
#####################    Data Input Region  ##############################################
##########################################################################################

set_seed(42)

# Hyperparameters
epochs = 50
batch_size = 700
patience = 30
hidden_dim = 132
learning_rate = 0.001
weight_decay = 1e-4
mlp_layers = 1
atom_layers = 3
mol_layers = 3


# Change to your own specifications
root = './env/data_splits.xlsx'
sheet_name = 'Melting Point'

df = pd.read_excel(root, sheet_name=sheet_name).iloc[:21] #REMOVE the slice, just because fragmentation is so slow
smiles = df['SMILES'].to_numpy()
target = df['Target'].to_numpy()
### Global feature from sheet, uncomment
#global_feats = df['Global Feats'].to_numpy()

#### REMOVE, just for testing ####
#global_feats = np.random.randn(len(smiles))
##################################


############ We need to standardize BEFORE loading it into a DataSet #############
mean_target, std_target = np.mean(target), np.std(target)
target = standardize(target, mean_target, std_target)
# mean_global_feats, std_global_feats = np.mean(global_feats), np.std(global_feats)
# global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

fragmentation_scheme = "MG_plus_reference"
print("initializing frag...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme)
frag_dim = fragmentation.frag_dim
print("done.")

# Load into DataSet
data = DataSet(smiles=smiles, target=target, global_features=None, filter=True, fragmentation=fragmentation)

train, val, test = split_data(data, split_type='consecutive', split_frac=[0.8, 0.1, 0.1],)
############################################################################################
############################################################################################
############################################################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# num_global_feats is the dimension of global features per observation
mlp = return_hidden_layers(mlp_layers)
net_params = {
              "device": device, #shouldn't be passed in in this way, but best we have for now  
              "num_atom_type": 44, # == node_in_dim TODO: check matches with featurizer or read from featurizer
              "num_bond_type": 12, # == edge_in_dim
              "dropout": 0.0,
              "MLP_layers":mlp_layers,
              "frag_dim": frag_dim,
              "final_dropout": 0.0,
            # for origin:
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
              "L1_depth": 1,
              "L1_layers": 1,
              "L1_dropout": 0.0,

              "L2_depth": 1,
              "L2_layers": 1,
              "L2_dropout": 0.0,

              "L3_depth": 1,
              "L3_layers": 1,
              "L3_dropout": 0.0,

              "L1_hidden_dim": 132, #3*node in_dim
              "L2_hidden_dim": 133,
              "L3_hidden_dim": 36,
              }
model = GroupGAT.GCGAT_v4pro(net_params)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=patience, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9999, min_lr=0.0000000000001,
                                           patience=patience)

loss_func = torch.nn.functional.l1_loss

model.to(device)
train_model(model=model, loss_func=loss_func, optimizer=optimizer, train_data_loader=train,
            val_data_loader=val, epochs=epochs, device=device, batch_size=batch_size, model_needs_frag=True)

####### Generating prediction tensor for the TEST set (Not rescaled) #########

pred = test_model(model=model, test_data_loader=test, device=device, batch_size=batch_size) #TODO: make it able to take a loss func
pred_metric(prediction=pred, target=test.y, metrics='all', print_out=True)

# ---------------------------------------------------------------------------------------



####### Example for rescaling the MAE prediction ##########

test_mae = pred_metric(prediction=pred, target=test.y, metrics='mae', print_out=False)['mae']
test_mae_rescaled = test_mae * std_target + mean_target
print(f'Rescaled MAE for the test set {test_mae_rescaled:.3f}')

# ---------------------------------------------------------------------------------------


####### Example for overall evaluation of the MAE #########

train_preds = test_model(model=model, test_data_loader=train, device=device) #TODO
val_preds = test_model(model=model, test_data_loader=val, device=device)

train_mae = pred_metric(prediction=train_preds, target=train.y, metrics='mae', print_out=False)['mae']
val_mae = pred_metric(prediction=val_preds, target=val.y, metrics='mae', print_out=False)['mae']

overall_mae = (train_mae+val_mae+test_mae)/3 * std_target + mean_target
print(f'Rescaled overall MAE {overall_mae:.3f}')