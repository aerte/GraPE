
from grape_chem.models import GroupGAT_jittable
from grape_chem.utils import DataSet, train_model_jit, EarlyStopping, split_data, test_model_jit, pred_metric, return_hidden_layers, set_seed, JT_SubGraph, FragmentGraphDataSet
from grape_chem.datasets import FreeSolv 
from torch.optim import lr_scheduler
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Union, List, Tuple
from torch import Tensor
import os

## Install GraPE with: pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

def standardize(x, mean, std):
    return (x - mean) / std

def unstandardize(x, mean, std):
    return x * std + mean

##########################################################################################
#####################    Data Input Region  ##############################################
##########################################################################################

set_seed(42)

# Change to your own specifications
root = './env/pka_dataset.xlsx'
sheet_name = ''

df = pd.read_excel(root,)#.iloc[:25] 
smiles = df['SMILES'].to_numpy()
target = df['Target'].to_numpy()

#specific to one xlsx with a "Tag" column
tags = df['Tag'].to_numpy()
unique_tags = np.unique(tags)
tag_to_int = {'Train': 0, 'Val': 1, 'Test': 2}
custom_split = np.array([tag_to_int[tag] for tag in tags])

### Global feature from sheet, uncomment
#global_feats = df['Global Feats'].to_numpy()


############ We need to standardize BEFORE loading it into a DataSet #############
mean_target, std_target = np.mean(target), np.std(target)
target = standardize(target, mean_target, std_target)

########################## fragmentation #########################################
fragmentation_scheme = "MG_plus_reference"
print("initializing frag...")
fragmentation = JT_SubGraph(scheme=fragmentation_scheme, save_file_path='env/frag_pka_run.pth')
frag_dim = fragmentation.frag_dim
print("done.")


########################### FreeSolv ###################################################
#data = FreeSolv(fragmentation=fragmentation)
########################################################################################

######################## QM9 / testing /excel ##########################################
#data = DataSet(smiles=smiles, target=target, global_features=global_feats, filter=True, fragmentation=fragmentation, custom_split=custom_split)
#custom_split = data.custom_split #messy but it gets recomputed in this way
########################################################################################


#train_set, val_set, _ = data.split_and_scale(scale=True, split_type='random')

#train, val, test = split_data(data, split_type='custom', custom_split=custom_split,)
############################################################################################
############################################################################################
############################################################################################

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Hyperparameters
epochs = 1000
batch_size = 700
patience = 30
hidden_dim = 47
learning_rate = 0.00126
weight_decay = 0.003250012
mlp_layers = 4
atom_layers = 3
mol_layers = 3
#final_droupout = 0.257507
final_droupout = 0.257507

init_lr = 0.001054627
min_lr = 1.00E-09  

# num_global_feats is the dimension of global features per observation
mlp = return_hidden_layers(mlp_layers)
net_params = {
              "device": device, #shouldn't be passed in in this way, but best we have for now  
              "num_atom_type": 44, # == node_in_dim TODO: check matches with featurizer or read from featurizer
              "num_bond_type": 12, # == edge_in_dim
              "dropout": 0.0,
              "MLP_layers":2,
              "frag_dim": frag_dim,
              "final_dropout": final_droupout,
            # for origins:
              "num_heads": 1,
            # for AFP:
              "node_in_dim": 44, 
              "edge_in_dim": 12, 
              "num_global_feats":0, 
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
model = torch.jit.script(GroupGAT_jittable.GCGAT_v4pro_jit(net_params))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_Stopper = EarlyStopping(patience=62, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7552366725079, min_lr=min_lr,
                                           patience=30)

loss_func = torch.nn.functional.mse_loss

model.to(device)

model_filename = 'gcgat_pka_jitted_latest.pth'

if os.path.exists(model_filename):
    print(f"Model file '{model_filename}' found. Loading the trained model.")
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
else:
    print("please trian the model first.")

####### Generating prediction tensor for the TEST set (Not rescaled) #########

# ---------------------------------------------------------------------------------------

new_smiles = ['O=C(O)c1ccc(N)cc1', 'O=C(O)c1cc(N)ccc1']

# Since we don't have actual target values, we'll use placeholders
new_targets = np.zeros(len(new_smiles))

# Create the DataSet with the new SMILES
new_data = DataSet(
    smiles=new_smiles,
    target=new_targets,
    filter=True,
    fragmentation=fragmentation
)

# Create DataLoader for the new data
new_loader = DataLoader([data for data in new_data], batch_size=len(new_data))

def test_model_jit_with_parity(
    model: torch.nn.Module,
    test_data_loader: Union[List,], #Union[List, DataLoader],
    device: str = None,
    batch_size: int = 32,
    return_latents: bool = False,
    model_needs_frag: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Auxiliary function to test a trained JIT-compiled model and return the predictions as well as the targets and optional latent node
    representations. Can initialize DataLoaders if only list of Data objects are given.

    Notes
    -----
    This function is designed for JIT-compiled models that require individual tensors as input.

    Parameters
    ----------
    model : torch.nn.Module
        The JIT-compiled model to be tested.
    test_data_loader : list of Data or DataLoader
        A list of Data objects or the DataLoader directly to be used as the test graphs.
    device : str, optional
        Torch device to be used ('cpu', 'cuda', or 'mps'). Default is 'cpu'.
    batch_size : int, optional
        Batch size of the DataLoader if not given directly. Default is 32.
    return_latents : bool, optional
        Determines if the latents should be returned. **If used, the model must include `return_lats` parameter**.
        Default is False.
    model_needs_frag : bool, optional
        Whether the model needs fragment graphs or not. Default is False.

    Returns
    -------
    Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor]
        Returns predictions and targets, and optionally latents if `return_latents` is True.
    """

    device = torch.device('cpu') if device is None else torch.device(device)
    if not isinstance(test_data_loader, DataLoader):
        test_data_loader = DataLoader([data for data in test_data_loader], batch_size=batch_size)

    model.eval()

    with torch.no_grad():
        preds_list = []
        targets_list = []
        if return_latents:
            latents_list = []
        with tqdm(total=len(test_data_loader)) as pbar:
            for idx, batch in enumerate(test_data_loader):
                # Move batch to device
                batch = batch.to(device)

                if model_needs_frag:
                    # Extract necessary tensors for models that require fragment information
                    data_x = batch.x
                    data_edge_index = batch.edge_index
                    data_edge_attr = batch.edge_attr
                    data_batch = batch.batch

                    # Fragment graphs
                    frag_graphs = batch.frag_graphs  # List[Data]
                    frag_batch = Batch.from_data_list(frag_graphs).batch.to(device)
                    frag_x = Batch.from_data_list(frag_graphs).x.to(device)
                    frag_edge_index = Batch.from_data_list(frag_graphs).edge_index.to(device)
                    frag_edge_attr = Batch.from_data_list(frag_graphs).edge_attr.to(device)

                    # Junction graphs (motif graphs)
                    motif_graphs = batch.motif_graphs  # List[Data]
                    junction_batch = Batch.from_data_list(motif_graphs).batch.to(device)
                    junction_x = Batch.from_data_list(motif_graphs).x.to(device)
                    junction_edge_index = Batch.from_data_list(motif_graphs).edge_index.to(device)
                    junction_edge_attr = Batch.from_data_list(motif_graphs).edge_attr.to(device)

                    motif_nodes = frag_x  # Assuming motif nodes are fragment node features

                    if hasattr(batch, 'global_feats'):
                        global_feats = batch.global_feats
                    else:
                        global_feats = torch.empty((data_x.size(0), 1), dtype=torch.float32).to(device)

                    global_feats = global_feats.to(device)
                    
                    # Forward pass
                    if return_latents:
                        # Assuming the model's forward method supports `return_lats` parameter
                        out, lat = model(
                            data_x,
                            data_edge_index,
                            data_edge_attr,
                            data_batch,
                            frag_x,
                            frag_edge_index,
                            frag_edge_attr,
                            frag_batch,
                            junction_x,
                            junction_edge_index,
                            junction_edge_attr,
                            junction_batch,
                            motif_nodes,
                            global_feats
                        )
                        lat = lat.detach().cpu()
                        latents_list.append(lat)
                    else:
                        out = model(
                            data_x,
                            data_edge_index,
                            data_edge_attr,
                            data_batch,
                            frag_x,
                            frag_edge_index,
                            frag_edge_attr,
                            frag_batch,
                            junction_x,
                            junction_edge_index,
                            junction_edge_attr,
                            junction_batch,
                            motif_nodes,
                            global_feats
                        )
                else:
                    # For models that do not need fragment information
                    if return_latents:
                        out, lat = model(batch, return_lats=True)
                        lat = lat.detach().cpu()
                        latents_list.append(lat)
                    else:
                        out = model(batch)

                # Collect predictions and targets

                out_unscaled = unstandardize(out.detach().cpu(), mean_target, std_target)
                target_unscaled = unstandardize(batch.y.detach().cpu(), mean_target, std_target)

                preds_list.append(out_unscaled)
                targets_list.append(target_unscaled)

                pbar.update(1)

    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    if return_latents:
        latents = torch.cat(latents_list, dim=0)
        return preds, targets, latents
    else:
        return preds, targets


# Get predictions using the test_model_jit_with_parity function
preds, _ = test_model_jit_with_parity(
    model=model,
    test_data_loader=new_loader,
    device=device,
    batch_size=len(new_data),
    model_needs_frag=True,
)

# Print the predictions
for i, smile in enumerate(new_smiles):
    print(f"SMILES: {smile}, Predicted value: {unstandardize(preds[i].item(), mean_target, std_target)}")

####### Example for rescaling the MAE prediction ##########

