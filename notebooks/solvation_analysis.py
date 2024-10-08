import torch
from torch.utils.data import random_split, DataLoader, Subset
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import lr_scheduler
from grape_chem.models import DMPNN, AFP
from grape_chem.utils.split_utils import RevIndexedSubSet
from grape_chem.utils.data import DataSet
from grape_chem.utils.model_utils import train_model, test_model, pred_metric, return_hidden_layers, set_seed
from grape_chem.utils.featurizer import AtomFeaturizer
from grape_chem.utils import EarlyStopping

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from typing import List, Union

# Function to calculate molecular weight
def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)


set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### PARAMETERS
epochs = 1 #50
batch_size = 64 #64
scheduler_batch = 50 #50
hidden_dim = 300 #300
depth = 3 #3
dropout = 0.0 #0.0
patience = 30
patience_scheduler = 5
warmup_epochs = 2
init_lr = 0.0001 # chemprop default 0.0001
max_lr = 0.002 # chemprop default 0.001
final_lr = 0.0001 # 0.0001
weight_decay = 0.0 #0.0
model_seed = 50

mlp_layers = 4
atom_layers = 3
mol_layers = 3

num_global_feats = 0

learning_rate = 0.0001
### PARAMETERS

root = 'C:\\Users\\Thoma\\GraPE\\notebooks\\solvation\\solvation.csv'
#input = "C:\\Users\\Thoma\\chempropv1\\input.csv"

df = pd.read_csv(root, sep=';', encoding='utf-8')
df = df[:100]
print("head df", df.head())
print("INITIAL SIZE DATASET", len(df))


smiles = df['SMILES']
temperature = df['Temperature'].to_numpy()
target = df['Energy']

molecular_weights = np.array([calculate_molecular_weight(s) for s in smiles])

global_feats = np.column_stack((temperature, molecular_weights))
#global_feats = None
#print("Global feats shape: ", global_feats.shape, "global feats head: ", global_feats[:5])

allowed_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'I']
atom_feature_list = ['chemprop_atom_features']
bond_feature_list = ['chemprop_bond_features']

dataset = DataSet(smiles=smiles, target=target, global_features=global_feats, allowed_atoms=allowed_atoms, atom_feature_list=atom_feature_list, bond_feature_list=bond_feature_list, log=False, only_organic=False, filter=True, allow_dupes=True)

#dataset = DataSet(smiles=smiles, target=target, global_features=global_feats)
print("Len dataset: ", len(dataset))

# DMPNN
train_data, val_data, test_data = dataset.split_and_scale(split_frac=[0.8,0.1,0.1], scale=True, seed=model_seed, is_dmpnn=True, split_type='random')

# AFP
#train_data, val_data, test_data = dataset.split_and_scale(split_frac=[0.8,0.1,0.1], scale=True, seed=model_seed, is_dmpnn=False, split_type='random')
 

sample = dataset[20] 
print("Dataset sample:")
print("sample:",sample)
print("sample.x:",sample.x)
print("sample.edge_attr:",sample.edge_attr)
node_in_dim = sample.x.shape[1]
edge_in_dim = sample.edge_attr.shape[1]

mlp = return_hidden_layers(mlp_layers)
print("mlp layers:", mlp)
model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, dropout=dropout, depth=depth, node_hidden_dim=hidden_dim, num_global_feats=num_global_feats, mlp_out_hidden=512)
#model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, out_dim=1, num_global_feats=num_global_feats, hidden_dim=hidden_dim,
#            mlp_out_hidden=mlp, num_layers_atom=atom_layers, num_layers_mol=mol_layers)
print('Full model:\n--------------------------------------------------')
print(model)
num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of learnable weights in the model: {num_learnable_params}')
print('Full model:\n--------------------------------------------------')
model = model.to(device)

# DMPNN requires Reverse Indexed Subset
train_data, val_data, test_data = RevIndexedSubSet(train_data), RevIndexedSubSet(val_data), RevIndexedSubSet(test_data)


# criterion = MSELoss()
# # optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
# # Early stopper
# early_stopper = EarlyStopping(patience=patience)
# # Scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, min_lr=0.000000001, patience=patience_scheduler)
# # TRAINING
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_stopper = EarlyStopping(patience=patience, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, min_lr=0.0000000000001,
                                           patience=patience)

loss_func = torch.nn.functional.l1_loss

train_model(model=model, loss_func=loss_func, optimizer=optimizer, train_data_loader=train_data, val_data_loader=val_data, epochs=epochs, batch_size=batch_size, early_stopper=early_stopper, scheduler=scheduler, device=device)

# TESTING
pred = test_model(model=model, test_data_loader=test_data, device=device)
test_targets = [data.y for data in test_data]
targets = torch.cat(test_targets, dim=0)

# METRICS
pred_metric(prediction=pred, target=targets, metrics='all', rescale_data=dataset)

# Parity plot
import matplotlib.pyplot as plt
import seaborn as sns

pred =  dataset.rescale_data(pred)
targets = dataset.rescale_data(targets)

pred = pred.cpu().detach().numpy()
targets = targets.cpu().detach().numpy()

plt.figure(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x=targets, y=pred, alpha=0.7)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Parity plot test data')
plt.show()

#
# print("TESTING on new data")
# simons_results = 'C:\\Users\\Thoma\\chempropv1\\datasets\\results_simon.xlsx'

# test = pd.read_excel(simons_results, sheet_name='NoRings', engine='openpyxl')

# df_test = test[['SMILES', 'Temperature', 'Target']]
# print(df_test.head())
# temperature = df_test['Temperature'].to_numpy()
# df_test = df_test.drop(columns=['Temperature'])
# molecular_weights_test = np.array([calculate_molecular_weight(s) for s in df_test['SMILES']])
# global_feats_test = np.column_stack((temperature, molecular_weights_test))

# dataset = DataSet(smiles=df_test['SMILES'], target=df_test['Target'], global_features=global_feats_test)
# train_data, val_data, test_data = dataset.split_and_scale(split_frac=[0.0,0.0,1.0], scale=True, seed=model_seed, is_dmpnn=True, split_type='random')

# test_simon = RevIndexedSubSet(test_data)

# pred_simon = test_model(model=model, test_data_loader=test_simon, device=device)

# test_targets = [data.y for data in test_simon]
# targets = torch.cat(test_targets, dim=0)
# pred_metric(prediction=pred_simon, target=targets, metrics='all', rescale_data=dataset)

# TESTING
pred = test_model(model=model, test_data_loader=train_data, device=device)
test_targets = [data.y for data in train_data]
targets = torch.cat(test_targets, dim=0)

# METRICS
pred_metric(prediction=pred, target=targets, metrics='all', rescale_data=dataset)

# Parity plot
import matplotlib.pyplot as plt
import seaborn as sns

pred =  dataset.rescale_data(pred)
targets = dataset.rescale_data(targets)

pred = pred.cpu().detach().numpy()
targets = targets.cpu().detach().numpy()

plt.figure(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x=targets, y=pred, alpha=0.7)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Parity plot train data')
plt.show()

# TESTING
pred = test_model(model=model, test_data_loader=val_data, device=device)
test_targets = [data.y for data in val_data]
targets = torch.cat(test_targets, dim=0)

# METRICS
pred_metric(prediction=pred, target=targets, metrics='all', rescale_data=dataset)

# Parity plot
import matplotlib.pyplot as plt
import seaborn as sns

pred =  dataset.rescale_data(pred)
targets = dataset.rescale_data(targets)

pred = pred.cpu().detach().numpy()
targets = targets.cpu().detach().numpy()

plt.figure(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x=targets, y=pred, alpha=0.7)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Parity plot val data')
plt.show()

# TESTING
pred = test_model(model=model, test_data_loader=dataset, device=device)
test_targets = [data.y for data in dataset]
targets = torch.cat(test_targets, dim=0)

# METRICS
pred_metric(prediction=pred, target=targets, metrics='all', rescale_data=dataset)

# Parity plot
import matplotlib.pyplot as plt
import seaborn as sns

pred =  dataset.rescale_data(pred)
targets = dataset.rescale_data(targets)

pred = pred.cpu().detach().numpy()
targets = targets.cpu().detach().numpy()

plt.figure(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x=targets, y=pred, alpha=0.7)
plt.xlabel('True')
plt.ylabel('Predicted')
plt.title('Parity plot all data')
plt.show()