import torch
import pandas as pd
import numpy as np
from grape_chem.models import DMPNN, AFP
from grape_chem.utils import train_model, test_model, pred_metric, return_hidden_layers, RevIndexedSubSet, DataSet
from grape_chem.utils.model_utils import set_seed
from grape_chem.utils.model_utils import load_model 
from grape_chem.utils.featurizer import AtomFeaturizer
from grape_chem.utils import EarlyStopping

from torch.optim import lr_scheduler
from rdkit import Chem
from rdkit.Chem import Descriptors

# Function to calculate molecular weight
def calculate_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

### PARAMETERS
epochs = 1 #50
batch_size = 64 #64
scheduler_batch = 50 #50
hidden_dim = 300 #300
depth = 3 #3
dropout = 0.0 #0.0
patience = 4
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

model_name = '15-09-2024-afp'
### PARAMETERS

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

allowed_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'I']
atom_feature_list = ['chemprop_atom_features']
bond_feature_list = ['chemprop_bond_features']

dippr_dataset = True
load_data = True
if load_data:
    if dippr_dataset:
        dippr = 'C:\\Users\\Thoma\\GraPE\\notebooks\\dippr.csv'
        df = pd.read_csv(dippr, sep=';', encoding='latin')
        smiles = df['SMILES']
        target = df['Const_Value']
        num_global_feats = 1
        molecular_weights = np.array([calculate_molecular_weight(s) for s in smiles])
        global_features = np.array(molecular_weights)
        data = DataSet(smiles=smiles, target=target, global_features=global_features, allowed_atoms=allowed_atoms, atom_feature_list=atom_feature_list, bond_feature_list=bond_feature_list, log=False, only_organic=False, filter=True, allow_dupes=True)
        sample = data[50]
        node_in_dim = sample.x.shape[1]
        edge_in_dim = sample.edge_attr.shape[1]
        mlp = return_hidden_layers(mlp_layers)
        print("Dataset length: ", len(data))
        train_data, val_data, test_data = data.split_and_scale(split_frac=[0.8,0.1,0.1], scale=True, seed=model_seed, is_dmpnn=False, split_type='random')
        if 'dmpnn' in model_name:
            train_data, val_data, test_data = RevIndexedSubSet(train_data), RevIndexedSubSet(val_data), RevIndexedSubSet(test_data)        
        dataset_dict = {'allowed_atoms': data.allowed_atoms, 'atom_feature_list': data.atom_feature_list, 'bond_feature_list': data.bond_feature_list, 'data_mean': data.mean, 'data_std': data.std, 'data_name': data.data_name}
        print("Dataset dict: data mean:", dataset_dict['data_mean'], "data std:", dataset_dict['data_std'])
    else:
        solvation = 'C:\\Users\\Thoma\\GraPE\\notebooks\\solvation\\solvation.csv'
        df = pd.read_csv(solvation, sep=';', encoding='utf-8')
        df = df[:1000]
        smiles = df['SMILES']
        temperature = df['Temperature'].to_numpy()
        target = df['Energy']
        num_global_feats = 2

### Init model to save a valid checkpoint###
model = AFP(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, out_dim=1, dataset_dict=dataset_dict)
#model = DMPNN(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, node_hidden_dim=hidden_dim, rep_dropout=dropout, mlp_out_hidden=mlp, num_global_feats=num_global_feats, dataset_dict=dataset_dict)
print('Full model:\n--------------------------------------------------')
print(model)
num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of learnable weights in the model: {num_learnable_params}')
print('Full model:\n--------------------------------------------------')
model = model.to(device)

### Train the model ###
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
early_stopper = EarlyStopping(patience=patience, model_name=model_name)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9999, min_lr=0.0000000000001,
                                           patience=patience)

loss_func = torch.nn.functional.l1_loss

train_model(model=model, loss_func=loss_func, optimizer=optimizer, train_data_loader=train_data, val_data_loader=val_data, epochs=epochs, batch_size=batch_size, early_stopper=early_stopper, scheduler=scheduler, device=device)

#global_feats = np.column_stack((temperature, molecular_weights))
global_feats = None
if global_feats is not None:
    print("Global feats shape: ", global_feats.shape, "global feats head: ", global_feats[:5])


#### Load model ####
model_path =  'C:\\Users\\Thoma\\code\\GraPE\\'+model_name+'.pt'
model_class = 'AFP'
property_name = 'Energy'
property_unit = 'kcal/mol'
print("Device: ", device)
model, dataset = load_model(model_class, model_path, device)

# Prepare input data (SMILES)
input_smiles = ['CC', 'CCc1ccccn1', 'C=C(C)C#C', 'CC(C)CCCC']  # Replace with actual input
df_input = pd.DataFrame(input_smiles, columns=['SMILES'])
smiles = df_input['SMILES']

print("Device: ", device)
preds = dataset.predict_smiles(smiles, model)
print("PREDICTIONS: ", preds)

# # Perform inference
# with torch.no_grad():
#     output = model(dataset)
# # regression
# print(output)