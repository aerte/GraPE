import pandas as pd
from grape.utils import DataSet, split_data, train_model, test_model, pred_metric, RevIndexedDataset
from grape.models import MPNN, MPNN_Model, DMPNNModel
import random
import dgl
import torch
import os


df = pd.read_excel('/Users/faerte/Desktop/grape/data-sets/BMP_model_predictions.xlsx')
# Load it into DataSet format
datar = DataSet(smiles=df.SMILES, target=df.Target, filter=False)

df_test = pd.read_excel('/Users/faerte/Desktop/grape/data-sets/n-alkanes.xlsx')
data = DataSet(smiles=df_test.SMILES, target=df_test.nC, filter=False)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # torch.set_num_threads(1)
    # np.random.seed(seed) # turn it off during optimization
    random.seed(seed)
    torch.manual_seed(seed)  # annote this line when ensembling
    dgl.random.seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # Faster than the command below and may not fix results
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

set_seed(0)

# MPNN
node_hidden_dim = 80
batch_size = 32

model_message = MPNN(num_layers=1, node_hidden_dim=node_hidden_dim,
                     node_in_dim=data.num_node_features, edge_in_dim=data.num_edge_features)

model_mpnn = MPNN_Model(node_in_dim=data.num_node_features, edge_in_dim=data.num_edge_features,
                   message_nn=model_message, node_hidden_dim=node_hidden_dim, set2set_steps=3)

model_mpnn.load_state_dict(torch.load('best_model_mpnn.pt'))

alkanes_pred_mpnn = test_model(model=model_mpnn, loss_func='mae',test_data_loader=data.data)

## D-MPNN

# Define model
node_hidden_dim = 114
batch_size = 32
depth = 4

model_dmpnn = DMPNNModel(node_hidden_dim, data.num_node_features, data.num_edge_features, depth=depth,
                   dropout=0.15)

model_dmpnn.load_state_dict(torch.load('best_model_dmpnn.pt'))

data_dmpnn = RevIndexedDataset(data.data)

alkanes_pred_dmpnn = test_model(model=model_dmpnn, loss_func='mae',test_data_loader=data_dmpnn)

## Conversion
alkanes_pred_mpnn = datar.rescale_data(alkanes_pred_mpnn.detach().numpy())+273.15
alkanes_pred_dmpnn = datar.rescale_data(alkanes_pred_dmpnn.detach().numpy())+273.15

last_point = df_test.index[-1]+1
df_tm = pd.read_excel('/Users/faerte/Desktop/grape/data-sets/tm_gc.xlsx')
prediction = df_tm.prediction

### Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.rcParams['font.size'] = 16
plt.scatter(df_test.nC, alkanes_pred_mpnn, color='blue', label='MPNN prediction',linewidth=0.1,marker='P')
plt.scatter(df_test.nC, alkanes_pred_dmpnn, color='red', label='DMPNN prediction',linewidth=0.1,marker='X')
plt.scatter(df_test.nC, prediction[:last_point], color='green',label='GC prediction',linewidth=0.1,marker='s')
plt.xlabel('Number of [C]', fontsize=16)
plt.ylabel('Temperature in Kelvin', fontsize=16)
plt.tight_layout()
plt.legend()
plt.show()