from grape.models import AFP
from grape.datasets import FreeSolv
from grape.utils import DataSet, train_model, EarlyStopping, split_data, test_model, pred_metric
from torch.optim import lr_scheduler
import numpy as np
import torch

## Install GraPE with: pip install "git+https://github.com/aerte/GraPE.git#subdirectory=python"

##########################################################################################
#####################    Data Input Region  ##############################################
##########################################################################################

free = FreeSolv()
global_feats = np.random.randn(len(free))
smiles, target = free.smiles, free.target

# num_global_feats is the dimension of global features per observation
model = AFP(node_in_dim=44, edge_in_dim=12, num_global_feats=1)

epochs = 500
batch_size = 128

############################################################################################
############################################################################################
############################################################################################


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data = DataSet(smiles=smiles, target=target, global_features=global_feats)
train, val, test = split_data(data, split_type='random', split_frac=[0.8,0.1,0.1])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
early_Stopper = EarlyStopping(patience=30, model_name='random', skip_save=True)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9999, min_lr=0.0000000000001, patience=30)
loss_func = torch.nn.functional.l1_loss

model.to(device)
train_model(model=model, loss_func=loss_func, optimizer=optimizer, train_data_loader=train,
            val_data_loader=val, epochs=500, device=device, batch_size=128)
pred = test_model(model=model, loss_func=loss_func, test_data_loader=test, device=device, batch_size=128)
pred_metric(prediction=pred, target=test.y)
