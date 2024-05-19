from grape.datasets import FreeSolv
from grape.models import AFP, MEGNet_gnn, DMPNNModel, MPNN_Model, Weave_Model
from grape.utils import train_model, EarlyStopping, RevIndexedSubSet
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

batch_size = 32

data = FreeSolv(split_type='random')
train, val, test = data.train, data.val, data.test
train, val, test = RevIndexedSubSet(train), RevIndexedSubSet(val), RevIndexedSubSet(test)

model1 = AFP(node_in_dim=44, edge_in_dim=12, mlp_out_hidden=[512,256, 128])
model2 = MEGNet_gnn(node_in_dim=44, edge_in_dim=12, mlp_out_hidden=[512,256, 128])
model3 = DMPNNModel(node_in_dim=44, edge_in_dim=12, mlp_out_hidden=[512, 256, 128])
model4 = MPNN_Model(node_in_dim=44, edge_in_dim=12, mlp_out_hidden=[512, 256, 128])
model5 = Weave_Model(node_in_dim=44, edge_in_dim=12, mlp_out_hidden=[512, 256, 128])

models = [model2]


device = torch.device('cpu')

loss_func = nn.functional.mse_loss

optimizers, schedulers, earlyStoppers, schedulers, losses = [], [], [], [], []
for model in models:
    optimizers.append(torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6))
    earlyStoppers.append(EarlyStopping(patience=50, model_name='random'))
    schedulers.append(lr_scheduler.ReduceLROnPlateau(optimizers[-1],
                                                     mode='min', factor=0.9, min_lr=0.0000000000001, patience=30))

    loss = train_model(model=model,
                       loss_func='mse',
                       optimizer=optimizers[-1],
                       train_data_loader=train,
                       val_data_loader=val,
                       batch_size=batch_size,
                       epochs=500,
                       early_stopper=earlyStoppers[-1],
                       scheduler=schedulers[-1])

    losses.append(loss)