from grape.datasets import BradleyDoublePlus
from grape.utils import split_data
data = BradleyDoublePlus()

train, val, test = split_data(data, split_type='random', random_state=42)

from grape.models import MPNN, MPNN_Model, MGConv, SimpleGNN
import torch

node_hidden_dim = 64
batch_size = 32

model_message = MPNN(num_layers=1, node_hidden_dim=node_hidden_dim,
                node_in_dim=data.num_node_features, edge_in_dim=data.num_edge_features)


model = MPNN_Model(node_in_dim=data.num_node_features, edge_in_dim=data.num_edge_features,
                   message_nn=model_message, node_hidden_dim=node_hidden_dim, set2set_steps=3)

from torch import nn

loss_func = nn.functional.mse_loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

from grape.utils import EarlyStopping
early_stopper = EarlyStopping(patience=50)

from torch.optim import lr_scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, min_lr=0.0000000000001, patience=30)

from grape.utils import train_model

train_loss, val_loss = train_model(model = model,
                                   loss_func = 'mse',
                                   optimizer = optimizer,
                                   train_data_loader= train,
                                   val_data_loader = val,
                                   batch_size=batch_size,
                                   epochs=500,
                                   early_stopper=early_stopper,
                                    scheduler=scheduler)


