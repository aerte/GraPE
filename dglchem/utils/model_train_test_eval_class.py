from time import sleep

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

__all__ = [
    'train_model'
]

# Should be used after the model is initialized
def train_model(model, loss_func, optimizer, train_data_loader, test_data_loader, device = None, epochs = 50,  batch_size = 32,
                early_stopping: bool = True, patience = 3):

    device = torch.device('cpu') if device is None else device

    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size)

    if not isinstance(train_data_loader, DataLoader) and test_data_loader:
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size)

    model.train()
    train_loss = []
    test_loss = []

    loss_cut = float('inf')
    counter = 0

    with tqdm(total = epochs) as pbar:
        for i in range(epochs):
            temp = np.zeros(len(train_data_loader))
            for idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                out = model(batch.to(device))
                loss_val = loss_func(batch.y, out)

                temp[idx] = loss_val.detach().cpu().numpy()


                loss_val.backward()
                optimizer.step()

            loss_train = np.mean(temp)
            train_loss.append(loss_train)

            temp = np.zeros(len(test_data_loader))
            for idx, batch in enumerate(test_data_loader):
                out = model(batch.to(device))
                temp[idx] = loss_func(batch.y, out).detach().cpu().numpy()

            loss_test = np.mean(temp)
            test_loss.append(loss_test)

            if i%5 == 0:
                pbar.set_description(f"epoch={i}, training loss= {loss_train:.1f}, validation loss= {loss_test}")

            if loss_test < loss_cut:
                loss_cut = loss_test
            else:
                counter+=1

            if counter==3:
                print('Model hit early stop threshold. Ending training.')
                break


            pbar.update(1)

    return train_loss, test_loss




