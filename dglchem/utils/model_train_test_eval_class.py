from time import sleep

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

__all__ = [
    'train_model'
]

# Should be used after the model is initialized
def train_model(model, loss_func, optimizer, train_data_loader, device = None, epochs = 50,  batch_size = 32,
                early_stopping: bool = True, patience = 3):

    device = torch.device('cpu') if device is None else device

    if not isinstance(train_data_loader, DataLoader):
        train_data_loader = DataLoader(train_data_loader, batch_size = batch_size)

    model.train()
    loss = []

    with tqdm(total = epochs) as pbar:
        for i in range(epochs):
            for idx, batch in enumerate(train_data_loader):
                optimizer.zero_grad()
                out = model(batch.to(device))
                loss_val = loss_func(batch.y, out)

                loss_cpu = loss_val.detach().cpu().numpy()
                loss.append(loss_cpu)

                loss_val.backward()
                optimizer.step()

            if i%5 == 0:
                pbar.set_description(f"epoch={i}, loss={loss_cpu:.1f}")

            pbar.update(1)

    return loss




