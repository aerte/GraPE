import json
from grape.models import *
from grape.utils import load_dataset_from_excel, return_hidden_layers, set_seed, EarlyStopping
from grape.utils import train_model, test_model, pred_metric
import torch
from torch.optim import lr_scheduler
import pandas as pd
import numpy as np

def load_model(model_name, config, device = None):
    """ Function to load a model based on a model name and a config dictionary. Is supposed to reduce clutter in the trainable function.

    model_name: str
        A string that defines the model to be loaded. Options:
        * "AFP"
        * "MPNN"
        * "DMPNN"
        * "MEGNet"
    config : ConfigSpace
    """

    mlp_out = return_hidden_layers(config['mlp_layers'])

    if model_name == "AFP":
        return AFP(node_in_dim=44, edge_in_dim=12, num_layers_mol=config["afp_mol_layers"],
                    num_layers_atom=config["depth"], rep_dropout=config["dropout"],
                    hidden_dim=config["gnn_hidden_dim"],
                    mlp_out_hidden=mlp_out)
    elif model_name == "MPNN":
        return MPNN_Model(node_in_dim=44, edge_in_dim=12, num_layers=config["depth"],
                          mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                          node_hidden_dim=config["gnn_hidden_dim"])
    elif model_name == "DMPNN":
        return DMPNNModel(node_in_dim=44, edge_in_dim=12, node_hidden_dim=config["gnn_hidden_dim"],
                          depth=config["depth"], dropout=0, mlp_out_hidden=mlp_out,
                          rep_dropout=config["dropout"])
    elif model_name == "MEGNet":
        return MEGNet_gnn(node_in_dim=44, edge_in_dim=12, node_hidden_dim=config["gnn_hidden_dim"],
                          edge_hidden_dim=config["edge_hidden_dim"], depth=config["depth"],
                          mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                          device=device)




if __name__ == "__main__":

    import argparse
    set_seed(42)
    root = '/Users/faerte/Desktop/grape/notebooks/data_splits.xlsx'

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default='free', choices=['mp', 'logp', 'qm', 'free'],
                        help='the data that will be trained on (default: %(default)s)')
    data_name = parser.parse_args().data

    if data_name == 'free':
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="FreeSolv", is_dmpnn=True,
                                                                     return_dataset=True)
        data_name = "FreeSolv"
    elif data_name == 'mp':
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="Melting Point", is_dmpnn=True,
                                                                     return_dataset=True)
        data_name = "Melting Point"
    elif data_name == 'qm9':
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="Heat capacity", is_dmpnn=True,
                                                                     return_dataset=True)
        data_name = "QM9"
    else:
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="LogP", is_dmpnn=True,
                                                                     return_dataset=True)
        data_name = "LogP"

    #model_names = ["AFP", "MPNN", "DMPNN", "MEGNet"]
    model_names = ["MPNN"]
    df = pd.DataFrame()

    for name in model_names:
        # Load config from best set


        with open('results/best_hyperparameters_' + name +'_'+ data_name + '.json', 'r') as f:
            config = json.load(f)["best_config"]

        # Load model and
        model = load_model(name, config, device = "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
        early_Stopper = EarlyStopping(patience=30, model_name=name, skip_save=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'],
                                                   min_lr=0.0000000000001, patience=30)
        loss_fn = torch.nn.functional.l1_loss

        # Train and test
        train_loss, test_loss = train_model(model=model, loss_func=loss_fn,
                                            optimizer=optimizer, scheduler=scheduler,
                                            train_data_loader=train_set, val_data_loader=val_set,
                                            batch_size=300, epochs=300)

        train_preds = test_model(model=model, loss_func=None, test_data_loader=train_set)
        val_preds = test_model(model=model, loss_func=None, test_data_loader=val_set)
        test_preds = test_model(model=model, loss_func=None, test_data_loader=test_set)

        prediction_types = ['mae', 'rsme', 'mdape', 'r2']
        type_names = ['MAE', 'RSME', 'MDAPE', 'r2']

        df_add = pd.DataFrame()

        # Generate results
        for pred_type, type_ in zip(prediction_types, type_names):
            train_metric = pred_metric(train_preds, train_set, metrics=[pred_type], rescale_data=data)[pred_type]
            val_metric = pred_metric(val_preds, val_set, metrics=[pred_type], rescale_data=data)[pred_type]
            test_metric = pred_metric(test_preds, test_set, metrics=[pred_type], rescale_data=data)[pred_type]

            df_add['Test '+ type_] = test_metric
            df_add['Overall '+ type_] = (train_metric + val_metric + test_metric) / 3

        df = pd.concat([df, df_add])

    with pd.ExcelWriter('results_sheet.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, index=False)







