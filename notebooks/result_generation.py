import json
from grape.models import *
from grape.utils import return_hidden_layers, set_seed, EarlyStopping, rescale_arrays, DataSet, RevIndexedSubSet
from grape.utils import train_model, test_model, pred_metric
import torch
from torch.optim import lr_scheduler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def load_dataset_from_excel(file_path, dataset, is_dmpnn=False, is_megnet=False, return_dataset=False):
    """
    dataset: str
        A string that defines what dataset should be used, specifically loaded from a graphs-splits sheet. Options:
        * "Melting Point"
        * "LogP"
        * "Heat capacity"
        * "FreeSolv"
    is_dmpnn: bool
        If graphs for DMPNN has to be loaded. Default: False
    """

    df = pd.read_excel(file_path, sheet_name=dataset)

    data = DataSet(smiles=df.SMILES, target=df.Target, filter=False, scale=False)

    # convert given labels to a list of numbers and split dataset
    labels = df.Split.apply(lambda x: ['train', 'val', 'test'].index(x)).to_list()

    if is_megnet:
        data.generate_global_feats(seed=42)
    train_set, val_set, test_set = data.split_and_scale(custom_split=labels, scale=True)

    # In case graphs for DMPNN has to be loaded:
    if is_dmpnn:
        train_set, val_set, test_set = RevIndexedSubSet(train_set), RevIndexedSubSet(val_set), RevIndexedSubSet(
            test_set)

    if return_dataset:
        return train_set, val_set, test_set, data
    return train_set, val_set, test_set


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
        return MPNN(node_in_dim=44, edge_in_dim=12, num_layers=config["depth"],
                          mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                          node_hidden_dim=config["gnn_hidden_dim"])
    elif model_name == "DMPNN":
        return DMPNN(node_in_dim=44, edge_in_dim=12, node_hidden_dim=config["gnn_hidden_dim"],
                          depth=config["depth"], dropout=0, mlp_out_hidden=mlp_out,
                          rep_dropout=config["dropout"])
    elif model_name == "MEGNet":
        return MEGNet(node_in_dim=44, edge_in_dim=12, global_in_dim=1, node_hidden_dim=config["gnn_hidden_dim"],
                         edge_hidden_dim=config["edge_hidden_dim"], depth=config["depth"],
                         mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                         device=device)




if __name__ == "__main__":

    import argparse
    set_seed(42)
    root = '/Users/faerte/Desktop/grape/notebooks/data_splits.xlsx'

    parser = argparse.ArgumentParser()
    parser.add_argument('graphs', type=str, default='free', choices=['mp', 'logp', 'qm', 'free'],
                        help='the graphs that will be trained on (default: %(default)s)')
    parser.add_argument('--mode', type=str, default='pred', choices=['pred', 'metric','plots'],
                        help='the results generated (default: %(default)s)')

    data_name = parser.parse_args().data
    mode = parser.parse_args().mode

    if data_name == 'free':
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="FreeSolv", is_dmpnn=True,
                                                                     return_dataset=True, is_megnet=True)
        data_name = "FreeSolv"
        data_name_ = data_name

    elif data_name == 'mp':
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="Melting Point", is_dmpnn=True,
                                                                     return_dataset=True, is_megnet=True)
        data_name_ = "Melting_Point"
        data_name = "Melting Point"

    elif data_name == 'qm':
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="Heat capacity", is_dmpnn=True,
                                                                     return_dataset=True, is_megnet=True)
        data_name = "QM9"
        data_name_ = data_name

    else:
        train_set, val_set, test_set, data = load_dataset_from_excel(file_path=root, dataset="LogP", is_dmpnn=True,
                                                                     return_dataset=True, is_megnet=True)
        data_name = "LogP"
        data_name_ = data_name



    #model_names = ["AFP", "MPNN", "DMPNN", "MEGNet"]
    #model_names = ["MEGNet"]
    model_names = ["AFP", "MPNN", "DMPNN"]
    
    df = pd.DataFrame()

    if mode == 'pred':
        for name in model_names:
            # Load config from best set
            with open('results/best_hyperparameters_' + name +'_'+ data_name_ + '.json', 'r') as f:
                config = json.load(f)["best_config"]

            relative_model_name = 'results/'+name+'_'+data_name_

            # Load model and
            model = load_model(name, config, device = "cpu")
            optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
            early_Stopper = EarlyStopping(patience=30, model_name=relative_model_name, skip_save=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'],
                                                       min_lr=0.0000000000001, patience=30)
            loss_fn = torch.nn.functional.l1_loss

            # Train and test
            train_loss, test_loss = train_model(model=model, loss_func=loss_fn,
                                                optimizer=optimizer, scheduler=scheduler,
                                                train_data_loader=train_set, val_data_loader=val_set,
                                                batch_size=300, epochs=300,
                                                early_stopper=early_Stopper, model_name=relative_model_name)

            model.load_state_dict(torch.load(relative_model_name+'.pt'))

            train_preds = test_model(model=model, test_data_loader=train_set)
            val_preds = test_model(model=model, test_data_loader=val_set)
            test_preds = test_model(model=model, test_data_loader=test_set)

            trp, vap, tep =  rescale_arrays((train_preds, val_preds, test_preds), data=data)

            df_train = pd.DataFrame({name+' Prediction':trp})
            df_val = pd.DataFrame({name+' Prediction':vap})
            df_test = pd.DataFrame({name+' Prediction':tep})

            df_preds = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)

            df = pd.read_excel(root, sheet_name=data_name)
            df[name + " Prediction"] = df_preds[name + " Prediction"]

            with pd.ExcelWriter(root, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer:
                df.to_excel(writer, sheet_name=data_name, index=False)




    elif mode == 'metric':
        with (pd.ExcelWriter('/Users/faerte/Desktop/grape/notebooks/results/result_sheet.xlsx', engine='openpyxl')
              as writer):

            df_data = pd.read_excel(root, sheet_name=data_name, engine='openpyxl')

            train_target = df_data["Target"][df_data.Split == "train"].to_numpy()
            val_target = df_data["Target"][df_data.Split == "val"].to_numpy()
            test_target = df_data["Target"][df_data.Split == "test"].to_numpy()

            first = True

            for name in model_names:
                df_temp = pd.DataFrame({"Model":[name]})

                train_pred = df_data[name + " Prediction"][df_data.Split=="train"].to_numpy()
                val_pred = df_data[name + " Prediction"][df_data.Split=="val"].to_numpy()
                test_pred = df_data[name + " Prediction"][df_data.Split=="test"].to_numpy()

                type_names = ['MAE', 'RMSE', 'MDAPE', 'R2']
                prediction_types = ['mae', 'rmse', 'mdape', 'r2']

                # Generate results
                for pred_type, type_ in zip(prediction_types, type_names):
                    train_metric = pred_metric(train_pred, train_target, metrics=[pred_type])[pred_type]
                    val_metric = pred_metric(val_pred, val_target, metrics=[pred_type])[pred_type]
                    test_metric = pred_metric(test_pred, test_target, metrics=[pred_type])[pred_type]

                    df_temp['Test '+ type_] = test_metric
                    df_temp['Overall '+ type_] = (train_metric + val_metric + test_metric) / 3

                if first:
                    first = False
                    df = df_temp
                else:
                    df = pd.concat([df, df_temp],axis=0,ignore_index=True)

            print(df)
            df.to_excel(writer, sheet_name=data_name, index=False)

    elif mode == 'plots':

        #rcParams.update({'figure.autolayout': True})
        plt.rcParams.update({'font.size': 22})

        title_names = dict({'mp': 'Melting Point',
                            'qm':'Heat Capacity',
                            'free':'FreeSolv',
                            'logp':'Logp'})


        true_labels = dict({'mp': '$T_{m}$(exp) in $[Celcius]$ \n (a)',
                            'logp': '$\log K_{ow}$(exp), ratio',
                            'free': '$H_{hyd}$(exp) $[\\frac{kJ}{mol}]$',
                            'qm': '$c_v$(exp) $[\\frac{\\text{cal}}{\\text{mol} K}]$'})

        pred_labels = dict({'mp': '$T_{m}$(pred) in $[Celcius]$',
                            'logp': '$\log K_{ow}$(pred), ratio',
                            'free': '$H_{hyd}$(pred) $[\\frac{kJ}{mol}]$',
                            'qm': '$c_v$(pred) $[\\frac{\\text{cal}}{\\text{mol} K}]$'})

        plot_name = parser.parse_args().data
        title_name, true_label, pred_label = title_names[plot_name], true_labels[plot_name], pred_labels[plot_name]

        df_data = pd.read_excel(root, sheet_name=data_name, engine='openpyxl')
        test_target = df_data["Target"][df_data.Split == "test"].to_numpy()



        #################### Parity Plot ###################
        fig, ax = plt.subplots(figsize = (10,10))

        min_val, max_val = np.min(test_target), np.max(test_target)

        styles = ['x', 's', 'o', '*']
        colors = ['b','g','r','orange']
        widths = [2, 0.7, 0.7, 0.7]

        for name, style, color, width in zip(model_names, styles, colors, widths):
            test_pred = df_data[name + " Prediction"][df_data.Split == "test"].to_numpy()

            min_val = min(np.min(test_pred), min_val)
            max_val = max(np.max(test_pred), np.max(max_val))

            ax.scatter(test_target, test_pred, linewidth=width, marker=style, color=color, label=name)

        ax.axline((0, 0), slope=1, color='black')
        ax.set_xlabel(true_label)
        ax.set_ylabel(pred_label)
        plt.xlim(min_val-50, max_val+50)
        plt.ylim(min_val-50, max_val+50)
        plt.legend()
        plt.title(title_name)
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        fig.savefig(fname=f'results/parity_plot_'+plot_name+'.svg', format='svg')
        plt.show()








