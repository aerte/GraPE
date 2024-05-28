############################ Optimization script ############################################

from ray.air import RunConfig
import dgl
import os
from grape.models import AFP, MPNN_Model, DMPNNModel, MEGNet_gnn
from grape.utils import EarlyStopping, train_model
from functools import partial
import torch
from torch.optim import lr_scheduler
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune import Tuner
from ray import tune, train
import numpy as np
import pandas as pd
from grape.utils import DataSet, split_data, train_epoch, val_epoch, RevIndexedSubSet
from torch_geometric.loader import DataLoader
from grape.utils import pred_metric

import ConfigSpace as CS

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # torch.set_num_threads(1)
    # np.random.seed(seed) # turn it off during optimization
    np.random.seed(seed)
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

def return_hidden_layers(num):
    """ Returns a list of hidden layers, starting from 2**num*32, reducing the hidden dim by 2 every step.

    Example
    --------

    >>>return_hidden_layers(3)

    [256, 128, 64]
    """
    return [2**i*32 for i in range(num, 0,-1)]

def load_dataset_from_excel(dataset, is_dmpnn = False):
    """
    dataset: str
        A string that defines what dataset should be used, specifically loaded from a data-splits sheet. Options:
        * "Melting Point"
        * "LogP"
        * "Heat capacity"
        * "FreeSolv"
    is_dmpnn: bool
        If data for DMPNN has to be loaded. Default: False
    """
    
    df = pd.read_excel('/zhome/4a/a/156124/GraPE/notebooks/data_splits.xlsx', sheet_name=dataset)

    data = DataSet(smiles=df.SMILES, target=df.Target, filter=False, scale=True)


    # convert given labels to a list of numbers and split dataset
    labels = df.Split.apply(lambda x: ['train', 'val', 'test'].index(x)).to_list()

    train_set, val_set, test_set = split_data(data, custom_split=labels)
    
    # In case data for DMPNN has to be loaded:
    if is_dmpnn:
        train_set, val_set, test_set = RevIndexedSubSet(train_set), RevIndexedSubSet(val_set), RevIndexedSubSet(test_set)

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





def trainable(config: dict, data_name:str, model_name:str, is_dmpnn:bool, device:torch.device):
        """ The trainable for Ray-Tune.

        Parameters
        -----------
            config: dict
                A ConfigSpace dictionary adhering to the required parameters in the trainable. Defines the search space of the HO.
            data_name: str
                The data to be used.
            model_name: str
                The model to be loaded.
        """

        ################### Loading the data #########################################################################

        if data_name == 'free':
            train_set, val_set, _ = load_dataset_from_excel("FreeSolv",is_dmpnn=is_dmpnn)
        elif data_name == 'mp':
            train_set, val_set, _ = load_dataset_from_excel("Melting Point",is_dmpnn=is_dmpnn)
        elif data_name == 'qm9':
            train_set, val_set, _ = load_dataset_from_excel("Heat capacity",is_dmpnn=is_dmpnn)
        else:
            train_set, val_set, _ = load_dataset_from_excel("LogP",is_dmpnn=is_dmpnn)

        ################### Defining the model #########################################################################
        
        model = load_model(model_name=model_name, config=config, device = device)
        
        model.to(device=device)

        ################################################################################################################

        optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
        early_Stopper = EarlyStopping(patience=30, model_name='random', skip_save=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'],
                                                min_lr=0.0000000000001, patience=30)
        
        loss_function = torch.nn.functional.l1_loss

        train_data = DataLoader(train_set, batch_size = 300)
        val_data = DataLoader(val_set, batch_size = 300)

        iterations = 300

        for i in range(iterations):

            model.train()
            train_loss = train_epoch(model=model, loss_func=loss_function, optimizer=optimizer, train_loader=train_data, device=device)
            val_loss = val_epoch(model=model, loss_func=loss_function, val_loader=val_data, device=device)
            

            # We report the loss to ray tune every 15 steps, that way tune's scheduler can interfere
            if i%15 == 0:
                train.report({"mae_loss": val_loss})
                



if __name__ == '__main__':

    model_name = 'AFP'


    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default='free', choices=['mp', 'logp', 'qm', 'free'],
                        help='the data that will be trained on (default: %(default)s)')
    parser.add_argument('--samples', type=int, default=100,
                        help='the number of samples/instances that will be running (default: %(default)s)')
    parser.add_argument('--model', type=str, default='afp', choices=['afp','mpnn','dmpnn','megnet'],
                        help='the model to be used (default: %(default)s)')
    
    args = parser.parse_args()
    data_name = args.data
    n_samples = args.samples
    model_ = args.model
    is_dmpnn = False

    ################################# Selecting the options ######################################

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = 1
    else:
        device = torch.device("cpu")
        gpu = 0

    if data_name == 'free':
        dataset = 'FreeSolv'
    elif data_name == 'mp':
        dataset = 'Melting_Point'
    elif data_name == 'qm':
        dataset = 'QM9'
    else:
        dataset = 'LogP'

    if model_ == 'mpnn':
        model_name = "MPNN"
    elif model_ == "dmpnn":
        model_name = "DMPNN"
        is_dmpnn = True
    elif model_ == "megnet":
        model_name = "MEGNet"
    elif model_ == "afp":
        model_name = "AFP"
    else:
        model_name = "AFP"


    ################################# Search space ######################################

    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter("depth", lower=1, upper=5))
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter("gnn_hidden_dim", lower=32, upper=256))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter('initial_lr', lower=1e-5, upper=1e-1))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-1))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("lr_reduction_factor", lower=0.4, upper=0.99))
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter("dropout", lower=0., upper=0.4))
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter("mlp_layers", lower=1, upper=4))
    # If AFP is selected
    if model_name == "AFP":
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter("afp_mol_layers", lower=1, upper=4))
    elif model_name == "MEGNet":
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter("edge_hidden_dim", lower=32, upper=256))


    ################################# --------------------- ######################################

    my_trainable = partial(trainable, data_name=data_name, model_name=model_name, is_dmpnn=is_dmpnn,
                           device=device)

    trainable_with_resources = tune.with_resources(my_trainable, {"cpu":4, "gpu":gpu})

    ### Define search algorithm
    algo = TuneBOHB(config_space,mode='min', metric="mae_loss",)

    ## Get the trial control algorithm
    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=1000,
        )

    ## Initialize the tuner
    tuner = Tuner(my_trainable, tune_config=tune.TuneConfig(scheduler=scheduler,
                                            search_alg=algo,
                                            mode='min',
                                            metric="mae_loss",
                                            num_samples=n_samples),
                                            run_config=train.RunConfig(
                                                name="bohb_exp",
                                                stop={"training_iteration": 100})
    )


    result = tuner.fit()

    import json

    best_result = result.get_best_result(metric="mae_loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics

    results_to_save = {
        "best_config": best_config,
        "best_metrics": best_metrics
    }

    file_name = "/zhome/4a/a/156124/GraPE/notebooks/results/best_hyperparameters_" + model_name +"_"+dataset+".json"

    with open(file_name, "w") as file:
        json.dump(results_to_save, file, indent=4)