############################ Optimization script ############################################

from ray.air import RunConfig
import dgl
import os
from grape.models import AFP
from grape.utils import EarlyStopping, train_model
from grape.datasets import FreeSolv
from functools import partial
import torch
from torch.optim import lr_scheduler
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune import Tuner
from ray import tune, train
import numpy as np
import pandas as pd
from grape.utils import DataSet, split_data

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

def load_dataset_from_excel(dataset):
    """
    dataset: str
        A string that defines what dataset should be used, specifically loaded from a data-splits sheet. Options:
        * "Melting Point"
        * "LogP"
        * "Heat capacity"
        * "FreeSolv"
    """
    
    df = pd.read_excel('data_splits.xlsx', sheet_name=dataset)

    data = DataSet(smiles=df.SMILES, target=df.Target, filter=False, scale=True)


    # convert given labels to a list of numbers and split dataset
    labels = df.Split.apply(lambda x: ['train', 'val', 'test'].index(x)).to_list()

    train_set, val_set, test_set = split_data(data, custom_split=labels)

    return train_set, val_set, test_set



def trainable(config: dict):
        """ The trainable for Ray-Tune.

        Parameters
        -----------
            config: dict
                A ConfigSpace dictionary adhering to the required parameters in the trainable. Defines the search space of the HO.
        """

        ################### Defining the model #########################################################################

        mlp_out = return_hidden_layers(config['mlp_layers'])

        data = FreeSolv(split_type='random', split_frac=[0.8, 0.1, 0.1])
        train_set, val_set, test_set = data.train, data.val, data.test

        model = AFP(node_in_dim=44, edge_in_dim=12, num_layers_mol=config["afp_mol_layers"],
                    num_layers_atom=config["depth"],
                    hidden_dim=config["gnn_hidden_dim"],
                    mlp_out_hidden=mlp_out)
        
        model.to(device=device)

        ################################################################################################################

        optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
        early_Stopper = EarlyStopping(patience=30, model_name='random', skip_save=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'],
                                                min_lr=0.0000000000001, patience=30)

        train_loss, val_loss = train_model(model=model,
                                        loss_func='mse',
                                        optimizer=optimizer,
                                        train_data_loader=train_set,
                                        val_data_loader=val_set,
                                        batch_size=32,
                                        epochs=500,
                                        early_stopper=early_Stopper,
                                        scheduler=scheduler,
                                        device=device)

        best_loss = early_Stopper.best_score

        train.report({"mse_loss": best_loss})  # This sends the score to Tune.





if __name__ == '__main__':

    set_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = 1
    else:
        device = torch.device("cpu")
        gpu = 0


    model_name = 'AFP'
    dataset = 'Melting Point'


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
        CS.UniformFloatHyperparameter("dropout", lower=0., upper=1.4))
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter("mlp_layers", lower=1, upper=4))
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter("afp_mol_layers", lower=1, upper=4))

    print(config_space)


    ################################# -------------- ######################################


    ### Define search algorithm
    algo = TuneBOHB(config_space, metric="mse_loss", mode="min")

    ### Get the trial control algorithm
    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="mse_loss",
        mode="min",
        max_t=100)

    ### initialize the trainable with the dataset from the top
    my_trainable = trainable

    trainable_with_resources = tune.with_resources(my_trainable, {"cpu":1, "gpu":gpu})
    ### Initialize the tuner
    tuner = Tuner(my_trainable, tune_config=tune.TuneConfig(scheduler=HyperBandForBOHB(),
                                            search_alg=algo,
                                            mode='min',
                                            metric="mse_loss"))

    result = tuner.fit()

    import json

    best_result = result.get_best_result(metric="mse_loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics

    results_to_save = {
        "best_config": best_config,
        "best_metrics": best_metrics
    }

    file_name = "./results/best_hyperparameters_" + model_name +"_"+dataset+".json"

    with open(file_name, "w") as file:
        json.dump(results_to_save, file, indent=4)