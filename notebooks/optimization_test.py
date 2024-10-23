############################ Optimization script ############################################
from ray.air import RunConfig
import ray
import dgl
import os
import tempfile
from grape_chem.models import AFP, MPNN, DMPNN, MEGNet, GroupGAT_jittable
from grape_chem.utils import EarlyStopping, train_model
from functools import partial
import torch
from torch.optim import lr_scheduler
# from ray.tune.search.optuna import OptunaSearch
from ray.tune import Tuner
from ray import tune, train
import numpy as np
import pandas as pd
from grape_chem.utils import DataSet, split_data, train_epoch_jittable, val_epoch_jittable, RevIndexedSubSet
from torch_geometric.loader import DataLoader
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
import ConfigSpace as CS
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer

#root = '/zhome/4a/a/156124/GraPE/notebooks/data_splits.xlsx'
root = 'env/data_splits.xlsx'
ray.init(_temp_dir="/media/paul/external_drive/tmp_ray")

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
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
    return [2 ** i * 32 for i in range(num, 0, -1)]

def load_dataset_from_excel(dataset, is_dmpnn=False, is_megnet=False):
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

    df = pd.read_excel(root, sheet_name=dataset)

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

    return train_set, val_set, test_set

def load_pka_dataset(fragmentation=None):
    """
    Loads the pKa dataset with global features and applies fragmentation.
    """
    if fragmentation is None:
        from grape_chem.utils import JT_SubGraph
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fragmentation_scheme_file_path = os.path.join(script_dir, 'env', 'MG_plus_reference')
        save_fragmentation_file_path = os.path.join(script_dir, 'env', 'fragmentation_data')
        fragmentation = JT_SubGraph(scheme=fragmentation_scheme_file_path, save_file_path=save_fragmentation_file_path, verbose=False)
        frag_dim = fragmentation.frag_dim
    
    def standardize(x, mean, std):
        return (x - mean) / std

    # Path to your pKa dataset
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = script_dir+'/env/pka_dataset.xlsx'
    df = pd.read_excel(root)

    smiles = df['SMILES'].to_numpy()
    target = df['Target'].to_numpy()
    tags = df['Tag'].to_numpy()

    # Map tags to integers
    tag_to_int = {'Train': 0, 'Val': 1, 'Test': 2}
    custom_split = np.array([tag_to_int[tag] for tag in tags])

    # Load global features
    if 'Global Feats' in df.columns:
        global_feats = df['Global Feats'].to_numpy()
    else:
        # If no global features are provided, generate random ones (as in your workflow)
        global_feats = np.random.randn(len(smiles))

    # Standardize targets and global features
    mean_target, std_target = np.mean(target), np.std(target)
    target = standardize(target, mean_target, std_target)
    mean_global_feats, std_global_feats = np.mean(global_feats), np.std(global_feats)
    global_feats = standardize(global_feats, mean_global_feats, std_global_feats)

    # Initialize dataset with fragmentation and global features
    data = DataSet(
        smiles=smiles,
        target=target,
        global_features=global_feats,
        filter=False,
        fragmentation=fragmentation,
        custom_split=custom_split
    )

    # Split data
    train_set, val_set, test_set = split_data(
        data, split_type='custom', custom_split=custom_split
    )

    return train_set, val_set, test_set, mean_target, std_target

def load_model(model_name, config, device=None):
    """ Function to load a model based on a model name and a config dictionary. Is supposed to reduce clutter in the trainable function.

    model_name: str
        A string that defines the model to be loaded. Options:
        * "AFP"
        * "MPNN"
        * "DMPNN"
        * "MEGNet"
        * "GroupGAT"
    config : ConfigSpace
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    mlp_out = return_hidden_layers(int(config['MLP_layers']))

    if model_name == "AFP":
        return AFP(node_in_dim=44, edge_in_dim=12, num_layers_mol=int(config["afp_mol_layers"]),
                   num_layers_atom=int(config["depth"]), rep_dropout=config["dropout"],
                   hidden_dim=int(config["gnn_hidden_dim"]),
                   mlp_out_hidden=mlp_out)
    elif model_name == "MPNN":
        return MPNN(node_in_dim=44, edge_in_dim=12, num_layers=int(config["depth"]),
                    mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                    node_hidden_dim=int(config["gnn_hidden_dim"]))
    elif model_name == "DMPNN":
        return DMPNN(node_in_dim=44, edge_in_dim=12, node_hidden_dim=int(config["gnn_hidden_dim"]),
                     depth=int(config["depth"]), dropout=0, mlp_out_hidden=mlp_out,
                     rep_dropout=config["dropout"])
    elif model_name == "MEGNet":
        return MEGNet(node_in_dim=44, edge_in_dim=12, global_in_dim=1, node_hidden_dim=int(config["gnn_hidden_dim"]),
                      edge_hidden_dim=int(config["edge_hidden_dim"]), depth=int(config["depth"]),
                      mlp_out_hidden=mlp_out, rep_dropout=config["dropout"],
                      device=device)
    elif model_name == "GroupGAT":
        mlp_out = return_hidden_layers(int(config['MLP_layers']))
        net_params = {
            "device": device,
            "num_atom_type": 44,
            "num_bond_type": 12,
            "dropout": config["dropout"],
            "MLP_layers": int(config["MLP_layers"]),
            "frag_dim": 219, #TODO: don't hardcode 
            "final_dropout": config.get("final_dropout", 0.257507),
            "num_heads": 1,
            "node_in_dim": 44,
            "edge_in_dim": 12,
            "num_global_feats": 1,
            "hidden_dim": int(config["hidden_dim"]),
            "mlp_out_hidden": mlp_out,
            "num_layers_atom": int(config["num_layers_atom"]),
            "num_layers_mol": int(config["num_layers_mol"]),
            "L1_layers_atom": int(config["L1_layers_atom"]),
            "L1_layers_mol": int(config["L1_layers_mol"]),
            "L1_dropout": config["L1_dropout"],
            "L1_hidden_dim": int(config["L1_hidden_dim"]),

            "L2_layers_atom": int(config["L2_layers_atom"]),
            "L2_layers_mol": int(config["L2_layers_mol"]),
            "L2_dropout": config["L2_dropout"],
            "L2_hidden_dim": int(config["L2_hidden_dim"]),

            "L3_layers_atom": int(config["L3_layers_atom"]),
            "L3_layers_mol": int(config["L3_layers_mol"]),
            "L3_dropout": config["L3_dropout"],
            "L3_hidden_dim": int(config["L3_hidden_dim"]),    
        }
        return GroupGAT_jittable.GCGAT_v4pro_jit(net_params)

def trainable(config, data_name: str = None, model_name: str = None, is_dmpnn: bool = False, device: torch.device = None, is_megnet: bool = False, is_groupgat: bool = True, fragmentation=None): #set groupgat to also false by default
    """
    The trainable for Ray Tune.

    Parameters
    -----------
        config: dict
            A ConfigSpace dictionary adhering to the required parameters in the trainable. Defines the search space of the HO.
        data_name: str
            The graphs to be used.
        model_name: str
            The model to be loaded.
    """

    ################### Loading the graphs #########################################################################
    if data_name == 'free':
        train_set, val_set, _ = load_dataset_from_excel("FreeSolv", is_dmpnn=is_dmpnn, is_megnet=is_megnet)
    elif data_name == 'mp':
        train_set, val_set, _ = load_dataset_from_excel("Melting Point", is_dmpnn=is_dmpnn, is_megnet=is_megnet)
    elif data_name == 'qm':
        train_set, val_set, _ = load_dataset_from_excel("Heat capacity", is_dmpnn=is_dmpnn, is_megnet=is_megnet)
    elif data_name == 'logp':
        train_set, val_set, _ = load_dataset_from_excel("LogP", is_dmpnn=is_dmpnn, is_megnet=is_megnet)
    else:
        train_set, val_set, _, _, _ = load_pka_dataset(fragmentation=fragmentation)
    print("loaded dataset")
    ################### Defining the model #########################################################################

    model = load_model(model_name=model_name, config=config, device=device)

    model.to(device=device)
    # model = train.torch.prepare_model(model)

    ################################################################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    early_Stopper = EarlyStopping(patience=30, model_name='random', skip_save=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_reduction_factor'],
                                               min_lr=1e-10, patience=10)
    loss_function = torch.nn.functional.mse_loss

    train_data = DataLoader(train_set, batch_size=300)
    val_data = DataLoader(val_set, batch_size=300)

    iterations = 300 #wtf? why is this hardcoded? TODO: investigate
    start_epoch = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state_dict = torch.load(
                os.path.join(checkpoint_dir, "model.pt"),
                map_location=device,
            )
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
            start_epoch = torch.load(os.path.join(checkpoint_dir, "extra_state.pt"))["epoch"] + 1

    model.train()
    for i in range(start_epoch, iterations):
        # Add logging to track progress
        #print(f"Trial {train.get_trial_id()}: Starting epoch {i + 1}/{iterations}")

        train_loss = train_epoch_jittable(model=model, loss_func=loss_function, optimizer=optimizer, train_loader=train_data,
                                 device=device)
        val_loss = val_epoch_jittable(model=model, loss_func=loss_function, val_loader=val_data, device=device)
        #print(f"Trial {train.trial_id}: Epoch {i + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        early_Stopper(val_loss=val_loss, model=model)

        # Report metrics to Ray Tune
        metrics = {
            "epoch": i + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mae_loss": val_loss  #TODO: check if correct
        }
        train.report(metrics)

        # Checkpointing
        should_checkpoint = i % config.get("checkpoint_freq", 15) == 0
        if should_checkpoint and train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                # Save model and optimizer state
                torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pt"))
                torch.save({"epoch": i}, os.path.join(temp_checkpoint_dir, "extra_state.pt"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Report checkpoint to Ray Tune
            train.report(metrics, checkpoint=checkpoint)
        # We report the loss to ray tune every 15 steps, that way tune's scheduler can interfere


if __name__ == '__main__':

    import argparse
    #temporarily disabling the args to chose a model, as  we're testing a GroupGAT workflow here
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default='pka', choices=['mp', 'logp', 'qm', 'free', 'pka'],
                        help='the data that will be trained on (default: %(default)s)')
    parser.add_argument('--samples', type=int, default=100,
                        help='the number of samples/instances that will be running (default: %(default)s)')
    # parser.add_argument('--model', type=str, default='afp', choices=['afp', 'mpnn', 'dmpnn', 'megnet', 'groupgat'],
    #                     help='the model to be used (default: %(default)s)')

    is_groupgat = True
    model_name = "GroupGAT"
    
    args = parser.parse_args()
    data_name = args.data
    n_samples = args.samples

    #model_ = args.model
    is_dmpnn = False
    is_megnet = False
    is_groupgat = True

    ################################# Selecting the options ######################################

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu = 1
    else:
        device = torch.device("cpu")
        gpu = 0


    # if model_name == "GroupGAT":
    #     from grape_chem.utils import JT_SubGraph
    #     #script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #     #fragmentation_scheme_file_path = os.path.join(script_dir, 'env', 'MG_plus_reference')
    #     fragmentation_scheme_file_path = 'MG_plus_reference'
    #     fragmentation = JT_SubGraph(scheme=fragmentation_scheme_file_path)
    # else:
    #     fragmentation = None

    if data_name == 'pka':
        dataset = 'pKa'
    elif data_name == 'free':
        dataset = 'FreeSolv'
    elif data_name == 'mp':
        dataset = 'Melting_Point'
    elif data_name == 'qm':
        dataset = 'QM9'
    else:
        dataset = 'LogP'

    model_ = "GroupGAT" #TODO: remove
    if model_ == 'mpnn':
        model_name = "MPNN"
    elif model_ == "dmpnn":
        model_name = "DMPNN"
        is_dmpnn = True
    elif model_ == "megnet":
        model_name = "MEGNet"
        is_megnet = True
    elif model_ == "afp":
        model_name = "AFP"
    else:
        model_name = "GroupGAT"

    ################################# Search space ######################################
    print("model name: ", model_name)
    config_space = CS.ConfigurationSpace()
    config_space.add(
        CS.UniformIntegerHyperparameter("depth", lower=1, upper=5))
    config_space.add(
        CS.UniformFloatHyperparameter('initial_lr', lower=1e-5, upper=1e-1))
    config_space.add(
        CS.UniformFloatHyperparameter("weight_decay", lower=1e-6, upper=1e-1))
    config_space.add(
        CS.UniformFloatHyperparameter("lr_reduction_factor", lower=0.4, upper=0.99))
    config_space.add(
        CS.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.15))
    config_space.add(
        CS.UniformIntegerHyperparameter("mlp_layers", lower=1, upper=4))
    # If AFP is selected
    if model_name == "AFP":
        config_space.add(
            CS.UniformIntegerHyperparameter("afp_mol_layers", lower=1, upper=4))
    # If MEGNet is selected
    elif model_name == "MEGNet":
        config_space.add(
            CS.UniformIntegerHyperparameter("edge_hidden_dim", lower=32, upper=256))
    elif model_name == "GroupGAT":
        config_space.add(CS.UniformIntegerHyperparameter("hidden_dim", lower=64, upper=512))

        config_space.add(CS.UniformIntegerHyperparameter("num_layers_atom", lower=1, upper=5))
        config_space.add(CS.UniformIntegerHyperparameter("num_layers_mol", lower=1, upper=5))
        config_space.add(CS.UniformFloatHyperparameter("final_dropout", lower=0.0, upper=0.5))
        config_space.add(CS.UniformIntegerHyperparameter("num_heads", lower=1, upper=4))
        config_space.add(CS.UniformIntegerHyperparameter("MLP_layers", lower=1, upper=3))
        # L1:
        config_space.add(CS.UniformIntegerHyperparameter("L1_layers_atom", lower=1, upper=5)) #layers
        config_space.add(CS.UniformIntegerHyperparameter("L1_layers_mol", lower=1, upper=5))  #depth
        config_space.add(CS.UniformFloatHyperparameter("L1_dropout", lower=0.0, upper=0.5))
        config_space.add(CS.UniformIntegerHyperparameter("L1_hidden_dim", lower=32, upper=256))

        config_space.add(CS.UniformIntegerHyperparameter("L2_layers_atom", lower=1, upper=5)) #layers
        config_space.add(CS.UniformIntegerHyperparameter("L2_layers_mol", lower=1, upper=5))  #depth
        config_space.add(CS.UniformFloatHyperparameter("L2_dropout", lower=0.0, upper=0.5))
        config_space.add(CS.UniformIntegerHyperparameter("L2_hidden_dim", lower=32, upper=256))

        config_space.add(CS.UniformIntegerHyperparameter("L3_layers_atom", lower=1, upper=5)) #layers
        config_space.add(CS.UniformIntegerHyperparameter("L3_layers_mol", lower=1, upper=5))  #depth
        config_space.add(CS.UniformFloatHyperparameter("L3_dropout", lower=0.0, upper=0.5))
        config_space.add(CS.UniformIntegerHyperparameter("L3_hidden_dim", lower=32, upper=256))
        
    # search_space = {
    #     "depth": tune.randint(1,5),
    #     "gnn_hidden_dim": tune.randint(32, 256),
    #     "lr_reduction_factor": tune.uniform(0.4, 0.99),
    #     "dropout": tune.uniform(0.1, 0.4),
    #     "afp_mol_layers": tune.randint(1, 4),
    #     "initial_lr": tune.uniform(1e-5, 1e-1),
    #     "weight_decay": tune.uniform(1e-6, 1e-1),
    #     "mlp_layers": tune.randint(1, 4)
    # }

    # if model_name == "AFP":
    #     search_space['afp_mol_layers'] = tune.randint(1, 4)
    # elif model_name == "MEGNet":
    #     search_space['edge_hidden_dim'] = tune.randint(32, 256)

    ################################# --------------------- ######################################


    my_trainable = partial(trainable, data_name=data_name, model_name=model_name, is_dmpnn=is_dmpnn,
                           is_megnet=is_megnet, device=device, is_groupgat=is_groupgat, ) #fragmentation=fragmentation

    trainable_with_resources = tune.with_resources(my_trainable, {"cpu": 4, "gpu": gpu})

    ### Define search algorithm
    algo = TuneBOHB(config_space, mode='min', metric="mae_loss", )

    ## Get the trial control algorithm
    scheduler = HyperBandForBOHB(
        time_attr="epoch",   # Changed from 'training_iteration' to 'epoch'
        max_t=1000,    # Set to match the iterations in the training loop
        reduction_factor=2.9,
    )

    from ray.tune import CLIReporter

    # Define the metrics and parameters to report
    reporter = CLIReporter(
        parameter_columns=["depth", "gnn_hidden_dim", "initial_lr", "dropout"],
        metric_columns=["epoch", "train_loss", "val_loss", "mae_loss"]
    )

    ## Initialize the tuner
    tuner = Tuner(trainable_with_resources,
                  tune_config=tune.TuneConfig(
                      scheduler=scheduler,
                      search_alg=algo,
                      mode='min',
                      metric="mae_loss",
                      num_samples=n_samples),
                  run_config=train.RunConfig(
                      name="bo_exp",
                      stop={"epoch": 300},
                      progress_reporter=reporter
                  #   param_space=search_space
                      ),

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

    #file_name = "/zhome/4a/a/156124/GraPE/notebooks/results/new_best_hyperparameters_" + model_name + "_" + dataset + ".json"

    directory = os.path.join("env", "bohb_results")
    file_name = os.path.join(directory, f"new_best_hyperparameters_{model_name}_{dataset}.json")
    os.makedirs(directory, exist_ok=True)
    with open(file_name, "w") as file:
        json.dump(results_to_save, file, indent=4)