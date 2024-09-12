import torch
from torch.optim import Adam
from torch.nn import MSELoss
from grape_chem.models import AFP, DMPNN
from torch_geometric.data import Data
from grape_chem.utils import train_model, test_model, pred_metric, EarlyStopping
from grape_chem.utils.model_utils import set_seed
from sklearn.utils import resample
import numpy as np

class Ensemble:
    def __init__(self, train_data, val_data, test_data, node_in_dim, edge_in_dim, device, hyperparams):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.device = device
        self.hyperparams = hyperparams

    def predict_ensemble(self, models):
        predictions = []
        metrics = []
        for model in models:
            model.to(self.device)
            pred = test_model(model=model, test_data_loader=self.test_data, device=self.device, batch_size=len(self.test_data))
            predictions.append(pred)
            test_targets = [data.y for data in self.test_data]
            targets = torch.cat(test_targets, dim=0).to(self.device)
            metric_results = pred_metric(prediction=pred, target=targets, metrics='all', print_out=False)
            metrics.append(metric_results)
        return metrics

    def create_model(self):
        if self.hyperparams['model'] == 'afp':
            return AFP(node_in_dim=self.node_in_dim, edge_in_dim=self.edge_in_dim).to(self.device)
        elif self.hyperparams['model'] == 'dmpnn':
            return DMPNN(node_in_dim=self.node_in_dim, edge_in_dim=self.edge_in_dim).to(self.device)
        else:
            raise ValueError('Invalid model type')

    def train_single_model(self, train_data, val_data, model_index):
        print(f'Training model {model_index + 1}/{self.hyperparams["n_models"]}')
        model = self.create_model()
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=self.hyperparams['learning_rate'])
        early_stopping = EarlyStopping(patience=self.hyperparams['early_stopping_patience'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.hyperparams['scheduler_factor'], patience=self.hyperparams['scheduler_patience']
        )

        train_model(model=model, loss_func=criterion, optimizer=optimizer, scheduler=scheduler,
                    train_data_loader=train_data, val_data_loader=val_data, epochs=self.hyperparams['epochs'], device=self.device,
                    batch_size=self.hyperparams['batch_size'], early_stopper=early_stopping)
        model.load_state_dict(torch.load('best_model.pt'))
        return model

    def run(self):
        train_samples, val_samples = self.create_samples()
        models = []
        for i in range(self.hyperparams['n_models']):
            model = self.train_single_model(train_samples[i], val_samples[i], i)
            models.append(model)
        metrics = self.predict_ensemble(models)
        return metrics

    def create_samples(self):
        raise NotImplementedError("Subclasses should implement this method")

class Bagging(Ensemble):
    def create_samples(self):
        def create_bootstrap_samples(data, n_samples):
            samples = []
            for _ in range(n_samples):
                indices = resample(range(len(data)), replace=True)
                sample_data = [data[i] for i in indices]
                samples.append(sample_data)
            return samples

        train_samples = create_bootstrap_samples(self.train_data, self.hyperparams['n_models'])
        val_samples = create_bootstrap_samples(self.val_data, self.hyperparams['n_models'])
        return train_samples, val_samples

class RandomWeightInitialization(Ensemble):
    def create_samples(self):
        train_samples = [self.train_data] * self.hyperparams['n_models']
        val_samples = [self.val_data] * self.hyperparams['n_models']
        return train_samples, val_samples

    def train_single_model(self, train_data, val_data, model_index):
        set_seed(self.hyperparams['weight_seed'] + model_index)
        return super().train_single_model(train_data, val_data, model_index)

class Jackknife(Ensemble):
    def create_samples(self):
        def create_synth_samples(reference_model, data, n_samples):
            synth_samples = []
            reference_model = reference_model.to(self.device)
            with torch.no_grad():
                predictions = test_model(reference_model, test_data_loader=data, device=self.device, batch_size=len(data))
            actual_values = [data[i].y.to(self.device) for i in range(len(data))]
            residuals = [actual_values[i] - predictions[i] for i in range(len(data))]

            for _ in range(n_samples):
                sampled_residuals = torch.stack(resample(residuals, replace=True))
                synthetic_values = predictions + sampled_residuals[0]
                synthetic_dataset = [Data(x=data[i].x, edge_index=data[i].edge_index, edge_attr=data[i].edge_attr,
                                          y=synthetic_values[i], revedge_index=getattr(data[i], 'revedge_index', None)) for i in range(len(data))]
                synth_samples.append(synthetic_dataset)
            return synth_samples

        reference_model = self.create_model()
        criterion = MSELoss()
        optimizer = Adam(reference_model.parameters(), lr=self.hyperparams['learning_rate'])
        early_stopping = EarlyStopping(patience=self.hyperparams['early_stopping_patience'])
        train_model(model=reference_model, loss_func=criterion, optimizer=optimizer, train_data_loader=self.train_data,
                    val_data_loader=self.val_data, epochs=self.hyperparams['epochs'], device=self.device, batch_size=self.hyperparams['batch_size'], early_stopper=early_stopping)
        
        train_samples = create_synth_samples(reference_model, self.train_data, self.hyperparams['n_models'])
        val_samples = [self.val_data] * self.hyperparams['n_models']
        return train_samples, val_samples

class BayesianBootstrap(Ensemble):
    def create_samples(self):
        def Dirichlet_sample(m, n):
            Dirichlet_params = np.ones(m * n)
            Dirichlet_weights = np.asarray([np.random.gamma(a, 1) for a in Dirichlet_params])
            Dirichlet_weights = Dirichlet_weights.reshape(m, n)
            row_sums = Dirichlet_weights.sum(axis=1)
            Dirichlet_weights /= row_sums[:, np.newaxis]
            return Dirichlet_weights

        def create_bayesian_samples(data, n_samples):
            weights_matrix = Dirichlet_sample(n_samples, len(data))
            samples = []
            
            for weights in weights_matrix:
                indices = np.random.choice(len(data), len(data), replace=True, p=weights)
                sampled_data = [data[i] for i in indices]
                samples.append(sampled_data)
            
            return samples

        train_samples = create_bayesian_samples(self.train_data, self.hyperparams['n_models'])
        val_samples = create_bayesian_samples(self.val_data, self.hyperparams['n_models'])
        return train_samples, val_samples