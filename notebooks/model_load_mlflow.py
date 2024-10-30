import torch
import pandas as pd
import numpy as np
from grape_chem.models import DMPNN, AFP
from grape_chem.utils import DataSet
from grape_chem.utils.model_utils import load_model
from grape_chem.logging.mlflow_logging import setup_mlflow

import mlflow
import mlflow.pytorch

import json
import os
from typing import Dict
from itertools import product

## Start server with: mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_inference(config: Dict):
    with setup_mlflow(config):
        mlflow.log_params(config)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pretrained model
        model_path = config['model_path']
        model_class = config['model_class']
        model, dataset = load_model(model_class, model_path, device)
        if model == -1 and dataset == -1:
            # continue to next configuration
            return
        
        print("Dataset = ", dataset)
        # Log model details
        mlflow.log_param("model_class", model_class)
        mlflow.log_param("model_path", model_path)
        
        # Prepare input data for inference
        input_smiles = config['input_smiles']
        mlflow.log_param("input_smiles", input_smiles)
        
        # Perform inference
        preds = dataset.predict_smiles(input_smiles, model)
        
        print("Predictions: ", preds)

        # Save and log predictions
        predictions_file = f"predictions_{config['run_name']}.json"
        with open(predictions_file, 'w') as f:
            json.dump(preds, f)
        
        mlflow.log_artifact(predictions_file)
        print(f"Predictions saved to {predictions_file}")
        # Removing file after logging with mlFLow
        os.remove(predictions_file)

def main():
    
    # Define configurations
    configs = {
        'experiment_name': ["Ray Molecular Property Prediction Inference"],
        'run_name': ["AFP Model Inference"],
        'model_path': [os.path.join(current_dir, '../models', 'AFP', 'AFP_29-09-2024-10_final.pt')],
        'model_class': ['AFP'],
        'input_smiles': [
            ['CC', 'CCc1ccccn1', 'C=C(C)C#C', 'CC(C)CCCC']
        ]
    }
    
    # Generate all possible combinations of configurations
    keys, values = zip(*configs.items())
    configurations = [dict(zip(keys, v)) for v in product(*values)]
    
    # Run inference for each configuration
    for i, config in enumerate(configurations):
        print(f"Running inference {i+1}/{len(configurations)}")
        config['run_name'] = f"{config['run_name']} - Config {i+1}"
        run_inference(config)

if __name__ == "__main__":
    main()