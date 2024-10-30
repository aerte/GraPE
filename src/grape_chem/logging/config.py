import yaml
import os
import argparse

from grape_chem.utils.data import get_path

def load_config(config_file_path):
    '''
    Load the config file and update the paths to the data, save, and pip_requirements.
    
    Args:
        config_file_path (str): Full path to the config file.
        
    Returns:
        dict: Updated configuration dictionary.
    '''
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the paths based on the directory of the config file
    if 'data_files' in config:
        for file in config['data_files']:
            index = config['data_files'].index(file)
            file = get_path(os.path.dirname(config_file_path), file)
            config['data_files'][index] = file
    if 'data_path' in config:
        config['data_path'] = get_path(os.path.dirname(config_file_path), config['data_path'])

    config['save_path'] = get_path(os.path.dirname(config_file_path), config['save_path'])
    config['pip_requirements'] = get_path(os.path.dirname(config_file_path), config['pip_requirements'])

    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file")
    parser.add_argument('--epochs', type=int, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, help="Batch size")
    parser.add_argument('--learning_rate', type=float, help="Learning rate")
    args = parser.parse_args()

    # Load config file
    config = load_config(args.config)

    # Override config with CLI arguments if provided
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate

    return config