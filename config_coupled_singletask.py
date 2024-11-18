# config.yaml

file_name: '/home/paul/Documents/spaghetti/GraPE/env/ICP.xlsx'
file_type: 'excel'
smiles_column: 'SMILES'
target_column: 'Value'
global_features_column: 'T'
split_column: 'Subset'
split_mapping:
  Training: 0
  Validation: 1
  Test: 2

fragmentation_scheme: 'MG_plus_reference'
fragmentation_save_file_path: '/home/paul/Documents/spaghetti/GraPE/env/ICP_fragmentation.pth'
fragmentation_verbose: False
