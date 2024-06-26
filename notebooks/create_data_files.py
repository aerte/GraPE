from grape_chem.datasets import *
from grape_chem.utils import set_seed
import pandas as pd
import numpy as np

set_seed(69)

with pd.ExcelWriter('data_splits_global_feats.xlsx', engine='openpyxl') as writer:
    ##### Bradley double plus #####
    # mp = BradleyDoublePlus(split_type='random', split_frac=[0.8, 0.1, 0.1])
    # train, test, val = mp.train, mp.test, mp.val
    # df_train = pd.DataFrame({'SMILES': train.smiles, 'Split': 'train', 'Target': train.y})
    # df_val = pd.DataFrame({'SMILES': val.smiles, 'Split': 'val', 'Target': val.y})
    # df_test = pd.DataFrame({'SMILES': test.smiles, 'Split': 'test', 'Target': test.y})
    # df = pd.concat([df_train, df_val, df_test])
    # df.to_excel(writer, sheet_name='Melting Point', index=False)
    #
    # ##### Log-P #####
    # mp = LogP(split_type='random', split_frac=[0.8, 0.1, 0.1])
    # train, test, val = mp.train, mp.test, mp.val
    # df_train = pd.DataFrame({'SMILES': train.smiles, 'Split': 'train', 'Target': train.y})
    # df_val = pd.DataFrame({'SMILES': val.smiles, 'Split': 'val', 'Target': val.y})
    # df_test = pd.DataFrame({'SMILES': test.smiles, 'Split': 'test', 'Target': test.y})
    # df = pd.concat([df_train, df_val, df_test])
    # df.to_excel(writer, sheet_name='LogP', index=False)

    ##### QM9 #####
    mp = QM9(split_type='random', split_frac=[0.8, 0.1, 0.1], global_feature_id=0)
    train, test, val = mp.train, mp.test, mp.val
    df_train = pd.DataFrame({'SMILES': train.smiles, 'Split': 'train', 'Target': train.y, 'global':train.global_features})
    df_val = pd.DataFrame({'SMILES': val.smiles, 'Split': 'val', 'Target': val.y, 'global':val.global_features})
    df_test = pd.DataFrame({'SMILES': test.smiles, 'Split': 'test', 'Target': test.y, 'global':test.global_features})
    df = pd.concat([df_train, df_val, df_test])
    df.to_excel(writer, sheet_name='qm', index=False)

    # ##### FreeSolv #####
    # mp = FreeSolv(split_type='random', split_frac=[0.8, 0.1, 0.1], scale=False)
    # train, test, val = mp.train, mp.test, mp.val
    # df_train = pd.DataFrame({'SMILES': train.smiles, 'Split': 'train', 'Target': train.y})
    # df_val = pd.DataFrame({'SMILES': val.smiles, 'Split': 'val', 'Target': val.y})
    # df_test = pd.DataFrame({'SMILES': test.smiles, 'Split': 'test', 'Target': test.y})
    # df = pd.concat([df_train, df_val, df_test])
    # df.to_excel(writer, sheet_name='FreeSolv', index=False)