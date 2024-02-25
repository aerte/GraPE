import os.path as osp

import pandas as pd
from torch_geometric.data import download_url
from dglchem.utils.data import MakeGraphDataSet

__all__ = [
    'BradleyDoublePlus'
]

class BradleyDoublePlus(MakeGraphDataSet):


    def __init__(self, root = None, target = None, global_features = None,
                 allowed_atoms = None, atom_feature_list = None, bond_feature_list = None, split = False,
                 split_type = None, split_frac = None, custom_split = None, log = False):

        global file_path
        if root is None:
            self.root = './data'

        self.file_name = 'BradleyDoublePlus.xlsx'

        self.raw_path = self.raw_dir

        if not osp.exists(osp.join(self.raw_path, self.file_name)):
            download_url('https://figshare.com/ndownloader/files/1503991',
                         folder = self.raw_path,
                         filename= self.file_name,
                         log = True)

            path = osp.join(self.raw_path, self.file_name)

        else:
            path = osp.join(self.raw_path, self.file_name)

        df = pd.read_excel(path)

        if target is None:
            target = 'mpC'
        if global_features is not None:
            global_features = df[global_features]


        super().__init__(smiles = df.smiles, target = df[target], global_features=global_features,
                         allowed_atoms = allowed_atoms, atom_feature_list = atom_feature_list,
                         bond_feature_list = bond_feature_list, split=split, split_type=split_type,
                         split_frac=split_frac, custom_split=custom_split, log = log)


