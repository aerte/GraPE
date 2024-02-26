import os.path as osp

import pandas as pd
from torch_geometric.data import download_url
from dglchem.utils.data import GraphDataSet

__all__ = [
    'BradleyDoublePlus'
]

class BradleyDoublePlus(GraphDataSet):
    """A dataset class inspired by the Torchvision datasets such as MNIST. It will download the *Jean-Claude Bradley
    Double Plus Good (Highly Curated and Validated) Melting Point* Dataset [1] should it not already exist, and then
    initializes it into a **GraphDataSet** class.

    Parameters:
    ----------
    root: str
        Indicates what the root or working directory is. Default: None
    target: str
        A string that indicates which of the features from the dataset should be the 'target'.
    global_features: list of str
        A list of strings indicating any additional features that should be included as global features.
    allowed_atoms: list of str
        List of allowed atom symbols. Default are the AFP atoms.
    atom_feature_list: list of str
        List of features to be applied. Default are the AFP atom features.
    bond_feature_list: list of str
        List of features that will be applied. Default are the AFP features
    split: bool
        An indicator if the dataset should be split. Only takes effect if nothing else regarding the split is specified
        and will trigger the default split. Default: False
    split_type: str
        Indicates what split should be used. Default: random. The options are:
        [consecutive, random, molecular weight, scaffold, stratified, custom]
    split_frac: array
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: array
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample.
    log: bool
        Decides if the filtering output and other outputs will be shown. Default: False
    save_data_filename: str
        The filename of the saved dataset. If given, the dataset will be automatically saved after processing.
        Default: None

    ----

    References

    [1] Jean-Claude Bradley and Andrew Lang and Antony Williams, Jean-Claude Bradley Double Plus Good (Highly Curated
    and Validated) Melting Point Dataset, 2014, http://dx.doi.org/10.6084/m9.figshare.1031637




    """


    def __init__(self, root = None, target = None, global_features = None, allowed_atoms = None,
                 atom_feature_list = None, bond_feature_list = None, split = False, split_type = None,
                 split_frac = None, custom_split = None, log = False, save_data_filename=None):

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


        if save_data_filename is not None:
            self.save_data_set(filename=save_data_filename)
            self.get_smiles()


