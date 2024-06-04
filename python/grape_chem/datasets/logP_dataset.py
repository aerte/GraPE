import os.path as osp

import pandas as pd
from torch_geometric.data import download_url
from grape_chem.utils.data import GraphDataSet

__all__ = [
    'LogP'
]

class LogP(GraphDataSet):
    """A dataset class inspired by the Torchvision datasets such as MNIST. It will download the corrected *logP* Dataset
    from https://github.com/nadinulrich/log_P_prediction/blob/30f2f6ad0d7806a3246a5b3da936aa02478d5202/Dataset_and_Predictions.xlsx
    [1], first introduced in [2], should it not already exist. It then initializes it into a **GraphDataSet** class.

     ----

    References

    [1] Ulrich, N., Goss, KU. & Ebert, A., Exploring the octanol–water partition coefficient dataset using deep learning techniques and datasets augmentation., Commun Chem 4, 90 (2021), http://dx.doi.org/10.1038/s42004-021-00528-9

    [2] Mansouri K, Grulke CM, Richard AM, Judson RS, Williams AJ., An automated curation procedure for addressing chemical errors and inconsistencies in public datasets used in QSAR modelling., SAR QSAR Environ Res. (2016), http://dx.doi.org/10.1080/1062936X.2016.1253611

    ----

    Parameters
    ------------
    root: str
        Indicates what the root or working directory is. Default: None
    global_features: list of str
        A list of strings indicating any additional features that should be included as global features.
    allowed_atoms: list of str
        List of allowed atom symbols. Default are the AFP atoms.
    only_organic: bool
        Checks if a molecule is ``organic`` counting the number of ``C`` atoms. If set to True, then molecules with less
        than one carbon will be discarded. Default: True
    atom_feature_list: list of str
        List of features to be applied. Default are the AFP atom features.
    bond_feature_list: list of str
        List of features that will be applied. Default are the AFP features
    split: bool
        An indicator if the dataset should be split. Only takes effect if nothing else regarding the split is specified
        and will trigger the default split. Default: False (recommended)
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
    seed: int
        The numpy seed used to generate the splits. Default: None

    """


    def __init__(self, root: str = None, global_features: list or str = None,
                 allowed_atoms: list[str] = None, only_organic: bool = True,
                 atom_feature_list: list[str] = None, bond_feature_list: list[str] = None,
                 split: bool = False, split_type: str = None, split_frac: list[float] = None,
                 custom_split: list[int] = None, log: bool = False, save_data_filename: str =None, seed: int = None):


        self.root = './graphs' if root is None else root

        file_name = 'LogP'

        self.raw_path = self.raw_dir

        if not osp.exists(osp.join(self.raw_path, file_name)):
            download_url(
                'https://github.com/nadinulrich/log_P_prediction/blob/30f2f6ad0d7806a3246a5b3da936aa02478d5202/Dataset_and_Predictions.xlsx?raw=true',
                 folder = self.raw_path,
                 filename= file_name,
                 log = True
            )

            path = osp.join(self.raw_path, file_name)

        else:
            path = osp.join(self.raw_path, file_name)

        df = pd.read_excel(path)
        labels = df.columns[3]
        self.target_name = 'logP'

        super().__init__(smiles = df.SMILES, target = df[labels], global_features=global_features,
                         allowed_atoms = allowed_atoms, only_organic=only_organic,
                         atom_feature_list = atom_feature_list,
                         bond_feature_list = bond_feature_list, split=split, split_type=split_type,
                         split_frac=split_frac, custom_split=custom_split, log = log, seed=seed)

        self.data_name = 'LogP'


        if save_data_filename is not None:
            self.save_data_set(filename=save_data_filename)
            self.get_smiles()
