from grape_chem.utils.data import GraphDataSet
from torch_geometric.datasets import QM9 as QM9pyg

__all__ = [
    'QM9'
]

class QM9(GraphDataSet):
    """A dataset implementation for the QM9 [1]. This is an **alternative to the pytorch-geometric** implementation
    of the same name. The reason for this implementation is to allow for easy access to the featurization
    step, i.e., allow for personalized featurization of the SMILES during pre-preprocessing. The dataset
    encompasses about 130,000 molecules with 19 regression targets, for more information please see
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.QM9.html#torch_geometric.datasets.QM9.



    Notes
    ------
    This is a large dataset, and in the current implementation, loading the graphs from scratch can
    take up to 2 minutes depending on the CPU.



     ----

    References

    [1] Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Leswing, K.,
    & Pande, V. S. (2017). Moleculenet: A benchmark for molecular machine learning.
    CoRR, abs/1703.00564. http://arxiv.org/abs/1703.00564

    ----

    Parameters
    ------------
    target_id: int
        An integer that indicates, which of the features from the dataset should be the 'target'. The
        default is ``5``:``heat capacity``
    global_feature_id: int
        An integer that indicates, which of the features from the dataset should be a global feature. For
        example, one could choose ``0``:``Dipole moment``. Default: None
    root: str
        Indicates what the root or working directory is. Default: None
    global_features: list of str
        A list of strings indicating any additional features that should be included as global features.
    allowed_atoms: list of str
        List of allowed atom symbols. Default is the AFP atoms.
    only_organic: bool
        Checks if a molecule is ``organic`` counting the number of ``C`` atoms. If set to True, then molecules with less
        than one carbon will be discarded. Default: True
    atom_feature_list: list of str
        List of features to be applied. Default is the AFP atom features.
    bond_feature_list: list of str
        List of features that will be applied. Default is the AFP features
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


    def __init__(self, target_id: int = 5, global_feature_id: int = None,
                 root: str = None, global_features: list or str = None,
                 allowed_atoms: list[str] = None, only_organic: bool = True,
                 atom_feature_list: list[str] = None, bond_feature_list: list[str] = None,
                 split: bool = False, split_type: str = None, split_frac: list[float] = None,
                 custom_split: list[int] = None, log: bool = False, save_data_filename: str =None, seed:int=None):


        self.root = './graphs' if root is None else root

        self.raw_path = self.raw_dir

        data = QM9pyg(root = self.root)

        SMILES = data.smiles
        TARGET = data.y[:, target_id]

        if global_feature_id is not None:
            global_features = data.y[:, global_feature_id]

        super().__init__(smiles = SMILES, target = TARGET, global_features=global_features,
                         allowed_atoms = allowed_atoms, only_organic=only_organic,
                         atom_feature_list = atom_feature_list,
                         bond_feature_list = bond_feature_list, split=split, split_type=split_type,
                         split_frac=split_frac, custom_split=custom_split, log = log, seed=seed)

        self.data_name = 'QM9'


        if save_data_filename is not None:
            self.save_data_set(filename=save_data_filename)
            self.get_smiles()