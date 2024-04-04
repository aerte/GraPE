# Module for datasets plots
from collections import defaultdict
import numpy as np
import dgl
import torch

from rdkit.Chem import MolFromSmiles, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs

from torch_geometric.data import Data
from dgllife.utils import splitters

__all__ = [
    'SubSet',
    'torch_subset_to_SubSet',
    'split_data'
]

import grape.utils


class SubSet(object):
    """An adaptation of the Pytorch Subset (https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset) to fit
    into the GraPE workflow. The advantage of using SubSet is that it gives direct access to the subsets smiles and
    targets.

    To use this function, the dataset **must be instantiated using DataSet or GraphDataSet**.

    Parameters
    -----------
    dataset
        The full dataset, where dataset[i] should return the ith datapoint.
    indices: list
        Full list of indices of the subset to be constructed.

    """

    def __init__(self, dataset, indices: list):
        self.dataset = dataset
        self.indices = indices
        self.y = dataset.target[indices]
        self.smiles = dataset.smiles[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if isinstance(item, list):
            return self.dataset[[self.indices[i] for i in item]]
        return self.dataset[self.indices[item]]

def torch_subset_to_SubSet(subset: dgl.data.Subset or torch.utils.data.Subset):
    """Returns the GraPE SubSet from the DGL or PyTorch Subset.

    Parameters
    ----------
    subset: dgl.data.Subset or torch.utils.data.Subset

    Returns
    -------
    SubSet
    """
    assert isinstance(subset.dataset, grape.utils.DataSet) or isinstance(subset.dataset, grape.utils.GraphDataSet),(
        'The subsets underlying dataset has to be either DataSet or GraphDataSet.')
    return SubSet(subset.dataset, subset.indices)



def split_data(data, split_type: str = None, split_frac: float = None, custom_split: list = None,
               labels: np.array = None, task_id: int = None, bucket_size: int = 10) -> tuple:
    """

    Parameters
    ------------
    data: Any iterable
        An object that can be accessed per an index and iterated upon. Ex: a DataSet or array object
    split_type: str
        Indicates what split should be used. Default: random. The options are: ['consecutive', 'random',
        'molecular weight', 'scaffold', 'stratified', 'custom']
    split_frac: list
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: list
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample. Default: None
    labels: array
        An array of shape (N,T) where N is the number of datasets points and T is the number of tasks. Used for the
        Stratified Splitter.
    task_id: int
        The task that will be used for the Stratified Splitter.
    bucket_size: int
        Size of the bucket that is used in the Stratified Splitter. Default: 10


    Returns
    ---------
    (train, val, test)
        - Lists containing the respective datasets objects.

    """

    if split_type is None:
        split_type = 'random'

    if split_frac is None:
        split_frac = [0.8,0.1,0.1]



    split_func = {
        'consecutive': splitters.ConsecutiveSplitter,
        'random': splitters.RandomSplitter,
        'molecular_weight': splitters.MolecularWeightSplitter,
        'scaffold': splitters.ScaffoldSplitter,
        'stratified': splitters.SingleTaskStratifiedSplitter
    }

    if split_type == 'custom' or custom_split is not None:
        assert custom_split is not None and len(custom_split) == len(data), (
            'The custom split has to match the length of the filtered dataset.'
            'Consider saving the filtered output with .get_smiles()')

        return data[custom_split == 0], data[custom_split == 1], data[custom_split == 2]

    match split_type:
        case 'consecutive':
            return split_func[split_type].train_val_test_split(data,frac_train=split_frac[0],frac_test=split_frac[1],
                                                                frac_val=split_frac[2])
        case 'random':
            return split_func[split_type].train_val_test_split(data, frac_train=split_frac[0], frac_test=split_frac[1],
                                                                frac_val=split_frac[2])
        case 'molecular_weight':
            return split_func[split_type].train_val_test_split(data, frac_train=split_frac[0], frac_test=split_frac[1],
                                                               frac_val=split_frac[2], log_every_n=1000)
        case 'scaffold':
            return split_func[split_type].train_val_test_split(data, frac_train=split_frac[0], frac_test=split_frac[1],
                                                    frac_val=split_frac[2], log_every_n=1000, scaffold_func='decompose')
        case 'stratified':
            return split_func[split_type].train_val_test_split(data, labels, task_id, frac_train=split_frac[0],
                                                frac_test=split_frac[1],frac_val=split_frac[2], bucket_size=bucket_size)


