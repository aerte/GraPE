# Module for splitting utilities

from typing import Generator, Union

import dgl
import torch

from dgllife.utils import splitters

__all__ = [
    'SubSet',
    'torch_subset_to_SubSet',
    'split_data'
]

import grape


##########################################################################
########### Data subsets and utils #######################################
##########################################################################

class SubSet(object):
    """An adaptation of the Pytorch Subset (https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset) to fit
    into the GraPE workflow. The advantage of using SubSet is that it gives direct access to the subsets smiles and
    targets.

    To use this function, the dataset **must be instantiated using DataSet or GraphDataSet**.

    Parameters
    -----------
    dataset: DataSet
        The full dataset, where dataset[i] should return the ith datapoint.
    indices: list of int
        Full list of indices of the subset to be constructed.

    """

    def __init__(self, dataset: grape.utils.DataSet, indices: list[int]):
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

def torch_subset_to_SubSet(subset: Union[dgl.data.Subset,torch.utils.data.Subset]) -> SubSet:
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

def mult_subset_to_gen(subsets: Union[tuple[dgl.data.Subset], tuple[torch.data.Subset]]) -> Generator:
    """Returns a Generator object corresponding to the length of the input with all the transformed SubSets.
    
    Parameters
    ----------
    subsets: tuple[dgl.data.Subset] or tuple[torch.utils.data.Subset]

    Returns
    -------
    Generator

    """
    for item in subsets:
        yield torch_subset_to_SubSet(item)

def unpack_gen(generator):
    return tuple(i for i in generator)

def mult_subset_to_SubSets(subsets: Union[tuple[dgl.data.Subset],tuple[torch.utils.data.Subset]]) -> tuple[SubSet]:
    """Returns a Generator object corresponding to the length of the input with all the transformed SubSets.

    Parameters
    ----------
    subsets: tuple[dgl.data.Subset] or tuple[torch.utils.data.Subset]

    Returns
    -------
    tuple[SubSets]

    """

    return unpack_gen(mult_subset_to_gen(subsets))


##########################################################################
########### Splitting utils ##############################################
##########################################################################


def split_data(data, split_type: str = None, split_frac: list[float] = None, custom_split: list = None, **kwargs) -> (
        tuple[SubSet, SubSet, SubSet]):
    """

    Parameters
    ------------
    data: Any iterable
        An object that can be accessed per an index and iterated upon. Ex: a DataSet or array object
    split_type: str
        Indicates what split should be used. Default: random. The options are: ['consecutive', 'random',
        'molecular weight', 'scaffold', 'stratified', 'custom']
    split_frac: list of float
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: list
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample. Default: None

    Returns
    ---------
    tuple[SubSet, SubSet, SubSet]
        The train, val and test splits respectively.

    Notes
    ------
    The stratified splitter from dgllife requires *two extra* inputs, specifically:

    labels: array
        An array of shape (N,T) where N is the number of datasets points and T is the number of tasks. Used for the
        Stratified Splitter.

    task_id: int
        The task that will be used for the Stratified Splitter.

    For all the other functions, please refer to their respective documentation to see what arguments can be used.

    See Also
    ---------
    https://lifesci.dgl.ai/api/utils.splitters.html
    # Here I should be inserting links to some documentation or my own documentation.
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
        'stratified': splitters.SingleTaskStratifiedSplitter,
        'butina': grape.splits.taylor_butina_clustering
    }

    if split_type == 'custom' or custom_split is not None:
        assert custom_split is not None and len(custom_split) == len(data), (
            'The custom split has to match the length of the filtered dataset.'
            'Consider saving the filtered output with .get_smiles()')

        return data[custom_split == 0], data[custom_split == 1], data[custom_split == 2]

    elif split_type == 'butina':
        return split_func[split_type](data, **kwargs)

    return mult_subset_to_SubSets(split_func[split_type](data).train_test_val_split(data, frac_train=split_frac[0],
                                                frac_test=split_frac[1],frac_val=split_frac[2], **kwargs))


