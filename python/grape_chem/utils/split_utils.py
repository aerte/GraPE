# Module for splitting utilities

from typing import Generator, Union
import numpy as np

import dgl

from dgllife.utils import splitters

import torch
import torch.utils.data
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import size_repr

from tqdm import tqdm

__all__ = [
    'SubSet',
    'torch_subset_to_SubSet',
    'split_data',
    'RevIndexedData',
    'RevIndexedSubSet',
    'RevIndexedDataset'
]





##########################################################################
########### Data subsets and utils #######################################
##########################################################################

class SubSet:
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

    def __init__(self, dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices
        self.y = dataset.target[indices]
        self.smiles = dataset.smiles[indices]

        # In the case global features are defined.
        if dataset.global_features is not None:
            self.global_features = dataset.global_features[indices]
        else:
            self.global_features = None
        if hasattr(dataset, 'mol_weights'):
            self.mol_weights = dataset.mol_weights[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if isinstance(item, list):
            return self.dataset[[self.indices[i] for i in item]]
        return self.dataset[self.indices[item]]

    def __iter__(self):
        """

        Returns:

        """

        for item in range(len(self)):
            yield self.dataset[self.indices[item]]

def torch_subset_to_SubSet(subset: Union[dgl.data.Subset,torch.utils.data.Subset]) -> SubSet:
    """Returns the GraPE SubSet from the DGL or PyTorch Subset.

    Parameters
    ----------
    subset: dgl.data.Subset or torch.utils.data.Subset

    Returns
    -------
    SubSet
    """
    #assert isinstance(subset.dataset, grape_chem.utils.DataSet) or isinstance(subset.dataset, grape_chem.utils.GraphDataSet),(
    #    'The subsets underlying dataset has to be either DataSet or GraphDataSet.')
    return SubSet(subset.dataset, subset.indices)

def mult_subset_to_gen(subsets: Union[tuple[dgl.data.Subset], tuple[torch.utils.data.Subset]]) -> Generator:
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


class RevIndexedData(Data):
    """Implementation by Ichigaku Takigawa, 2022, under the MIT License.
    https://github.com/itakigawa/pyg_chemprop/blob/main/pyg_chemprop_naive.py

    Useful in the following circumstances:
    If an algorithm requires the edge indices from node j to node i, PyTorch Geometric however stores the
    indices the other way around.

    This class establishes a new method in the Data class of PyTorch Geometric that stores all the reverse
    indices of all the Data edges as a new feature.

    Parameters
    -----------
    orig: Data
        Original Pytorch Geometric Data object.

    """
    def __init__(self, orig:Data):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys():
                self[key] = orig[key]
            edge_index = self["edge_index"]
            revedge_index = torch.zeros(edge_index.shape[1]).long()
            for k, (i, j) in enumerate(zip(*edge_index)):
                edge_to_i = edge_index[1] == i
                edge_from_j = edge_index[0] == j
                revedge_index[k] = torch.where(edge_to_i & edge_from_j)[0].item()
            self["revedge_index"] = revedge_index

    ########### Increment function, not sure if it is necessary

    #def __inc__(self, key, value, *args, **kwargs):
    #   ################
    #   if key == "revedge_index":
    #       return self.revedge_index.max().item() + 1
    #   else:
    #       return super().__inc__(key, value)
    ############

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))


class RevIndexedDataset(Dataset):
    """Implementation by Ichigaku Takigawa, 2022, under the MIT License.
    https://github.com/itakigawa/pyg_chemprop/blob/main/pyg_chemprop_naive.py

    This class takes a list of Data objects and stores all the new Data objects after applying
    the RevIndexedData class to them. This acts like a Torch Subset or a Grape SubSet, but is limited to the
    graphs itself.

    Parameters
    -----------
    orig: list of Data
        List of Pytorch Geometric Data objects.

    """

    def __init__(self, orig: list[Data]):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in tqdm(orig)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)



class RevIndexedSubSet(SubSet):
    """An extension of the SubSet class with the RevIndexedData class by Ichigaku Takigawa. The input has to
    be the GraPE SubSet object.

    Parameters
    ------------
    subset: SubSet
        A SubSet object.
    """

    def __init__(self, subset:SubSet):
        super(RevIndexedSubSet, self).__init__(subset.dataset, subset.indices)
        self.rev_data = [RevIndexedData(data) for data in tqdm(subset)]

    def __getitem__(self, idx):
        return self.rev_data[idx]

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


def split_data(data, split_type: str = None, split_frac: list[float] = None, custom_split: list = None,
               is_dmpnn:bool = False, seed:int = None, **kwargs) -> (tuple[SubSet, SubSet, SubSet]):
    """ Splits the data based on a split type into SubSets that access the original data via indices.

    Notes
    ------
    The stratified splitter from dgllife requires *two extra* inputs, specifically:

    * labels (array): An array of shape (N,T) where N is the number of datasets points and T is the number of tasks. Used for the
      Stratified Splitter.

    * task_id (int): The task that will be used for the Stratified Splitter.

    For all the other functions, please refer to their respective documentation to see what arguments can be used.

    See Also
    ---------
    https://lifesci.dgl.ai/api/utils.splitters.html

    Parameters
    ------------
    data: Any iterable
        An object that can be accessed per an index and iterated upon. Ex: a DataSet or array object
    split_type: str
        Indicates what split should be used. Default: random. The options are: ['consecutive', 'random',
        'molecular weight', 'scaffold', 'stratified', 'butina_realistic', 'butina', 'custom']
    split_frac: list of float
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: list
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample. Default: None
    is_dmpnn: bool
        Specifies if the input data should be a DMPNN Dataset, ie. the edges will be saved
        directionally. Default: False
    seed: int
        The numpy seed used to generate the splits. Default: None

    Returns
    ---------
    tuple[SubSet, SubSet, SubSet]
        The train, val and test splits respectively.

    """

    from grape_chem.splits import butina_train_val_test_splits, butina_realistic_splits

    if seed is not None:
        np.random.seed(seed)

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
        'butina_realistic': butina_realistic_splits,
        'butina': butina_train_val_test_splits
    }

    if split_type == 'custom' or custom_split is not None:
        assert custom_split is not None and len(custom_split) == len(data), (
            'The custom split has to match the length of the filtered dataset.'
            'Consider saving the filtered output with .get_smiles()')

        custom_split = np.array(custom_split)
        indices = np.arange(len(data))

        return (SubSet(data, indices[custom_split == 0]), SubSet(data, indices[custom_split == 1]),
                SubSet(data, indices[custom_split == 2]))

    elif split_type == 'butina' or split_type == 'butina_realistic':
        train, val, test  = split_func[split_type](data.smiles, **kwargs)
        if is_dmpnn:
            return (RevIndexedSubSet(SubSet(data, train)), RevIndexedSubSet(SubSet(data, val)),
                    RevIndexedSubSet(SubSet(data, test)))
        return SubSet(data, train), SubSet(data, val), SubSet(data, test)


    train, val, test = mult_subset_to_SubSets(split_func[split_type].train_val_test_split(data, frac_train=split_frac[0],
                                                                       frac_test=split_frac[1], frac_val=split_frac[2],
                                                                       **kwargs))
    if is_dmpnn:
        train, val, test = RevIndexedSubSet(train), RevIndexedSubSet(val), RevIndexedSubSet(test)

    return train, val, test
