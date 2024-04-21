## Various data structures that can be used in analysis or ML models

import torch
import torch.nn as nn
import torch.utils.data
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import size_repr
from grape.utils import SubSet

from tqdm import tqdm

__all__ = [
    'RevIndexedData',
    'RevIndexedDataset',
    'RevIndexedSubSet'
]


class RevIndexedData(Data):
    """Implementation by Ichigaku Takigawa, 2022, under the MIT License.
    https://github.com/itakigawa/pyg_chemprop/blob/main/pyg_chemprop_naive.py

    Useful in the following circumstances:
    If an algorithm requires the edge indices from node j to node i, PyTorch Geometric however stores the
    indices the other way around.

    This class establishes a new method in the Data class of PyTorch Geometric that stores all the reverse
    indices of all the Data edges as a new feature.

    """
    def __init__(self, orig):
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
    data itself.

    """

    def __init__(self, orig):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in tqdm(orig)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


# TODO: Maybe make a smarter integration of this into the pre-existing system.
class RevIndexedSubSet(SubSet):
    """An extension of the SubSet class with the RevIndexedData class by Ichigaku Takigawa.

    """

    def __init__(self, subset):
        super(RevIndexedSubSet, self).__init__(subset.dataset, subset.indices)
        self.rev_data = [RevIndexedData(data) for data in tqdm(subset)]

    def __getitem__(self, idx):
        return self.rev_data[idx]