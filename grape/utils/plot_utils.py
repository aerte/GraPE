# Plot Utilities
from typing import Union
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder

__all__ = [
    "label_encoding"
]

def label_encoding(labels: Union[ndarray, list, dict]) -> tuple[list,list]:
    """A simple wrapper for transforming labels of chemical classes like those from ``classyfire`` into numerical lists.
    Uses scikit-learns LabelEncoder and can take ndarrays, list or dictionaries as input. Will in addition to the
    numeric labels return the unique ones for plotting.

    Parameters
    ----------
    labels: ndarray or list or dict
        Labels to be transformed.

    Returns
    -------
    list, list
        The numeric and unique labels.

    """

    #TODO: Account for labels being the keys instead and classify_compounds generally.

    if isinstance(labels, dict):
        labels = list(labels.values())
    enc = LabelEncoder()
    return enc.fit_transform(labels), list(np.unique(labels))