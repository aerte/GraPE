# PCA plot

import os
from typing import Optional, Union

import sklearn.preprocessing
from torch import Tensor
from numpy import ndarray
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'pca_2d_plot'
]


def pca_2d_plot(latents: Union[Tensor, ndarray], labels: Union[list[str]] = None, fig_size: tuple = (20, 6),
                fontsize = 'medium', save_fig: bool = False, path_to_export: str = None) -> plt.axes:
    """A function that projects latents or any other matrix onto it's first two principal components and plots it. Can
    use given labels to colorcode the projections.

    Parameters
    ----------
    latents: Tensor or ndarray
        Latents or any other matrix that will be used for a PCA model and projected on the two first principal
        components. The first dimension should be the observations and the second the features.
    labels: list[str]
        Optional list of labels that will be used for the plot. *Has to be equal to the latent length.* Default: None
    fig_size: tuple
        The output figure size. Default: (20,6)
    fontsize: str
        Decides the fontsize of the legend. Default: 'medium'
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None

    Returns
    -------
    plt.axes

    """

    if isinstance(latents, Tensor):
        latents = latents.cpu().detach().numpy()

    if labels is not None:
        assert latents.shape[0] == len(labels), 'The given label list must match the latents.'

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

    model_pca = sklearn.decomposition.PCA(n_components=2)
    model_pca.fit(latents)

    V = model_pca.components_.T
    projection = latents @ V

    fig, ax = plt.subplots(figsize=fig_size)

    if labels is not None:
        for label in np.unique(labels):
            idx = np.array(labels) == np.array(label)
            ax.scatter(projection[idx, 0], projection[idx, 1], label=label)
        ax.legend(fontsize=fontsize)
    else:
        ax.scatter(projection[:, 0], projection[:, 1])
    ax.set_title('PCA using the two first PC\'s')

    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/latent_plot.svg', format='svg')

    return ax