# t-SNE plot

import os
from typing import Optional, Union

from torch import Tensor
from numpy import ndarray
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

__all__ = [
    'tSNE_plot'
]

def tSNE_plot(latents: Union[Tensor, ndarray], coloring: Union[Tensor, ndarray] = None,
              labels: Union[list[str]] = None, perplexity: int = 50, n_iter: int = 500,
              fig_size: tuple = (20, 6),
                fontsize = 'medium', save_fig: bool = False, path_to_export: str = None,
              random_state: int = None) -> plt.axes:
    """

    Notes
    ------
    The coloring and labels parameters are mutually exclusive. THis means, if both are given, the function
    will prioritize the labels and coloring them accordingly.

    Parameters
    ----------
    latents
    coloring
    labels
    perplexity
    n_iter
    fig_size
    fontsize
    save_fig
    path_to_export
    random_state

    Returns
    -------

    """

    if isinstance(latents, Tensor):
        latents = latents.cpu().detach().numpy()
    if isinstance(coloring, Tensor):
        latents = latents.cpu().detach().numpy()

    if labels is not None:
        assert latents.shape[0] == len(labels), 'The given label list must match the latents in the first dimension.'
    if coloring is not None:
        assert latents.shape[0] == len(labels), 'The given coloring array must match the latents in the first dimension.'

    if save_fig and (path_to_export is None):

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)


    t_SNE = TSNE(n_components=2,
                 perplexity=perplexity,
                 n_iter=n_iter,
                 init='random',
                 random_state=random_state)

    lats_transformed = t_SNE.fit_transform(latents)

    fig, ax = plt.subplots(figsize=fig_size)

    if labels is not None:
        for label in np.unique(labels):
            idx = labels == label
            ax.scatter(lats_transformed[idx, 0], lats_transformed[idx, 1], label=label)
        ax.legend(fontsize=fontsize)
    elif coloring is not None:
        ax.scatter(lats_transformed[:, 0], lats_transformed[:, 1], c=coloring)
    else:
        ax.scatter(lats_transformed[:, 0], lats_transformed[:, 1])

    ax.set_title('t-distributed Stochastic Neighbor Embedding')

    if path_to_export is not None:
        fig.savefig(fname=f'{path_to_export}/tSNE_plot.svg', format='svg')

    return ax