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

def tSNE_plot(latents: Union[Tensor, ndarray], labels: Union[list[str]] = None,
              coloring: Union[Tensor, ndarray] = None, perplexity: int = 50, n_iter: int = 500,
              fig_size: tuple = (20, 6), fontsize = 'medium', save_fig: bool = False,
              path_to_export: str = None, random_state: int = None) -> plt.axes:
    """Plots the t-SNE (t-distributed Stochastic Neighbor Embedding) [1] of latents or any array input.
    The t-SNE is a non-linear dimensionality reduction technique that uses probabilistic similarity measures
    to determine the relative distance between each pair of latents. The result for 2 components is then a
    two-dimensional array hopefully providing insight in the underlying structure of the data.

    The t-SNE is notorious for being unstable and highly dependent on the hyperparameters (perplexity, number of
    iterations) so use with caution. **It is also not guaranteed to converge.**


    Notes
    ---------
    The coloring and labels parameters are mutually exclusive. This means, if both are given, the function
    will prioritize the labels and color the points accordingly. **It is recommended to use the labels
    argument.**


    References
    -----------
    [1] van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-sne. Journal of Machine
    Learning Research, 9 (86), 2579–2605. http://jmlr.org/papers/v9/vandermaaten08a.html

    Parameters
    ----------
    latents: Tensor or ndarray
        Latents or any other matrix that will be used for a PCA model and projected on the two first principal
        components. The first dimension should be the observations and the second the features.
    labels: list[str]
        Optional list of labels that will be used for the plot. *Has to be equal to the latent length.* Default: None
    coloring: list[str]
        Optional list of colors that will be used for the plot. *Has to be equal to the latent length.* Default: None
    perplexity: int
        Approximately the same as the number of neighbors for each point that the t-SNE will (try)
        to generate. Default: 50
    n_iter: int
        Number of iterations of the t-SNE algorithm will run for. Default: 500
    fig_size: tuple
        The output figure size. Default: (20,6)
    fontsize: str
        Decides the fontsize of the legend. Default: 'medium'
    save_fig: bool
        Decides if the plot is saved, is overridden if a path is given. Default: False
    path_to_export: str
        File location to save. Default: None
    random_state: int
        An optional argument to pass to the skikit-learn t-SNE function to seed the number
        generator. Default: None

    Returns
    -------
    plt.axes

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
            idx = np.array(labels) == np.array(label)
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