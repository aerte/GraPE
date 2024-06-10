# Butina clustering implementation
from collections import defaultdict
from typing import Union, Dict, List, Tuple
import numpy as np
from numpy import ndarray

from rdkit.Chem import MolFromSmiles, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs
from sklearn.model_selection import train_test_split

__all__ = [
    "taylor_butina_clustering",
    "butina_realistic_splits",
    "butina_train_val_test_splits"
]

def taylor_butina_clustering(smiles: list[str], ordering: str='largest_to_smallest',
                             tol:int = 5, threshold: float=0.8, nBits: int = 2048, radius: int = 3,
                             reordering: bool = True) -> Union[ndarray, tuple[dict, dict]]:
    """Clusters the datasets based on Butina clustering [1,2] and splits it into training, validation, and test datasets
    splits. After the molecules are clustered, they are assigned to the train split from largest to smallest until it is
    filled up, then the val split and finally the rest is assigned to the test split. Inspired by the great workshop
    code by Pat Walters, see https://github.com/PatWalters/workshop/blob/master/clustering/taylor_butina.ipynb as well
    as by code provided by Edgar Ivan Sanchez Medina.

    Notes
    ------
    The current function has two ways of returning the clusters:

    * 'largest_to_smallest': Returns one dictionary where the clusters are ordered from the largest cluster to the smallest.

    * 'small_and_large': Returns two dictionaries, one with a number of molecules **less** than a given threshold, and one with a number of molecules **greater** than a given threshold.

    ----

    References:\n
    [1] Darko Butina, Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast
    and Automated Way To Cluster Small and Large Data Sets, https://doi.org/10.1021/ci9803381 \n
    [2] Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50, 742-754 (2010),
    https://doi.org/10.1021/ci100050t

    -----

    Parameters
    -----------
    graphs: object
        An object like the DataSet class that can be indexed and stores the SMILES via datasets.smiles.
    ordering: str
        Decides how the clusters should be returned. The options are: ``largest_to_smallest`` and ``small_and_large``.
    tol: int
        The tolerance for what number of cluster members is considered large if the ordering ``small_and_large``
        was chosen. Default: 5
    threshold: float
        Distance threshold used for the Butina clustering [1]. Default: 0.35.
    nBits: int
        The number of bits used for the Morgan fingerprints [2]. Default: 2048.
    radius: int
        Atom radius used for the Morgan fingerprints [2]. Decides the size of the considered fragments. Default: 3.
    reordering:bool
        If set to True, then the unassigned molecule with the highest number of unassigned neighbors will be
        chosen as the next cluster center. Default: True.

    Returns
    ---------
    tuple[dict, dict] or dict


    """

    # 1) Finding the fingerprints of the molecules
    fingerprints = []
    for smile in smiles:
        mol = MolFromSmiles(smile)
        fingerprints.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits=nBits))
    nPoints = len(fingerprints)

    # 2) Distance matrix for 1-similarity
    dist_matrix = []
    for i in range(1, nPoints):
        similarity = DataStructs.BulkTanimotoSimilarity(fingerprints[i],fingerprints[:i])
        dist_matrix.extend([1-sim for sim in similarity])

    # 3) Clustering
    clusters = Butina.ClusterData(dist_matrix, nPts=nPoints, distThresh=threshold, isDistData=True,
                                  reordering=reordering)

    # 4) Assigning smiles to clustering

    if ordering == 'largest_to_smallest':
        return clusters

    elif ordering == 'small_and_large':
        small_clusters, large_clusters = {}, {}
        for mol_id, cluster in enumerate(clusters):
            if len(cluster) < tol:
                small_clusters[mol_id] = cluster
            else:
                large_clusters[mol_id] = cluster

        return small_clusters, large_clusters


    else:
        raise ValueError(f'Invalid ordering selected: {ordering}')

    #split_frac = [0.8,0.1,0.1] if split_frac is None else split_frac

    #for cluster in np.unique(Idx):
    #    if processed_len/nPoints >= np.sum(split_frac[:split + 1]):
    #        split+=1
    #        if split == 3: break

    #        splits[split] = []
    #        #print('next split')
#
#        splits[split].append([point for point in all_indices[Idx==cluster]])
#        processed_len += np.sum(Idx==cluster)
#        #print(f'{processed_len/nPoints*100}% processed.')#
#
#    #splits[0] = np.array(splits[0])
#    split_out = dict()
#
#    for i in range(3):
#        split_out[i] = []
#        for split in splits[0]:
#            for item in split:
#                split_out[i].append(item)



#    return SubSet(graphs, split_out[0]), SubSet(graphs, split_out[1]), SubSet(graphs, split_out[2])

def butina_train_val_test_splits(smiles: list[int], split_frac: list[int] = [0.8, 0.1, 0.1], tol: int = 5,
                                 random_state=None, **kwargs) -> tuple[list[int], list[int], list[int]]:
    """Clusters the datasets based on Butina clustering [1,2] and splits it into training, validation, and test datasets
    splits.

    After the molecules are clustered, we only take the **large** clusters as specified by the tol parameter. Any
    cluster with fewer members than the tol parameter will not be included. Inspired by some code provided by
    Edgar Ivan Sanchez Medina

    Notes
    ------
    This function can take *additional arguments* to modify the taylor butina clustering parameters, see the documentation
    of function ``tylor_butina_clustering`` for more information.

    ----

    References:\n
    [1] Darko Butina, Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast
    and Automated Way To Cluster Small and Large Data Sets, https://doi.org/10.1021/ci9803381 \n
    [2] Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50, 742-754 (2010),
    https://doi.org/10.1021/ci100050t

    Parameters
    ----------
    smiles: list[str]
        List of SMILES strings.
    split_frac: list[int]
        List of split fractions, specifically for the train, validation and test sets. Default: [0.8,0.1,0.1]
    tol: int
        The tolerance for what number of cluster members is considered large if the ordering ``small_and_large``
        was chosen. Default: 5
    random_state: int
        Random seed for reproducibility. Default: None



    Returns
    -------
    tuple[list[int], list[int], list[int]]

    """

    train_index, val_index, test_index = [], [], []

    _, large_clusters = taylor_butina_clustering(smiles, ordering='small_and_large', **kwargs)
    val_size = split_frac[1]/(split_frac[1]+split_frac[2])
    test_size = 1-val_size

    for key in large_clusters:
        train_i, remain_i = train_test_split(large_clusters[key], train_size=split_frac[0],
                                             test_size=1-split_frac[0],
                                             random_state=random_state)
        train_index.extend(train_i)

        if len(remain_i) <= 1:
            test_index.extend(remain_i)
        else:
            val_i, test_i = train_test_split(remain_i, train_size=val_size,
                                           test_size=test_size,
                                           random_state=random_state)

            val_index.extend(val_i)
            test_index.extend(test_i)

    return train_index, val_index, test_index

def butina_realistic_splits(smiles, split_frac: list[int] = [0.8,0.1,0.1], **kwargs) -> (
        tuple)[list[int], list[int], list[int]]:
    """Clusters the datasets based on Butina clustering [1,2] and splits it into training, validation, and test datasets
    splits.

    After the molecules are clustered, they are assigned to the train split from largest to smallest until it is
    filled up, then the val split and finally the rest is assigned to the test split. This results in a realistic
    graphs split, as demonstrated by [3].

    Notes
    ------
    This function can take *additional arguments* to modify the taylor butina clustering parameters, see the documentation
    of function ``tylor_butina_clustering`` for more information.


    ----

    References:\n
    [1] Darko Butina, Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast
    and Automated Way To Cluster Small and Large Data Sets, https://doi.org/10.1021/ci9803381 \n
    [2] Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50, 742-754 (2010),
    https://doi.org/10.1021/ci100050t\n
    [3] Martin, Eric J. and Polyakov, Valery R. and Zhu, Xiang-Wei and Tian, Li and Mukherjee, Prasenjit and Liu, Xin,
    All-Assay-Max2 pQSAR: Activity Predictions as Accurate as Four-Concentration IC50s for 8558 Novartis Assays,
    https://doi.org/10.1021/acs.jcim.9b00375

    Parameters
    ----------
    smiles: list[str]
        List of SMILES strings.
    split_frac: list[int]
        List of split fractions, specifically for the train, validation and test sets. Default: [0.8,0.1,0.1]

    Returns
    -------
    tuple[list[int], list[int], list[int]]

    """

    clusters = taylor_butina_clustering(smiles, ordering='largest_to_smallest', **kwargs)

    nPoints = len(smiles)
    all_indices = np.arange(nPoints)
    Idx = np.zeros([nPoints, ], dtype=np.int32)

    for mol_id, cluster in enumerate(clusters):
        for mol in cluster:
            Idx[mol] = mol_id

    splits = defaultdict()
    splits[0] = []
    processed_len = 0
    split = 0

    Idx = np.zeros([nPoints, ], dtype=np.int32)

    for mol_id, cluster in enumerate(clusters):
        for mol in cluster:
            Idx[mol] = mol_id

    for cluster in np.unique(Idx):
        if processed_len/nPoints >= np.sum(split_frac[:split + 1]):
            split+=1
            if split == 3: break

            splits[split] = []
            #print('next split')

        splits[split].append([point for point in all_indices[Idx==cluster]])
        processed_len += np.sum(Idx==cluster)
        # print(f'{processed_len/nPoints*100}% processed.')

    split_out = dict()

    for i in range(3):
        split_out[i] = []
        for split in splits[i]:
            for item in split:
                split_out[i].append(item)


    return split_out[0], split_out[1], split_out[2]
