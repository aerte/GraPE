# Butina clustering implementation
from collections import defaultdict
import numpy as np

from rdkit.Chem import MolFromSmiles, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs

from grape.utils import SubSet


def taylor_butina_clustering(data, threshold: float =0.8, nBits: int = 2048, radius: int = 3,
                             split_frac: list[float] = None, log:bool = True) -> tuple[SubSet, SubSet, SubSet]:
    """Clusters the datasets based on Butina clustering [1] and splits it into training, validation and test datasets
    splits. After the molecules are clustered, they are assigned to the train split from largest to smallest until it is
    filled up, then the val split and finally the rest is assigned to the test split. Inspired by the great workshop
    code by Pat Walters, see https://github.com/PatWalters/workshop/blob/master/clustering/taylor_butina.ipynb.

    ----

    References:\n
    [1] Darko Butina, Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast
    and Automated Way To Cluster Small and Large Data Sets, https://doi.org/10.1021/ci9803381 \n
    [2] Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50, 742-754 (2010),
    https://doi.org/10.1021/ci100050t

    -----

    Parameters
    -----------
    data: object
        An object like the DataSet class that can be indexed and stores the SMILES via datasets.smiles.
    threshold: float
        Distance threshold used for the Butina clustering [1]. Default: 0.35.
    nBits: int
        The number of bits used for the Morgan fingerprints [2]. Default: 2048.
    radius: int
        Atom radius used for the Morgan fingerprints [2]. Decides the size of the considered fragments. Default: 3.
    split_frac: list[float]
        List of datasets split fractions. Default: [0.8,0.1,0.1].
    log: bool
        If true, prints a short summary of many single molecule clusters are in the butina clustering. Default: True

    Returns
    ---------
    SubSet, SubSet, SubSet
        Returns the respective lists of Data objects that be fed into a DataLoader.


    """

    all_indices = np.array(data.indices())

    # 1) Finding the fingerprints of the molecules
    fingerprints = []
    for smile in data.smiles:
        mol = MolFromSmiles(smile)
        fingerprints.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits=nBits))
    nPoints = len(fingerprints)

    # 2) Distance matrix for 1-similarity
    dist_matrix = []
    for i in range(1, nPoints):
        similarity = DataStructs.BulkTanimotoSimilarity(fingerprints[i],fingerprints[:i])
        dist_matrix.extend([1-sim for sim in similarity])

    # 3) Clustering
    clusters = Butina.ClusterData(dist_matrix, nPts=nPoints, distThresh=threshold, isDistData=True)

    # 4) Assigning smiles to clustering
    Idx = np.zeros([nPoints,], dtype=np.int32)

    for mol_id, cluster in enumerate(clusters):
        for mol in cluster:
            Idx[mol] = mol_id

    splits = defaultdict()
    splits[0] = []
    processed_len = 0
    split = 0

    single = 0
    for i in np.unique(Idx):
        if np.sum(Idx==i) == 1:
            single+=1
    if log:
        print(f'Number of single molecule clusters: {single} and the ratio is: {single/len(np.unique(Idx)):.3f} of '
              f'single molecule clusters.')


    split_frac = [0.8,0.1,0.1] if split_frac is None else split_frac

    for cluster in np.unique(Idx):
        if processed_len/nPoints >= np.sum(split_frac[:split + 1]):
            split+=1
            if split == 3: break

            splits[split] = []
            #print('next split')

        splits[split].append([point for point in all_indices[Idx==cluster]])
        processed_len += np.sum(Idx==cluster)
        #print(f'{processed_len/nPoints*100}% processed.')

    #splits[0] = np.array(splits[0])
    split_out = dict()

    for i in range(3):
        split_out[i] = []
        for split in splits[0]:
            for item in split:
                split_out[i].append(item)



    return SubSet(data, split_out[0]), SubSet(data, split_out[1]), SubSet(data, split_out[2])