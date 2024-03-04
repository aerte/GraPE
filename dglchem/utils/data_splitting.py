# Module for data splitting
from collections import defaultdict

from rdkit.Chem import MolFromSmiles, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit import DataStructs

from torch_geometric.data import Data

from dgllife.utils import splitters

def taylor_butina_clustering(data, threshold: float =0.35, nBits: int = 1024, radius: int = 3,
                             split_frac: list =None) -> Data:
    """Clusters the data based on Butina clustering [1] and splits it into training, testing and validation data splits.
    Splitting will occur from largest to smallest cluster. Inspired by the great workshop code by Pat Walters,
    see https://github.com/PatWalters/workshop/blob/master/clustering/taylor_butina.ipynb.

    Parameters
    ----------
    data: object
        An object like the DataSet class that can be indexed and stores the SMILES via data.smiles.
    threshold: float
        Distance threshold used for the Butina clustering [1]. Default: 0.35.
    nBits: int
        The number of bits used for the Morgan fingerprints [2]. Default: 1024.
    radius: int
        Atom radius used for the Morgan fingerprints [2]. Decides the size of the considered fragments. Default: 3.
    split_frac: list of float
        List of data split fractions. Default: [0.8,0.1,0.1].

    Returns
    -------
    train, test, val -
        Returns the respective lists of Data lists that be fed into a DataLoader.

    ----

    References: \n
    [1] Darko Butina, Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast
    and Automated Way To Cluster Small and Large Data Sets, https://doi.org/10.1021/ci9803381 \n
    [2] Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50, 742-754 (2010),
    https://doi.org/10.1021/ci100050t


    """

    # 1) Finding the fingerprints of the molecules
    fingerprints = []
    for smile in data.smiles:
        mol = MolFromSmiles(smile)
        fingerprints.append(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius = radius, nBits=nBits))
    nPoints = len(fingerprints)

    # 2) Distance matrix for 1-similarity
    dist_matrix = []
    for i in range(nPoints):
        similarity = DataStructs.BulkTanimotoSimilarity(fingerprints[i],fingerprints[:i])
        dist_matrix.extend([1-sim for sim in similarity])

    # 3) Clustering
    clusters = Butina.ClusterData(dist_matrix, nPts=nPoints, distThresh=threshold, isDistData=True)

    # 4) Assigning smiles to clustering
    Idx = np.zeros([nPoints,], dtype=np.int8)

    for id, cluster in enumerate(clusters):
        Idx[np.array(cluster)] = id

    splits = defaultdict()
    splits[0] = []
    processed_len = 0
    split = 0

    # Data split:

    split_frac = [0.8,0.1,0.1] if split_frac is None else split_frac

    for cluster in np.unique(Idx):
        if processed_len/nPoints >= np.sum(split_frac[:split + 1]):
            split+=1
            if split == 3: break

            splits[split] = []
            # print('next split')

        splits[split].append([point for point in data[Idx==cluster]])
        processed_len += np.sum(Idx==cluster)
        # print(f'{processed_len/nPoints*100}% processed.')

    return splits[0], splits[1], splits[2]



def split_data(data, split_type: str = None, split_frac = None, custom_split = None, task_id = None):
    """

    Parameters
    ----------
    data: Any iterable
        An object that can be accessed per an index and iterated upon. Ex: a DataSet or np.array object
    split_type: str
        Indicates what split should be used. Default: random. The options are: ['consecutive', 'random',
        'molecular weight', 'scaffold', 'stratified', 'custom']
    split_frac: array
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: array
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample. Default: None

    Returns
    -------
    train, test, val
        - Lists containing the respective data objects.

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
#        'stratified': splitters.SingleTaskStratifiedSplitter
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
        #case 'stratified':
        #    return split_func[split_type].train_val_test_split(data, frac_train=split_frac[0], frac_test=split_frac[1],
        #                                            frac_val=split_frac[2], log_every_n=1000, scaffold_func='decompose')
