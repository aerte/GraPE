# Graph constructor built on top of pytorch geometric

import os
from collections.abc import Sequence
from typing import Union

import pickle

import pandas as pd
import numpy as np
from numpy import ndarray
import rdkit.Chem.Draw
import torch
from torch import tensor, Tensor

from rdkit import Chem
from rdkit.Chem import rdmolops, MolFromSmiles, Draw, MolToSmiles
from rdkit import RDLogger
import seaborn as sns


from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from grape_chem.utils.featurizer import AtomFeaturizer, BondFeaturizer
#from grape_chem.analysis import smiles_analysis
from grape_chem.utils.split_utils import split_data, SubSet
from grape_chem.utils.feature_func import mol_weight

from sklearn.preprocessing import StandardScaler

import mlflow
from typing import Dict, Union, List


RDLogger.DisableLog('rdApp.*')

__all__ = ['filter_smiles',
           'is_multidimensional',
           'construct_dataset',
           'DataLoad',
           'DataSet',
           'GraphDataSet',
           'load_dataset_from_excel',
           'load_dataset_from_csv',
           'extract_data_from_dataframe',
           'extract_data_from_multiple_dataframes']

def standardize(x, mean, std):
    return (x - mean) / std


##########################################################################
########### filtering ####################################################
##########################################################################

def filter_smiles(smiles: list[str], target: Union[list[str], list[float], ndarray], allowed_atoms: list[str] = None,
                  only_organic: bool = True, allow_dupes: bool = False, log: bool = False,
                  global_feats = None) -> Union[list,list]:
    """Filters a list of smiles based on the allowed atom symbols.

    Parameters
    ------------
    smiles: list of str
        Smiles to be filtered.
    target: list of int or list of float or ndarray
        Target of the graphs.
    allowed_atoms: list of str
        Valid atom symbols, non-valid symbols will be discarded. Default: [``C``, ``N``, ``O``, ``S``, ``F``, ``Cl``,
         ``Br``, ``I``, ``P``]
    only_organic: bool
        Checks if a molecule is ``organic`` counting the number of ``C`` atoms. If set to True, then molecules with less
        than one carbon will be discarded. Default: True
    log: bool
        Determines if there should be print-out statements to indicate why mols were filtered out. Default: False
    allow_dupes: bool
        Decides if duplicate smiles should be allowed. Default: False

    Returns
    ----------
    list[str]
        A list of filtered smiles strings.

    """

    if allowed_atoms is None:
        allowed_atoms = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']

    if global_feats is not None:
        # New implementation to handle nD global features
        print("##########################################################################################")
        print("Multidimensional global features: ", is_multidimensional(global_feats), "type: ", type(global_feats))	
        #print("Global features shape: ", len(global_feats), len(global_feats[0]))
        #print("global_feats: ", global_feats)
        print("##########################################################################################")

        if is_multidimensional(global_feats):
            # Column name for each global feature
            n_features = global_feats.shape[1]
            global_feat_columns = [f'global_feat_{i}' for i in range(n_features)]
            
            # Dictionary with global feature columns
            global_feats_dict = {f'global_feat_{i}': global_feats[:, i] for i in range(n_features)}
            
            # Combine the dictionaries to create the DataFrame
            df = pd.DataFrame({'smiles': smiles, 'target': target, **global_feats_dict})
        else:
            # If global_feats is 1D, add it as a single column (original GraPE implementation)
            df = pd.DataFrame({'smiles': smiles, 'target': target, 'global_feat': global_feats})

    elif target.ndim > 1:
        target_dict = {f'target_{i}': target[:, i] for i in range(target.shape[1])}
        df = pd.DataFrame({'smiles': smiles, **target_dict})
    else:
        df = pd.DataFrame({'smiles': smiles, 'target': target})

    indices_to_drop = []
    for element in smiles:
        mol = Chem.MolFromSmiles(element)

        if mol is None:
            if log:
                print(f'SMILES {element} in index {list(df.smiles).index(element)} is not valid.')
            indices_to_drop.append(list(df.smiles).index(element))

        else:
            if mol.GetNumHeavyAtoms() < 2:
                if log:
                    print(f'SMILES {element} in index {list(df.smiles).index(element)} consists of less than 2 heavy atoms'
                        f' and will be ignored.')
                indices_to_drop.append(list(df.smiles).index(element))
            else:
                carbon_count = 0
                for idx, atoms in enumerate(mol.GetAtoms()):
                    #print(f"atom {idx} ",atoms.GetSymbol(), "in molecule ", element)
                    if atoms.GetSymbol() not in allowed_atoms:
                        if log:
                            print(f'SMILES {element} in index {list(df.smiles).index(element)} contains the atom {atoms.GetSymbol()} that is not'
                                f' permitted and will be ignored.')
                        indices_to_drop.append(list(df.smiles).index(element))
                    else:
                        if atoms.GetSymbol() == 'C':
                            carbon_count += 1

                if carbon_count < 1 and only_organic:
                    indices_to_drop.append(list(df.smiles).index(element))
                    if log:
                        print(f'SMILES {element} in index {list(df.smiles).index(element)} does not contain at least one'
                            f' carbon and will be ignored.')

    indices_to_drop = list(indices_to_drop)
    unique_indices = list(set(indices_to_drop))
    print(f"Number of smiles to drop: {len(unique_indices)} out of {len(smiles)}")
    print("Length of df before dropping: ", len(df))
    df.drop(indices_to_drop, inplace=True)
    print("Length of df after dropping: ", len(df))
    df.reset_index(drop=True, inplace=True)

    # map smiles to mol and back to ensure SMILE notation consistency
    mols = df.smiles.map(lambda x: MolFromSmiles(x))
    df.smiles = mols.map(lambda x: MolToSmiles(x))

    if not allow_dupes:
        df.drop_duplicates(subset='smiles', inplace=True)

    if global_feats is not None:
        if global_feats.ndim > 1:
            return np.array(df.smiles), np.array(df.target), np.array(df[global_feat_columns])
        return np.array(df.smiles), np.array(df.target), np.array(df.global_feat)
    
    if target.ndim > 1:
        return np.array(df.smiles), np.array(df.drop(columns=['smiles'])), None
    return np.array(df.smiles), np.array(df.target), None

def is_multidimensional(lst):
    return any(isinstance(i, list) for i in lst)


# Remove duplicates while keeping NaNs
def filter_duplicates(group):
    # Keep the group only if there's no identical rows
    if len(group) > 1 and (group.dropna().duplicated(keep=False).any()):
        return pd.DataFrame(columns=group.columns)  # return empty DataFrame if duplicates exist
    return group



##########################################################################
########### constructing a dataset #######################################
##########################################################################

def construct_dataset(smiles: list[str], target: Union[list[int], list[float], ndarray], allowed_atoms: list[str] = None,
                      atom_feature_list: list[str] = None, bond_feature_list: list[str] = None,
                      global_features = None) -> list[Data]:
    """Constructs a dataset out of the smiles and target lists based on the feature lists provided. The dataset will be
    a list of torch geometric Data objects, using their conventions.

    Parameters
    ------------
    smiles : list of str
        Smiles that are featurized and passed into a PyG DataSet.
    target: list of int or list of float or ndarray
        Array of values that serve as the graph 'target'.
    allowed_atoms : list of str
        Smiles that are considered in featurization. Default: [``C``, ``N``, ``O``, ``S``, ``F``, ``Cl``, ``Br``,
        ``I``, ``P``]
    atom_feature_list : list of str
        Features of the featurizer, see utils.featurizer for more details. Default: All implemented features.
    bond_feature_list : list of str
        Bond features of the bond featurizer, see utils.featurizer for more details. Default: All implemented features.
    global_features
        A list of global features matching the length of the SMILES or target. Default: None

    Returns
    --------
    list of Data
        list of Pytorch-Geometric Data objects

    """
    atom_featurizer = AtomFeaturizer(allowed_atoms=allowed_atoms,
                                    atom_feature_list = atom_feature_list)

    bond_featurizer = BondFeaturizer(bond_feature_list=bond_feature_list)


    data = []

    for (smile, i) in zip(smiles, range(len(smiles))):
        mol = MolFromSmiles(smile) # get into rdkit object
        edge_index = dense_to_sparse(torch.tensor(rdmolops.GetAdjacencyMatrix(mol)))[0]
        x = atom_featurizer(mol) #creates nodes
        edge_attr = bond_featurizer(mol) #creates "edges" attrs
        data_temp = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y=tensor([target[i]],
                                                                                         dtype=torch.float32))

        if global_features is not None:
            data_temp['global_feats'] = torch.tensor([global_features[i]], dtype=torch.float32)
        else:
            data_temp['global_feats'] = None


        data.append(data_temp)

    return data #actual pyg graph


##########################################################################
########### Data classes #################################################
##########################################################################


class DataLoad(object):
    """The basic datasets-loading class. It holds the root as well as the raw  and processed files directories.

    """
    def __int__(self, root = None):
        if root is None:
            root = './data'
        self.root = root

    @property
    def raw_dir(self):
        """

        Returns: path

        """
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self):
        """

        Returns: path

        """
        return os.path.join(self.root, 'processed')




class DataSet(DataLoad):
    """A class that takes a path to a pickle file or a list of smiles and targets. The datasets is stored in
    Pytorch-Geometric Data instances and be accessed like an array.

    Heavily inspired by the PyTorch-Geometric Dataset class, especially indices and index_select.

    Parameters
    ------------
    file_path: str
        The path to a pickle file that should be loaded and the datasets therein used.
    smiles: list of str
        List of smiles to be made into a graph.
    target: list of in or float
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list of str
        List of allowed atom symbols.
    only_organic: bool
        Checks if a molecule is ``organic`` counting the number of ``C`` atoms. If set to True, then molecules with less
        than one carbon will be discarded. Default: True
    atom_feature_list: list of str
        List of features to be applied. Default: All implemented features.
    bond_feature_list: list of str
        List of features that will be applied. Default: All implemented features.
    log: bool
        Decides if the filtering output and other outputs will be shown.


    """
    def __init__(self, file_path: str = None, smiles: list[str] = None, target: Union[list[int], list[float],
    ndarray] = None, global_features:Union[list[float], ndarray] = None, filter: bool=True,allowed_atoms:list[str] = None,
    only_organic: bool = True, allow_dupes: bool = False, atom_feature_list: list[str] = None, bond_feature_list: list[str] = None,
    log: bool = False, root: str = None, indices:list[int] = None, mean: Union[float, List[float]] = None, std: float = None):


        super().__int__(root)
        if smiles is None:
            if allowed_atoms is not None and atom_feature_list is not None and bond_feature_list is not None and mean is not None and std is not None:
                self.allowed_atoms = allowed_atoms
                self.atom_feature_list = atom_feature_list
                self.bond_feature_list = bond_feature_list
                self.mean = mean
                self.std = std
                
            else:
                raise ValueError('If smiles are not given, allowed_atoms, atom_feature_list, bond_feature_list, mean and std must be given.')

        else:
            assert (file_path is not None) or (smiles is not None and target is not None),'path or (smiles and target) must given.'
            if file_path is not None:
                with open(file_path, 'rb') as handle:
                    try:
                        df = pd.read_pickle(handle)
                        print('Loaded dataset.')
                    except:
                        raise ValueError('A dataset is stored as a DataFrame.')

                self.smiles = np.array(df.smiles)
                print("Smiles: ", self.smiles)

                self.global_features = np.array(df.global_features)
                self.graphs = list(df.graphs)

            else:
                if filter:
                    self.smiles, self.raw_target, self.global_features = filter_smiles(smiles, target,
                                                                                    allowed_atoms= allowed_atoms,
                                                                                        only_organic=only_organic, log=log,
                                                                                        allow_dupes=allow_dupes,
                                                                                    global_feats=global_features)

                else:
                    self.smiles, self.raw_target = np.array(smiles), np.array(target)
                    self.global_features = np.array(global_features) if global_features is not None else None

                self.target = self.raw_target

                self.graphs = construct_dataset(smiles=self.smiles,
                                                target=self.target,
                                                global_features=self.global_features,
                                                allowed_atoms = allowed_atoms,
                                                atom_feature_list = atom_feature_list,
                                                bond_feature_list = bond_feature_list)

                self.global_features = global_features

            self.allowed_atoms = allowed_atoms
            self.atom_feature_list = atom_feature_list
            self.bond_feature_list = bond_feature_list
            self._indices = indices
            self.num_node_features = self.graphs[0].num_node_features
            self.num_edge_features = self.graphs[0].num_edge_features
            self.data_name=None
            self.mean, self.std = mean, std

            self.mol_weights = np.zeros(len(self.smiles))

            for i in range(len(self.smiles)):
                self.mol_weights[i] = mol_weight(Chem.MolFromSmiles(self.smiles[i]))

    @staticmethod
    def standardize(target):
        if isinstance(target, Tensor):
            target = target.cpu().detach().numpy()
        target_ = (target - np.mean(target)) / np.std(target)
        return target_, np.mean(target), np.std(target)

    @staticmethod
    def rescale(target, mean, std):
        return (target *std) + mean

    def rescale_data(self, target, index = None):
        num_targets = self.mean.shape[0] if self.mean.ndim > 0 else 1
        if num_targets > 1:  
            if self.mean is None or self.std is None:
                mean, std = np.mean(target), np.std(target)
            else:
                mean, std = self.mean[index].item(), self.std[index].item()
            return self.rescale(target, mean, std)
        else:
            if self.mean is None or self.std is None:
                mean, std = np.mean(self.target), np.std(self.target)
            else:
                mean, std = self.mean, self.std
            print("Target: ", target)
            print("Mean: ", mean)
            print("Std: ", std)
            return self.rescale(target, mean, std)
        

    def get_mol_weight(self):
        """Calculates the molecular weights of the DataSet smiles.

        Returns
        -------
        ndarray
            The molecular weight of the stored smiles.

        """
        weights = np.zeros(len(self.smiles))

        for i in range(len(self.smiles)):
            weights[i] = mol_weight(Chem.MolFromSmiles(self.smiles[i]))

        return weights



    def save_dataset(self, filename:str=None):
        """Loads the dataset (specifically the smiles, target, global features and graphs) to a pandas DataFrame and
        dumps it into a pickle file. Saving and loading the same dataset is about 20x faster than recreating it from
        scratch.

        Parameters
        ------------
        filename: str
            Name of the datasets file. Default: 'dataset'


        """
        if filename is None:
            if self.data_name is None:
                filename = 'dataset'
            else:
                filename = self.data_name

        path = self.processed_dir

        if not os.path.exists(path):
            os.makedirs(path)

        df_save = pd.DataFrame({'smiles':self.smiles,'target':self.raw_target,'global_features':self.global_features,
                   'graphs':self.graphs})

        path = os.path.join(path,filename+'.pickle')

        with open(path,'wb') as handle:
            pickle.dump(df_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f'File saved at: {path}')


    def save_smiles(self, path: str =None):
        """Save the smiles as a text file in a given location.

        Parameters
        ------------
        path: str
            Path where the smiles should be saved. Default: The processed files' directory.

        """

        path = self.processed_dir if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt(path+'/filtered_smiles.txt', X = np.array(self.smiles), fmt='%s')

    def get_mol(self) -> list[rdkit.Chem.Mol]:
        """Return a list containing all rdkit.Chem.Mol of the SMILES in the DataSet.

        Returns
        ---------
        list(rdkit.Chem.Mol)

        Example
        -------
        >>> from grape_chem.datasets import BradleyDoublePlus
        >>> dataset = BradleyDoublePlus()
        >>> # Return the first 5 mol objects:
        >>> dataset.get_mol()[:5]
        [<rdkit.Chem.rdchem.Mol at 0x31cee26c0>,
         <rdkit.Chem.rdchem.Mol at 0x31cee2960>,
         <rdkit.Chem.rdchem.Mol at 0x31cee27a0>,
         <rdkit.Chem.rdchem.Mol at 0x31cee28f0>,
         <rdkit.Chem.rdchem.Mol at 0x31cee2a40>]

        """

        return list(map(lambda x: MolFromSmiles(x), self.smiles))

    def indices(self):
        return range(len(self.graphs)) if self._indices is None else self._indices

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return self.graphs[idx]
        else:
            return self.index_select(idx)

    def __iter__(self):
        for i in range(len(self.graphs)):
            yield self.graphs[i]

    def index_select(self, idx:object):
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.

        Modified from https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/data/dataset.html#Dataset

        Parameters
        ----------
            idx: obj
                Index list of datasets objects to retrieve.

        Returns
        -------
        list
            Python list of datasets objects.

        """

        indices = self.indices()

        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        return [self[i] for i in indices]

    def draw_smile(self, index: int = None, smile:str = None)->rdkit.Chem.Draw.MolDraw2D:
        """Draw the smile at a desired index or a passed SMILE, best used in the jupyter environment.

        Parameters
        ----------
        index: int
            The index of the smile that should be drawn.
        smile:str
            The smile that should be drawn.

        Returns
        -------
        Image

        """
        if index is None and smile is None:
            index = np.random.choice(len(self), size=1)
        elif smile is not None:
            Draw.MolToImage(MolFromSmiles(smile))
        return Draw.MolToImage(MolFromSmiles(self.smiles[index]), size=(100,100))



    def analysis(self, path_to_export:str = None, download:bool=False, plots:list = None, save_plots:bool = False,
                 fig_size:tuple[int,int]=None, filter_output_txt:bool = True) -> tuple:
        """Returns an overview of different aspects of the smiles dataset **after filtering** according to:
        https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/utils/analysis.py.
        This includes the frequency of symbols, degree frequency and more.

        Parameters
        ----------
        path_to_export: str
            Path to the folder where analysis results should be saved. Default: None (recommended).
        download: bool
            Decides if the results are downloaded. If either the path is given or download is set to true, the
            analysis results will be downloaded as a txt file.
        plots: list of str
            Bar plots of the analysis results. Default: None. Possible options are:
            ['atom_type_frequency', 'degree_frequency', 'total_degree_frequency', 'explicit_valence_frequency',
            'implicit_valence_frequency', 'hybridization_frequency', 'total_num_h_frequency', 'formal_charge_frequency',
            'num_radical_electrons_frequency', 'aromatic_atom_frequency', 'chirality_tag_frequency',
            'bond_type_frequency', 'conjugated_bond_frequency', 'bond_stereo_configuration_frequency',
            'bond_direction_frequency']
        save_plots: bool
            Decides if the plots are saved in the processed folder.
        fig_size: list
            2-D list to set the figure sizes. Default: [10,6]
        filter_output_txt: bool
            Filters the output text file of some excessive information. Default: True
            .

        Returns
        -------
        dictionary
            Summary of the results.
        figures (optional)
            Bar plots of the specified results.

        """
        from grape_chem.analysis import smiles_analysis

        return smiles_analysis(self.smiles, path_to_export, download, plots, save_plots, fig_size, filter_output_txt)



    def get_splits(self, split_type:str = None, split_frac:list[float, float, float] = None,
                   custom_split:list[list] = None, **kwargs):
        """ Returns the dataset split into training, testing and validation based on the given split type.

        Parameters
        ----------
        split_type: str
            Indicates what split should be used. It will either take a new argument or default
            to the initialized split fractions. The default initialization is 'random'. The options are:
            ['consecutive', 'random', 'molecular weight', 'scaffold', 'stratified', 'custom']
        split_frac: array
            Indicates what the split fractions should be. It will either take a new argument or default
            to the initialized split fractions. The default initialization is [0.8,0.1,0.1].
        custom_split: array
            The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
            where 0 indicates a training sample, 1 a testing sample and 2 a validation sample. It will either take
            a new argument of default to the initialized custom split. The default initialization is None.

        Returns
        -------
        SubSet, SubSet, SubSet
            List containing the respective datasets objects.

        """

        split_type = 'random' if split_type is None else split_type
        split_frac = [0.8, 0.1, 0.1] if split_frac is None else split_frac

        return split_data(data = self, split_type = split_type,
                            split_frac = split_frac, custom_split = custom_split, **kwargs)

    def generate_global_feats(self, seed:int = None):
        """Generates normally distributed random global features, fx. used for MEGNet. Subsequently,
        the global features are added to all the dataset graphs."""
        if seed is not None:
            np.random.seed(seed)
        self.global_features = np.random.randn(len(self))

        for i in range(len(self.graphs)):
            self.graphs[i]['global_feats'] = self.global_features[i]


    @staticmethod
    def scale_array(array, mean, std):
        return (array-mean)/std

    def split_and_scale(self, scale: bool = True, seed:int=None, return_scaling_params:bool = False, split_type:str = None,
                    split_frac:list[float, float, float] = None, custom_split:list[list] = None,
                    is_dmpnn: bool = False, **kwargs):
        """Splits and (optionally) rescales the dataset based on the training set. If it is rescaled,
        the mean and std will be saved inside the DataSet object.

        Parameters
        ----------
        scale: bool
            Decides if the data should be rescaled using the training set. Default: True.
        seed: int
            Numpy seed used for splitting. Default: None.
        return_scaling_params: bool
            Decides if the rescaling parameters (mean, std) should be returned. Default: False.
        split_type: str
            Indicates what split should be used. It will either take a new argument or default
            to the initialized split fractions. The default initialization is 'random'. The options are:
            ['consecutive', 'random', 'molecular weight', 'scaffold', 'stratified', 'butina_realistic', 'butina',
            'custom']
        split_frac: array
            Indicates what the split fractions should be. It will either take a new argument or default
            to the initialized split fractions. The default initialization is [0.8,0.1,0.1].
        custom_split: array
            The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
            where 0 indicates a training sample, 1 a testing sample and 2 a validation sample. It will either take
            a new argument of default to the initialized custom split. The default initialization is None.
        is_dmpnn: bool
            Specifies if the input data should be a DMPNN Dataset, ie. the edges will be saved directionally.
            Default: False


        Returns
        -------
        SubSet, SubSet, SubSet
            List containing the respective datasets objects.

        """
        if seed is not None:
            np.random.seed(seed)

        split_type = 'random' if split_type is None else split_type
        split_frac = [0.8, 0.1, 0.1] if split_frac is None else split_frac

        if scale == True:
            train, val, test = split_data(data = self, split_type = split_type,
                                split_frac = split_frac, custom_split = custom_split, **kwargs)

            # We use the training set for scaling
            # allow for multidimensional targets
            if train.y.ndim > 1:
                self.mean, self.std = np.nanmean(train.y, axis=0), np.nanstd(train.y, axis=0)

            else:
                train_y, self.mean, self.std = self.standardize(train.y)
            self.target = self.scale_array(self.target, self.mean, self.std)

            if train.global_features is not None:
                # Again, we use the training set for scaling the global features
                train_glob, self.mean_gobal_feats, self.std_gobal_feats = self.standardize(train.global_features)
                self.gobal_features = self.scale_array(self.global_features, self.mean_gobal_feats,
                                                       self.std_gobal_feats)

            self.graphs = construct_dataset(smiles=self.smiles,
                                            target = self.target,
                                            global_features=self.global_features,
                                            allowed_atoms=self.allowed_atoms,
                                            atom_feature_list=self.atom_feature_list,
                                            bond_feature_list=self.bond_feature_list)

        train, val, test = split_data(self, split_type=split_type, split_frac=split_frac, custom_split=custom_split,
                   is_dmpnn=is_dmpnn,**kwargs)
        if return_scaling_params is True:
            return train, val, test, self.mean, self.std
        return train, val, test

    def predict_smiles(self, smiles:Union[str, list[str]], model, mean:float = None,
                       std:float = None) -> dict:
        """A dataset-dependent prediction function. When a SMILE or a list of SMILES is passed together
        with a model that used the **same dataset or property** for training, that specific property is
        predicted from the passed SMILES.

        DataSet -> Model -> new smiles -> prediction and rescaling

        Notes
        ------
        The DataSet used for training **must have the same atom/bond features and target property**. But in a practical
        setting, one just has to make sure that the model predict the same target property given in the
        DataSet.


        Parameters
        ----------
        smiles: str and list of str
            A SMILE or a list of SMILES that will be used for prediction.
        model:
            A model using the **same features** as defined for the graphs inside the DataSet.
        mean: float
            A float representing the mean in the standardization. Default: None.
        std: float
            A float representing the standard deviation in the standardization. Default: None.


        Returns
        -------
        dict
            Predictions corresponding to each input SMILE in order.

        """
        from torch_geometric.loader import DataLoader
        from grape_chem.utils import RevIndexedData
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            out = dict({})
            model.to(device)
            model.eval()
            for smile, i in zip(smiles, range(len(smiles))):
                #try:
                graph = construct_dataset(smiles=[smile],
                                            target=list([0]),
                                            global_features=list([0]),
                                            allowed_atoms=self.allowed_atoms,
                                            atom_feature_list=self.atom_feature_list,
                                            bond_feature_list=self.bond_feature_list)
                # Account for dmpnn models
                graph = RevIndexedData(graph[0])           
                data_temp = next(iter(DataLoader([graph]))).to(device)
                temp = model(data_temp).cpu().detach().numpy()
                if self.mean is not None and self.std is not None:
                    temp = self.rescale(temp, self.mean, self.std)
                elif mean is not None and std is None:
                    temp = self.rescale(temp, mean, std)
                out[smile] = float(temp)
        except:
            print(f'{smiles[i]} is not valid.')

        return out








class GraphDataSet(DataSet):
    """A class that takes a path to a pickle file or a list of smiles and targets. The dataset is stored in
        Pytorch-Geometric Data instances and be accessed like an array. Additionally, if desired it splits the datasets and
        prepares the splits for training and validation. **If you wish to split the datasets immediately, please set
        split to True. Per default, it will not split.**

    Parameters
    ----------
    file_path: str
        The path to a pickle file that should be loaded and the datasets therein used.
    smiles: list[str]
        List of smiles to be made into a graph.
    target: list[float]
        List of target values that will act as the graphs 'y'.
    allowed_atoms: list[str]
        List of allowed atom symbols.
    atom_feature_list: list[str]
        List of features to be applied. Default are the AFP atom features.
    bond_feature_list: list[str]
        List of features that will be applied. Default are the AFP features
    split: bool
        An indicator if the dataset should be split. Only takes effect if nothing else regarding the split is specified
        and will trigger the default split. Default: False
    split_type: str
        Indicates what split should be used. Default: random. The options are:
        [consecutive, random, molecular weight, scaffold, stratified, custom]
    split_frac: list[float]
        Indicates what the split fractions should be. Default: [0.8, 0.1, 0.1]
    custom_split: list
        The custom split that should be applied. Has to be an array matching the length of the filtered smiles,
        where 0 indicates a training sample, 1 a testing sample and 2 a validation sample.
    log: bool
        Decides if the filtering output and other outputs will be shown.
    seed: int
        The numpy seed used to generate the splits. Default: None
    indices: list[int]
        Can be used to override the indices of the datasets objects. Recommended not to use.

    # TODO: Consider removing indices option
    """

    def __init__(self, file_path:str = None, smiles:list[str]=None, target: Union[list[int], list[float],
    ndarray] = None, global_features:Union[list[float], ndarray] = None,
    allowed_atoms:list[str] = None, only_organic: bool = True, atom_feature_list:list[str] = None,
    bond_feature_list:list[str] = None, split: bool = True, split_type:str = None, split_frac:list[float, float, float]
    = None, custom_split: list[int] = None, log: bool = False, seed:int = None, indices:list[int] = None):

        super().__init__(file_path=file_path, smiles=smiles, target=target, global_features=global_features,
                         allowed_atoms=allowed_atoms, only_organic=only_organic,
                         atom_feature_list=atom_feature_list,
                         bond_feature_list=bond_feature_list, log=log, indices=indices)

        if split_type is None:
            split_type = 'random'
        else:
            split = True

        if split_frac is None:
            split_frac = [0.8, 0.1, 0.1]

        self.custom_split = custom_split
        self.split_type = split_type
        self.split_frac = split_frac

        assert np.sum(self.split_frac), 'Split fractions should add to 1.'


        if split or (self.custom_split is not None):
            self.train, self.val, self.test = split_data(data = self, split_type = self.split_type,
                                                        split_frac = self.split_frac, custom_split = self.custom_split,
                                                         seed=seed)
        
def extract_combined_data(config: Dict, encodings=None, limit: int = None):
    if encodings is None:
        encodings = ['latin', 'utf-8', 'utf-8-sig', 'iso-8859-1', 'unicode_escape', 'cp1252']  # Default encodings

    combined_df = pd.DataFrame()
    global_features_names = config.get('global_features', [])
    smiles_column = config['smiles']
    target_column = config['target']
    data_files = config.get('data_files', [config['data_path']])  # Support both single and multiple dataset paths
    data_labels = config.get('data_labels', [''])  # Label is empty if single dataset
    
    for data, label in zip(data_files, data_labels):
        print(f"Processing dataset: {data} with target name: {label}")
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(data, sep=';', encoding=encoding)
                
                # Select relevant columns: smiles, target, and any global features
                columns = [smiles_column, target_column] + global_features_names
                df = df[columns]
                
                if limit is not None:
                    df = df[:limit]  # Limit the dataset size if required
                
                # Drop rows where the target column has NaN values
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                df.dropna(subset=[target_column], inplace=True)
                print(f"Dataframe for {data} loaded with encoding: {encoding}")
                break
            except Exception as e:
                print(f"Failed to read file {data} with encoding {encoding}. Error: {e}")
                continue

        if df is None:
            raise ValueError(f"Failed to read file {data} with any encoding")

        # Rename the target column to distinguish across datasets
        df.rename(columns={target_column: f"{target_column}_{label}"}, inplace=True)
        
        # Merge with combined DataFrame on SMILES using outer join
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on=smiles_column, how='outer')

    # Fill missing target values with NaN and extract global features
    combined_df.fillna(np.nan, inplace=True)
    target_columns = [f"{target_column}_{label}" for label in data_labels]
    targets = combined_df[target_columns].to_numpy()
    smiles = combined_df[smiles_column]
    
    # Standardize target values across datasets
    for target_col in target_columns:
        mean_target, std_target = np.mean(combined_df[target_col]), np.std(combined_df[target_col])
        combined_df[target_col] = standardize(combined_df[target_col], mean_target, std_target)

    # Extract and stack global features if specified
    global_features = None
    if global_features_names:
        global_features = np.hstack([combined_df[feature].to_numpy().reshape(-1, 1) for feature in global_features_names])
    
    return combined_df, smiles, targets, global_features

def extract_data_from_dataframe(config: Dict, encodings = None, limit: int = None):
    if encodings is None:
        encodings = ['latin', 'utf-8', 'utf-8-sig', 'iso-8859-1', 'unicode_escape', 'cp1252']  # Default encodings
    
    df = None
    smiles = None
    target = None
    global_features = None
    global_features_names = None
    
    print("config['data_files']: ", config['data_files'])
    for file in config['data_files']:
        for encoding in encodings:
            try:
                df = pd.read_csv(file, sep=';', encoding=encoding)
                if limit is not None:
                    df = df[:limit]  # Limit to reduce the size of the dataset
                print("########################################################################", df.head())

                # !!!! Drop rows where the target column has NaN values !!!!
                df[config['target']] = pd.to_numeric(df[config['target']], errors='coerce')
                df.dropna(subset=[config['target']], inplace=True)

                # Extract 'SMILES' and 'target' from the cleaned DataFrame
                smiles = df[config['smiles']]
                target = df[config['target']]

                global_features_names = config['global_features']
                print("df set with encoding: ", encoding)
                break  # Break if successful
            except Exception as e:
                print(f"Failed to read file with encoding {encoding}. Error: {e}")
                continue

    if df is None:
        raise ValueError("Failed to read file with any encoding")

    # Extract global features if they exist in the DataFrame
    if global_features_names and len(global_features_names) > 0:
        # Fetch the features from the dataframe
        for feature in global_features_names:
            if feature not in df.columns:
                raise ValueError(f"Global feature '{feature}' not found in the DataFrame")
            else:
                print("Global feature found: ", feature)

            global_feat = df[feature].to_numpy()

            # Reshape global_feat to 2D if it's 1D
            if global_feat.ndim == 1:
                global_feat = global_feat.reshape(1, -1)

            # Initialize features if it's None, else stack the arrays
            if global_features is None:
                global_features = global_feat  # Start with the first global_feat
            else:
                ########### Check this hstack or vstack
                global_features = np.hstack((global_features, global_feat))  # Stack subsequent global_feat
        
    # Check the length of features and convert to the appropriate format
    if global_features is not None and global_features.shape[0] == 1:
        global_features = global_features.flatten()  # Converts from shape (1, 100) to shape (100,)
        
    return df, smiles, target, global_features
    
def extract_data_from_multiple_dataframes(config: Dict, encodings = None, limit: int = None):
    if encodings is None:
        encodings = ['latin', 'utf-8', 'utf-8-sig', 'iso-8859-1', 'unicode_escape', 'cp1252']  # Default encodings
    
    # Create an empty DataFrame to store combined data
    combined_df = pd.DataFrame()

    # Iterate over the dataset files specified in the config
    for index, data in enumerate(config['data_files']): 
        if len(config['data_labels']) - 1 >= index:
            label = config['data_labels'][index]
        else:
            label = 'No target'

        print(f"Processing dataset: {data} with target label: {label}")
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(data, sep=';', encoding=encoding)
                #Drop all columns besides smiles target and global features
                df = df[[config['smiles'], config['target']]]

                if 'global_features' in config and len(config['global_features']) > 0:
                    # Select global features from df and concatenate them to the existing DataFrame
                    df = pd.concat([df, df[config['global_features']]], axis=1)

                #print("Length of dataset: ", len(df))
                print("shape of dataset: ", df.shape)
                #print("Columns of dataset: ", df.columns)
                if limit is not None:
                    df = df[:limit]  # Limit to reduce the size of the dataset
                
                # Drop rows where the target column has NaN values
                df[config['target']] = pd.to_numeric(df[config['target']], errors='coerce')
                df.dropna(subset=[config['target']], inplace=True)
                print("shape of target:", df[config['target']].shape)
                print(f"Dataframe for {data} loaded with encoding: {encoding}")
                break  # Break if successful to avoid trying other encodings
            except Exception as e:
                print(f"Failed to read file {data} with encoding {encoding}. Error: {e}")
                continue

        if df is None:
            raise ValueError(f"Failed to read file {data} with any encoding")
        
        # Ensure the target column is numeric 
        df['Const_Value'] = pd.to_numeric(df['Const_Value'], errors='coerce')
        
        print("Dataframe: ", df.head())
        # Add target name as a suffix for clarity
        df.rename(columns={'Const_Value': f'Const_Value_{label}'}, inplace=True)
        print("After renaming: ", df.head())

        
        # Merge with combined DataFrame using outer join
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on=config['smiles'], how='outer')

    # Apply filtering
    filtered_df = combined_df.groupby(config['smiles']).apply(filter_duplicates).reset_index(drop=True)
    
    # Print the filtered DataFrame
    print("Filtered DataFrame:\n", filtered_df)
    # Fill in missing target values with NaN
    combined_df.fillna(np.nan, inplace=True)
    print("Dataframe: ", combined_df.head())

    target_columns = [f"{config['target']}_{label}" for label in config['data_labels']]
    targets = combined_df[target_columns].to_numpy()
    #print("Targets: ", targets)
    print("Targets dimensions: ", targets.shape)
    smiles = combined_df[config['smiles']]
    global_features = None

    return combined_df, smiles, targets, global_features

def load_dataset_from_excel(file_path: str, dataset:str, is_dmpnn=False, return_dataset:bool = False):
    """ A convenience function to load a dataset from an excel file and a specific sheet there-in. The
    file path given is the path to the excel file and the dataset name given is the sheet name.

    Parameters
    ----------
    file_path: str
        The file path of the excel file.
    dataset: str
        A string that defines what dataset should be used, specifically loaded from a graphs-splits sheet.
        This means, that the sheet name has to correspond to ``dataset``. Options:

        * "Melting Point"

        * "LogP"

        * "Heat capacity"

        * "FreeSolv"
    is_dmpnn: bool
        If graphs for DMPNN has to be loaded. Default: False


    """
    from grape_chem.utils.split_utils import RevIndexedSubSet

    df = pd.read_excel(file_path, sheet_name=dataset)

    data = DataSet(smiles=df.SMILES, target=df.Target, global_features=None, filter=False)

    # convert given labels to a list of numbers and split dataset
    labels = df.Split.apply(lambda x: ['train', 'val', 'test'].index(x)).to_list()

    train_set, val_set, test_set = split_data(data, custom_split=labels)

    # In case graphs for DMPNN has to be loaded:
    if is_dmpnn:
        train_set, val_set, test_set = RevIndexedSubSet(train_set), RevIndexedSubSet(val_set), RevIndexedSubSet(
            test_set)

    if return_dataset:
        return train_set, val_set, test_set, data

    return train_set, val_set, test_set

def load_dataset_from_csv(config: Dict, return_dataset: bool = False, return_df: bool = False, limit: int = None, encodings = None):
    if ('data_files' in config and config['data_files'] is None) or ('data_files' not in config) :
        raise ValueError("File path not provided")
    if ("data_labels" in config and config['data_labels'] is None) or ('data_labels' not in config):
        raise ValueError("Data labels not provided")
    # Extract data from multiple CSV files
    #df, smiles, target, global_features = extract_combined_data(config, encodings, limit)
    if 'data_files' in config and len(config['data_files']) > 1:
        df, smiles, target, global_features = extract_data_from_multiple_dataframes(config, encodings, limit)
    else:    
         # Extract data from the CSV file
         df, smiles, target, global_features = extract_data_from_dataframe(config, encodings, limit)#
    #print("target shape: ", target.shape)

    if isinstance(target, torch.Tensor):
        print("target is a tensor")
        is_one_dimensional = target.dim() == 1
    elif isinstance(target, np.ndarray):
        print("target is a numpy array")
        is_one_dimensional = target.shape[1] == 1
    else:
        is_one_dimensional = False

    if 'dmpnn' in str(config['model_name']).lower():
        is_dmpnn = True
    else:
        is_dmpnn = False
        
    
    # this is solely to get the same exact splits for single and multi target results. Using the same datasets but only a single label. See config
    if is_one_dimensional:
        def ensure_same_splits(split_frac, target, smiles, global_features, config):
            '''Ensure that the same splits are used for single and multi-target datasets
            By splitting the df before the mask is applied, we can ensure that the same indices are used for both single and multi-target datasets
            Args:
                split_frac: list
                target: np.array
                smiles: np.array
                global_features: np.array
                config: dict
            Returns:
                train_indices: np.array
                val_indices: np.array
                test_indices: np.array       
            '''
            assert np.sum(split_frac) == 1, "Split fractions should add up to 1"
            num_samples = len(smiles)
            indices = np.arange(num_samples)
            if config['split_type'] == 'random':
                indices = np.random.permutation(indices)
                target, smiles = target[indices], smiles[indices]
                if global_features is not None:
                    global_features = global_features[indices]

            # Apply the split based on fractions
            train_idx = int(split_frac[0] * num_samples)
            val_idx = int((split_frac[0] + split_frac[1]) * num_samples)
            train_indices, val_indices, test_indices = indices[:train_idx], indices[train_idx:val_idx], indices[val_idx:]

            return train_indices, val_indices, test_indices
        split_frac = [0.8, 0.1, 0.1]
        train_indices, val_indices, test_indices = ensure_same_splits(split_frac, target, smiles, global_features, config)
        train_idx = len(train_indices)
        val_idx = len(train_indices) + len(val_indices)
        train_smiles, val_smiles, test_smiles = smiles[:train_idx], smiles[train_idx:val_idx], smiles[val_idx:]
        train_target, val_target, test_target = target[:train_idx], target[train_idx:val_idx], target[val_idx:]

        if global_features is not None:
            train_global_features, val_global_features, test_global_features = global_features[:train_idx], global_features[train_idx:val_idx], global_features[val_idx:]
        else:
            train_global_features, val_global_features, test_global_features = None, None, None
        ############ We need to standardize BEFORE loading targets into a DataSet ################
        mean, std = np.nanmean(train_target, axis=0), np.nanstd(train_target, axis=0)
        train_target, val_target, test_target = standardize(train_target, mean, std), standardize(val_target, mean, std), standardize(test_target, mean, std)
        
        # Remove the nan values from the target
        train_target = train_target.squeeze()
        val_target = val_target.squeeze()
        test_target = test_target.squeeze()
        
        if global_features is not None:
            train_global_features = train_global_features[~np.isnan(train_target)]
            val_global_features = val_global_features[~np.isnan(val_target)]
            test_global_features = test_global_features[~np.isnan(test_target)]
        
        train_smiles = train_smiles[~np.isnan(train_target)]
        val_smiles = val_smiles[~np.isnan(val_target)]
        test_smiles = test_smiles[~np.isnan(test_target)]
        
        train_target = train_target[~np.isnan(train_target)]
        val_target = val_target[~np.isnan(val_target)]
        test_target = test_target[~np.isnan(test_target)]

        # set indices based on the train, val, test splits after mask
        train_indices = np.arange(len(train_smiles))
        val_indices = np.arange(len(train_smiles), len(train_smiles) + len(val_smiles))
        test_indices = np.arange(len(train_smiles) + len(val_smiles), len(train_smiles) + len(val_smiles) + len(test_smiles))

        target = np.concatenate([train_target, val_target, test_target])
        smiles = np.concatenate([train_smiles, val_smiles, test_smiles])

        if global_features is not None:
            global_features = np.concatenate([train_global_features, val_global_features, test_global_features])
        
        if 'allowed_atoms' in config and config['allowed_atoms'] is not None:
            data = DataSet(smiles=smiles, target=target, global_features=global_features,
                    allowed_atoms=config['allowed_atoms'], 
                    atom_feature_list=config['atom_feature_list'], 
                    bond_feature_list=config['bond_feature_list'], 
                    log=False, only_organic=False, filter=True, allow_dupes=True, mean=mean, std=std)
        else:
            data = DataSet(smiles=smiles, target=target, global_features=global_features,
                        log=False, only_organic=False, filter=True, allow_dupes=True, mean=mean, std=std)
            
        print(" train indices: ", train_indices)
        print(" val indices: ", val_indices)
        print(" test indices: ", test_indices)
        
        train_set, val_set, test_set = SubSet(data, train_indices), SubSet(data, val_indices), SubSet(data, test_indices)
        
    elif not is_one_dimensional:
        if 'allowed_atoms' in config and config['allowed_atoms'] is not None:
            data = DataSet(smiles=smiles, target=target, global_features=global_features,
                    allowed_atoms=config['allowed_atoms'], 
                    atom_feature_list=config['atom_feature_list'], 
                    bond_feature_list=config['bond_feature_list'], 
                    log=False, only_organic=False, filter=True, allow_dupes=True)
        else:
            data = DataSet(smiles=smiles, target=target, global_features=global_features,
                        log=False, only_organic=False, filter=True, allow_dupes=True)
        
        
        # First split to get three separate sets    
        train_set, val_set, test_set = data.split_and_scale(scale = True, split_type=config['split_type'], split_frac=[0.8, 0.1, 0.1], is_dmpnn=is_dmpnn)
        print(data.mean, data.std)
        # Standardize the target values based on the training set
        # data.mean, data.std = np.nanmean(train_set.y, axis=0), np.nanstd(train_set.y, axis=0)
        # data.target = data.scale_array(data.target, data.mean, data.std)
        # train_set.y, val_set.y, test_set.y = data.scale_array(train_set.y, data.mean, data.std), data.scale_array(val_set.y, data.mean, data.std), data.scale_array(test_set.y, data.mean, data.std)

    # In case graphs for DMPNN has to be loaded:
    if is_dmpnn == True:
        from grape_chem.utils.split_utils import RevIndexedSubSet
        train_set, val_set, test_set = RevIndexedSubSet(train_set), RevIndexedSubSet(val_set), RevIndexedSubSet(
            test_set)

    if return_dataset:
        return df, train_set, val_set, test_set, data
    
    return df, train_set, val_set, test_set


##########################################################################
###########  Handle paths ################################################
##########################################################################
def save_splits_to_csv(df, train_set, val_set, test_set, save_folder=None):
    """ Saves the splits of a dataset to a csv file. The dataset is saved in the same folder as the original dataset
    and named after the original dataset with the suffix '_splits'.

    Parameters
    ----------
    df: pd.DataFrame
        The original dataset as a DataFrame.
    train_set: DataSet
        The training set.
    val_set: DataSet
        The validation set.
    test_set: DataSet
        The test set. Not used.
    
    """
    if save_folder is None:
        save_folder = 'C:\\Users\\Thoma\\code\\GraPE\\notebooks'


    df['Split'] = ['train' if i in train_set.indices else 'val' if i in val_set.indices else 'test' for i in range(len(df))]
    path = os.path.join(save_folder, "__splits.csv")
    df.to_csv(path, index=False)
    print(f"Data splits saved to {path}")
    return path



def get_path(*args):
    ''' Cross platform path join '''
    return os.path.normpath(os.path.join(*args))

def get_model_dir(path, model_name):
    ''' Get the model directory based on the model name 
    Args:
        model_name: str: The name of the model
        current_dir: str: The current directory

    Returns:
        str: The model directory
    ''' 
    model_dirs = {
        'afp': get_path(path, '../models', 'AFP'),
        'dmpnn': get_path(path, '../models', 'DMPNN'),
        'original': get_path(path, '../models', 'original_DMPNN'),
    }

    model_type = next((k for k in model_dirs.keys() if k in model_name.lower()), None)
    if not os.path.exists(model_dirs[model_type]):
        os.makedirs(model_dirs[model_type])
    if model_type is None:
        raise ValueError(f"Model name {model_name} not recognized")
    return model_dirs[model_type]
