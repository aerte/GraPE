# Consists of all the rdkit molecule function that will be used by the featurizers.
# Inspired by https://github.com/awslabs/dgl-lifesci

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

__all__ = [
    'one_hot',
    'mol_weight',
    'atom_symbol',
    'atom_degree',
    'atom_chiral',
    'atom_valence',
    'atom_hybridization',
    'atom_in_ring',
    'atom_num_H',
    'atom_num_rad_electrons',
    'atom_mass',
    'atom_is_aromatic',
    'atom_formal_charge',
    'bond_direction',
    'bond_type',
    'bond_stereo',
    'bond_is_conjugated',
    'bond_in_ring'
]

def one_hot(input_, mapping, encode_unknown = True):
    """One hot encodes an arbitrary input given a mapping.

    Parameters
    ------------
    input_: Any
        The input value that will be encoded. Could be an integer, a string or rdkit bond type.
    mapping: list
        A list of values that the input will be encoding to.
    encode_unknown: bool
        Decides if, should the input not be represented in the mapping, the unknowns input is encoded as 'other' at
        the end. Default: False

    Returns
    ---------
    list
        One hot encoding.

    """

    if isinstance(input_, list):
        input_ = input_[0]

    encoding = list(map(lambda x: x == input_, mapping))

    if encode_unknown and (input_ not in mapping):
        encoding.append(True)
    elif encode_unknown:
        encoding.append(False)

    assert np.sum(np.array(encoding)) == 1, ('One of the elements has to be True, consider checking the values'
                                             'or encoding the unknowns.')

    return encoding

#############################################################################################################
#################################### MOLECULE FEATURES ######################################################
#############################################################################################################

def mol_weight(mol):
    """Returns the molecular weight of a smile.

    Parameters
    ------------
    mol: rdkit.Chem.rdkit.Mol
        RDKit molecule object.


    Returns
    ---------
    float
        The molecular weight of the SMILES molecule.

    """

    return Descriptors.ExactMolWt(mol)


#############################################################################################################
###################################### ATOM FEATURES ########################################################
#############################################################################################################

#### Atom Symbol
def atom_symbol(atom, allowed_set = None, encode_unknown=True):

    allowed_set =  ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P'] if allowed_set is None else allowed_set
    return one_hot(atom.GetSymbol(), allowed_set, encode_unknown)

def atom_number(atom, type_='', allowed_set = None, encode_unknown=False):

    assert type_ in ['one_hot', 'regular', ''], 'Wrong type, the options are: [one_hot, regular, ]'

    
    if type_ == 'one_hot':
        allowed_set = list(range(1,101)) if allowed_set is None else allowed_set
        return one_hot(atom.GetAtomicNum(), allowed_set, encode_unknown)
    elif type_ == 'regular':
        return [atom.GetAtomicNum()]
    else:
        return [atom.GetAtomicNum()]


#### Atom degree
def atom_degree(atom, type_='', allowed_set=None, encode_unknown=None):

    assert type_ in ['total_one_hot', 'total', 'one_hot', 'regular', ''], ('Wrong type, the options are:'
                                                                           '[total_one_hot, total, one_hot, regular, ]')

    if type_ == 'total_one_hot':
        allowed_set = list(range(6)) if allowed_set is None else allowed_set
        return one_hot(atom.GetTotalDegree(), allowed_set, encode_unknown)
    elif type_ ==  'total':
        return [atom.GetTotalDegree()]
    elif type_ ==   'one_hot':
        allowed_set = list(range(6)) if allowed_set is None else allowed_set
        return one_hot(atom.GetDegree(), allowed_set, encode_unknown)
    elif type_ ==   'regular':
        return [atom.GetDegree()]
    else:
        return [atom.GetDegree()]


#### Atom valency
def atom_valence(atom, type_='', allowed_set=None, encode_unknown=False):
    assert type_ in ['ex_one_hot', 'ex', 'im_one_hot', 'im', ''], ('Wrong type, the options are: '
                                                                    '[ex_one_hot, ex, im_one_hot, im, ]')

    
    if type_ == 'ex_one_hot':
        allowed_set = list(range(0, 6)) if allowed_set is None else allowed_set
        return one_hot(atom.GetExplicitValence(), allowed_set, encode_unknown)
    elif type_ == 'ex':
        return [atom.GetExplicitValence()]
    if type_ == 'im_one_hot':
        allowed_set = list(range(0, 6)) if allowed_set is None else allowed_set
        return one_hot(atom.GetImplicitValence(), allowed_set, encode_unknown)
    if type_ == 'im':
        return [atom.GetImplicitValence()]
    else:
        return [atom.GetImplicitValence()]


#### Atom hybridization
def atom_hybridization(atom, allowed_set=None, encode_unknown=True):

    if allowed_set is None:
        allowed_set = [Chem.rdchem.HybridizationType.SP,
                       Chem.rdchem.HybridizationType.SP2,
                       Chem.rdchem.HybridizationType.SP3,
                       Chem.rdchem.HybridizationType.SP3D,
                       Chem.rdchem.HybridizationType.SP3D2]
    return one_hot(atom.GetHybridization(), allowed_set, encode_unknown)


#### Total number of H of an Atom
def atom_num_H(atom, type_='', allowed_set=None, encode_unknown=False):

    assert type_ in ['one_hot', 'regular', ''], 'Wrong type, the options are: [one_hot, regular, ]'

    
    if type_ == 'one_hot':
        allowed_set = list(range(5)) if allowed_set is None else allowed_set
        return one_hot(atom.GetTotalNumHs(), allowed_set, encode_unknown)
    elif type_ == 'regular':
        return [atom.GetTotalNumHs()]
    else:
        return [atom.GetTotalNumHs()]

#### Formal charge
def atom_formal_charge(atom, type_='', allowed_set=None, encode_unknown=False):

    assert type_ in ['one_hot', 'regular', ''], 'Wrong type, the options are: [one_hot, regular, ]'

    if type_ == 'one_hot':
        allowed_set = list(range(-2,3)) if allowed_set is None else allowed_set
        return one_hot(atom.GetFormalCharge(), allowed_set, encode_unknown)
    elif type_ == 'regular':
        return [atom.GetFormalCharge()]
    else:
        return [atom.GetFormalCharge()]

#### Radical electrons
def atom_num_rad_electrons(atom, type_='', allowed_set=None, encode_unknown=False):

    assert type_ in ['one_hot', 'regular', ''], 'Wrong type, the options are: [one_hot, regular, ]'

    if type_ == 'one_hot':
        allowed_set = list(range(5)) if allowed_set is None else allowed_set
        return one_hot(atom.GetNumRadicalElectrons(), allowed_set, encode_unknown)
    elif type_ == 'regular':
        return [atom.GetNumRadicalElectrons()]
    else:
        return [atom.GetNumRadicalElectrons()]


def atom_is_aromatic(atom):

    return [atom.GetIsAromatic()]


def atom_in_ring(atom):

    return [atom.IsInRing()]


def atom_chiral(atom, type_='', allowed_set=None, encode_unknown=False):

    assert type_ in ['tag', 'type', 'center'], 'Wrong type, the options are: [tag, type, center]'

    if type_ == 'tag':
        if allowed_set is None:
            allowed_set = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                            Chem.rdchem.ChiralType.CHI_OTHER]
        return one_hot(atom.GetChiralTag(), allowed_set, encode_unknown)
    
    elif type_ == 'type':
        if not atom.HasProp('_CIPCode'):
            return [False, False]

        allowed_set = ['R', 'S'] if allowed_set is None else allowed_set
        return one_hot(atom.GetProp('_CIPCode'), allowed_set, encode_unknown)
    
    elif type_ == 'center':
        return [atom.HasProp('_ChiralityPossible')]

def atom_mass(atom, scale=0.01):

    return [atom.GetMass()*scale]


#############################################################################################################
###################################### BOND FEATURES ########################################################
#############################################################################################################

def bond_type(bond, allowed_set=None, encode_unknown=False):

    if allowed_set is None:
        allowed_set = [Chem.rdchem.BondType.SINGLE,
                       Chem.rdchem.BondType.DOUBLE,
                       Chem.rdchem.BondType.TRIPLE,
                       Chem.rdchem.BondType.AROMATIC]

    return one_hot(bond.GetBondType(), allowed_set, encode_unknown)

def bond_is_conjugated(bond):

    return [bond.GetIsConjugated()]


def bond_in_ring(bond):
    return [bond.IsInRing()]

def bond_stereo(bond, allowed_set=None, encode_unknown=False):

    if allowed_set is None:
        allowed_set = [Chem.rdchem.BondStereo.STEREONONE,
                       Chem.rdchem.BondStereo.STEREOANY,
                       Chem.rdchem.BondStereo.STEREOE,
                       Chem.rdchem.BondStereo.STEREOZ,
                       Chem.rdchem.BondStereo.STEREOCIS,
                       Chem.rdchem.BondStereo.STEREOTRANS]
    return one_hot(bond.GetStereo(), allowed_set, encode_unknown)

def bond_direction(bond, allowed_set=None, encode_unknown=False):

    if allowed_set is None:
        allowed_set = [Chem.rdchem.BondDir.NONE,
                       Chem.rdchem.BondDir.ENDUPRIGHT,
                       Chem.rdchem.BondDir.ENDDOWNRIGHT]

    return one_hot(bond.GetBondDir(), allowed_set, encode_unknown)