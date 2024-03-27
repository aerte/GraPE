import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
__all__ = ['filter_smiles']

def filter_smiles(smiles, allowed_set = None, print_out = False):
    ''' Filters a list of smiles.

    Args
    ----------
    smiles: list of str
        Smiles to be filtered.
    allowed_set: list of str
        Valid atom symbols, non-valid symbols will be discarded. Default: [``B``, ``C``, ``N``, ``O``,
            ``F``, ``Si``, ``P``, ``S``, ``Cl``, ``As``, ``Se``, ``Br``, ``Te``, ``I``, ``At``]
    print_out: bool
        Determines if there should be print-out statements to indicate why mols were filtered out. Default: False

    Returns
    ----------
    list[str]
        A list of filtered smiles strings.

    '''

    if allowed_set is None:
        allowed_set = ['B', 'C', 'N', 'O','F', 'Si', 'P',
                       'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']

    df = pd.DataFrame(smiles, columns=['smiles'])
    indices_to_drop = []

    for element in smiles:
        mol = Chem.MolFromSmiles(element)

        if mol is None:
            if print_out:
                print(f'SMILES {element} in index {list(df.smiles).index(element)} is not valid.')
            indices_to_drop.append(list(df.smiles).index(element))

        else:
            if mol.GetNumHeavyAtoms() < 2:
                if print_out:
                    print(f'SMILES {element} in index {list(df.smiles).index(element)} consists of less than 2 heavy atoms'
                        f' and will be ignored.')
                indices_to_drop.append(list(df.smiles).index(element))

            else:
                for atoms in mol.GetAtoms():
                    if atoms.GetSymbol() not in allowed_set:
                        if print_out:
                            print(f'SMILES {element} in index {list(df.smiles).index(element)} contains the atom {atoms.GetSymbol()} that is not'
                                f' permitted and will be ignored.')
                        indices_to_drop.append(list(df.smiles).index(element))

    df.drop(indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return list(df.smiles)