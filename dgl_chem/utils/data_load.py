import pandas as pd
from rdkit import Chem

__all__ = ['filter_smiles']

def filter_smiles(smiles, allowed_set = None):

    if allowed_set is None:
        allowed_set = ['C','O']

    df = pd.DataFrame(smiles, columns=['smiles'])
    indices_to_drop = []

    for element in smiles:
        mol = Chem.MolFromSmiles(element)

        if mol is None:
            print(f'SMILES {element} in index {list(df.smiles).index(element)} is not valid.')
            indices_to_drop.append(list(df.smiles).index(element))

        else:
            if mol.GetNumHeavyAtoms() < 2:
                print(f'SMILES {element} in index {list(df.smiles).index(element)} consists of less than 2 heavy atoms'
                      f' and will be ignored.')
                indices_to_drop.append(list(df.smiles).index(element))

            else:
                for atoms in mol.GetAtoms():
                    if atoms.GetSymbol() not in allowed_set:
                        print(f'SMILES {element} in index {list(df.smiles).index(element)} contains the atom {atoms.GetSymbol()} that is not'
                              f' permitted and will be ignored.')
                        indices_to_drop.append(list(df.smiles).index(element))

    df.drop(indices_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return list(df.smiles)