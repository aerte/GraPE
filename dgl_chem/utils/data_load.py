import pandas as pd
import re

def filter_smiles(smiles, allowed_set = None):

    # Should probably be importing the allowed set from somewhere
    if allowed_set is None:
        allowed_set = ['C','O']

    df = pd.DataFrame(smiles)

    list_elements = []

    for mol in df.smiles:
        al = re.findall('[A-Z][a-z]?', mol)
        for a in range(len(al)):
            al[a] = re.sub('[A-Z]c', '', al[a])
            al[a] = re.sub('[A-Z]n', '', al[a])
            if not al[a] in list_elements and not al[a] == '':
                list_elements.append(al[a])

    print(list_elements)

    for element in list_elements:
        if element not in allowed_set:
            df.drop(df[df.smiles.str.contains(element)].index, inplace=True)
            df.reset_index(drop=True, inplace=True)

    return df