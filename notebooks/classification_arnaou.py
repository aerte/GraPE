import time

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import json
import os
import urllib.request
import math

# read the data
df_raw = pd.read_excel('BradleyDoubleGood.xlsx')

# convert J to kJ
df_raw['mpC'] = df_raw['mpC']

# average measurements if multiple temperature entries are present for the same compound
df_mean = df_raw.groupby(['SMILES'])['mpC'].mean().reset_index()
#%%
# find how many entries they have
mol_occ = df_mean.groupby(by='SMILES')['SMILES'].count()

# drop compounds with less than 5 measurements
df_clean = df_mean#df_mean[~df_mean['SMILES'].isin(list(mol_occ[mol_occ < 5].index))].copy()

# get list of unique smiles
smi_list = df_clean['SMILES'].unique()

# construct data frame for summary
df_summary = pd.DataFrame(columns=['SMILES', 'inchikey', 'class'])

# get Inchikey
mol_list = []
for smi in smi_list:
    try:
        mol_list.append(Chem.MolFromSmiles(smi))
    except:
        mol_list.append('')

print(mol_list)

inchikey_rdkit = []
for mol in mol_list:
    try:
        inchikey_rdkit.append(Chem.inchi.MolToInchiKey(mol))
    except:
        inchikey_rdkit.append('')

# download classification using inchikey
path_folder = 'classyfire'
if not os.path.exists(path_folder):
    os.makedirs(path_folder)

missing_keys = False
path_report = 'missing_keys.txt'
report = open(path_report, 'w')


def print_report(string, file=report):
    file.write('\n' + string)


for i in tqdm(range(len(inchikey_rdkit[:100]))):
    key = inchikey_rdkit[i]
    url = 'https://cfb.fiehnlab.ucdavis.edu/entities/'+str(key)+'.json'
    try:
        with urllib.request.urlopen(url) as webpage:
            data = json.loads(webpage.read().decode())

        with open(path_folder + '/' + str(i) + '.json', 'w') as f:
            json.dump(data, f)
    except:
        print(f'Was not able to access the webpage at step {i}')
        print_report(str(i) + '    ' + str(key))
        missing_keys = True
        pass

    time.sleep(math.ceil(len(inchikey_rdkit)/12/60))

report.close()

if missing_keys:
    print('Some InChikeys were not available. Please check "Missing_ichikeys.txt" file.')
else:
    os.remove(path_report)
