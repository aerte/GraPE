from rdkit import Chem
import pandas as pd
import numpy as np
#from STOUT import translate_forward
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
import cirpy

def are_equal(s1: str, s2: str) -> bool:
    """" Function that checks whether two SMILES are identical or not since SMILES notation is non-unique
    :param s1: SMILES string 1
    :param s2: SMILES string 2

    :returns: a Boolean indicating whether they are equal or no
    """
    s1_ = (Chem.MolToSmiles(Chem.MolFromSmiles(s1), canonical=True, isomericSmiles=True))
    s2_ = (Chem.MolToSmiles(Chem.MolFromSmiles(s2), canonical=True, isomericSmiles=True))

    return s1_ == s2_


def get_unique(smiles: list) -> list:
    """" Function that returns a unique list of SMILES
    :param smiles: a list of SMILES

    :returns: a list of unique SMILES
    """
    set_smiles = set()

    for s in smiles:
        set_smiles.add(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=True))

    return list(set_smiles)


def ClassifyCompounds(df: pd.DataFrame) -> tuple:
    """Function that classifies compounds into the following classes:
        ['Hydrocarbons', 'Oxygenated', 'Nitrogenated', 'Chlorinated', 'Fluorinated', 'Brominated', 'Iodinated',
        'Phosphorous containing', 'Sulfonated', 'Silicon containing']


    :param df: dataframe, contains property datasets and a column named "SMILES" that contains the smiles of the compounds

    :return: dictionary with the classes and associated index of compounds and dictionary with the summarized lengths
    """
    # Defining the class names
    class_names = ['Hydrocarbons', 'Oxygenated', 'Nitrogenated', 'Chlorinated', 'Fluorinated', 'Brominated',
                   'Iodinated', 'Phosphorous containing', 'Sulfonated', 'Silicon containing']
    # Defining the class tags
    class_patterns = ['C', 'CO', 'CN', 'CCL', "CF", "CBR", "CI", "CP", "CS", "CSI"]

    class_dict = {}
    for i in class_names:
        class_dict[i] = []
    for j, smi in enumerate(df['SMILES']):
        s = ''.join(filter(str.isalpha, smi)).upper()
        for n in range(len(class_names)):
            allowed_char = set(class_patterns[n])
            if set(s) == allowed_char:
                if class_names[n] == 'Chlorinated':
                    if 'CL' in s:
                        class_dict[class_names[n]].append(j)
                elif class_names[n] == 'Brominated':
                    if 'BR' in s:
                        class_dict[class_names[n]].append(j)
                elif class_names[n] == "Silicon containing":
                    if 'SI' in s:
                        class_dict[class_names[n]].append(j)
                else:
                    class_dict[class_names[n]].append(j)

    sum_lst = []
    for key in class_dict:
        sum_lst.extend(class_dict[key])

        # check the consistence
    if len(sum_lst) == len(list(set(sum_lst))):
        multi_lst = list(set(range(len(df))) - set(sum_lst))
        class_dict['Multifunctional'] = multi_lst
        length_dict = {key: len(value) for key, value in class_dict.items()}
    else:
        raise ValueError('The sum is not matching')

    return class_dict, length_dict


def CleanData(df: pd.DataFrame, allowed_atoms: list) -> pd.DataFrame:
    """Function that cleans the datasets from compounds not of interest. Also checks of compound sanity

    :param df: dataframe, contains property datasets and a column named "SMILES" that contains the smiles of the compounds
    :param allowed_atoms: str, list of strings of the atoms allowed to be included e.g. ["C", "c", "O", "N", "Cl", "Br",
     "F", "I", "S", "P", "Si"]
    :return: dataframe with cleaned property datasets namely smiles and target values as a minimum
    """
    # sometimes water ("O") is found in some dataset, as such it must be removed
    df = df.drop(df[df.SMILES == "O"].index)  # is this necessary at this point
    # the compounds should contain at least 1 carbon where C is not the C in Cl
    df = df[df.SMILES.str.replace("Cl", "").str.contains('|'.join(["C", "c"]))]
    # The compound should only allow specific elements/atoms
    df = df[df.SMILES.str.contains('|'.join(allowed_atoms))]
    # check the sanity of the compounds
    idx = []
    for i, smi in enumerate(df.SMILES):
        m = MolFromSmiles(smi)
        if m != None:
            idx.append(i)
        else:
            print(smi + " located at row " + str(i) + " is invalid")
    df = df.iloc[idx]
    # Sorting and resetting the indices
    df.sort_values(by='SMILES', inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def DipprParser(property_name: str) -> pd.DataFrame:
    """Function that loads the dipper property datasets into a pandas dataframe.

    :param property_name: str, the tag of the property
    :return: dataframe with property datasets namely smiles and target values as a minimum
    """
    # set path
    data_from = '../datasets/collection/prop_const/' + property_name + '.xlsx'
    df_raw = pd.read_excel(data_from, sheet_name='dippr', engine='openpyxl')
    # Retain Needed information
    df = df_raw[['Name', 'SMILES', 'Const_Value', 'Data_Type', 'Error']]
    # Remove entries that has NAN in the property values
    df = df.loc[~df.Const_Value.isna()]
    # adjust the error % and find how many unknowns
    df.loc[df.Error == 'Unknown', 'Error'] = '< nan%'
    # splitting
    df['Error'] = df['Error'].str.split('< ', n=1, expand=True)[1]
    df['Error'] = df['Error'].str.split('%', n=0, expand=True)[0]
    # Converting the strings and nan to numerics
    df['Error'] = pd.to_numeric(df['Error'], errors='coerce')
    # The entries with nan for errors are assigned the average error value
    df.loc[pd.isna(df.Error), 'Error'] = df['Error'].mean(skipna=True)
    # A bug that produces unnamed empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Convert type so it is always float64
    df['Const_Value'] = df['Const_Value'].astype(np.float64)
    # Rounding up the errors
    df.Error = df.Error.round(2)
    # Unit conversion procedures to make them homogenous
    if property_name == 'SOLP':
        df['Const_Value'] = df['Const_Value']/1000 # going from Pa1/2 to MPa1/2
    elif property_name == 'PC':
        df['Const_Value'] = df['Const_Value'] / 1e6  # going from Pa to MPa
    elif property_name in ['HFOR', 'HFUS', 'HCOM']:
        df['Const_Value'] = df['Const_Value'] / 1e6  # going from J/kmol to kJ/mol
    elif property_name == 'ENT':
        df['Const_Value'] = df['Const_Value'] / 1e6  # going from J/kmol/K to kJ/mol/K

    return df


def SpeedParser(property_name: str) -> pd.DataFrame:
    """Function that loads the pse4speed (SPEED) property datasets into a pandas dataframe.
    Data also available at: https://github.com/PEESEgroup/Pure-Component-Property-Estimation

    :param property_name: str, the tag of the property
    :return: dataframe with property datasets namely smiles and target values as a minimum
    """
    # set path
    data_from = '../datasets/collection/prop_const/' + property_name + '.xlsx'
    df_raw = pd.read_excel(data_from, sheet_name='speed', engine='openpyxl')
    # Rename the column for name
    df_raw.rename(columns={'Compound': "Name", 'Experimental': "Const_Value"}, inplace=True)
    # Remove entries that has NAN in the property values
    df = df_raw.loc[~df_raw.Const_Value.isna()]
    # A bug that produces unnamed empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Convert type so it is always float64
    df['Const_Value'] = df['Const_Value'].astype(np.float64)

    return df


def PhysPropParser(property_name: str) -> pd.DataFrame:
    """Function that loads the PHYSPROP property datasets into a pandas dataframe.
    Data also available at: https://pubmed.ncbi.nlm.nih.gov/27885862/

    :param property_name: str, the tag of the property
    :return: dataframe with property datasets namely smiles and target values as a minimum
    """
    # set path
    data_from = '../datasets/collection/prop_const/' + property_name + '.xlsx'
    df_raw = pd.read_excel(data_from, sheet_name='physprop', engine='openpyxl')
    # choose corresponding string
    prop_dict = {'LogP': 'Kow',
                 'BioD': 'LogHalfLife',
                 'BCF': 'LogBCF',
                 'LogWs': 'LogMolar'
                 }
    # choose columns of interest
    df_raw = df_raw[['SMILES', 'preferred_name', prop_dict.get(property_name)]]
    # Rename the column for name
    df_raw.rename(columns={'preferred_name': "Name", prop_dict.get(property_name): "Const_Value"}, inplace=True)
    # # Remove entries that has NAN in the property values
    df = df_raw.loc[~df_raw.Const_Value.isna()]
    # # A bug that produces unnamed empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # # Convert type so it is always float64
    df['Const_Value'] = df['Const_Value'].astype(np.float64)

    return df


def CapecParser(property_name: str) -> pd.DataFrame:
    """Function that loads the CAPEC property datasets into a pandas dataframe.

    :param property_name: str, the tag of the property
    :return: dataframe with property datasets namely smiles and target values as a minimum
    """
    # set path
    data_from = '../datasets/collection/prop_const/' + property_name + '.xlsx'
    df_raw = pd.read_excel(data_from, sheet_name='capec', engine='openpyxl')
    # Retain Needed information
    df = df_raw[['Name', 'SMILES', 'Const_Value']]
    # Remove entries that has NAN in the property values
    df = df.loc[~df.Const_Value.isna()]
    # A bug that produces unnamed empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Convert type so it is always float64
    df['Const_Value'] = df['Const_Value'].astype(np.float64)

    return df


def search(data, desired):
    set_desired = list(desired)
    matches = []

    for (s, v) in data:
        if s in set_desired:
            matches.append((s, v))
    return matches


def get_allenes_cummulenes(data):
    desired = []
    s = 'C=C'
    for i in range(100):
        desired.append(s)
        s += '=C'
    return search(data, desired)


def get_n_alkenes(data):
    desired = []
    for i in range(100):
        desired.append('C=C' + i * 'C')
        desired.append(i * 'C' + 'C=C')
    return search(data, desired)


def get_n_alkynes(data):
    desired = []
    for i in range(100):
        desired.append('C#C' + i * 'C')
        desired.append(i * 'C' + 'C#C')
    return search(data, desired)


def get_n_alkanes(data):
    desired = []

    for i in range(2, 100):
        desired.append(i * 'C')

    return search(data, desired)


def get_cyclic_alkanes(data):
    desired = []

    for i in range(1, 100):
        desired.append('C1' + i * 'C' + 'C1')

    return search(data, desired)


def get_benzenes(data):
    desired = []

    for i in range(1, 100):
        desired.append('c1ccccc1' + i * 'C')
        desired.append(i * 'C' + 'c1ccccc1')

    return search(data, desired)


def get_primary_alcohols(data):
    desired = []

    for i in range(2, 100):
        desired.append(i * 'C' + 'O')
        desired.append('O' + i * 'C')

    return search(data, desired)


def get_ketones(data):
    desired = []
    for i in range(100):
        desired.append((i * 'C') + 'C(C)=O')
        desired.append('O=(C)C' + (i * 'C'))
    return search(data, desired)


def get_esters(data):
    desired = []
    for i in range(0, 100):
        desired.append((i * 'C') + 'COC=O')
        desired.append('O=COC' + (i * 'C'))
    return search(data, desired)


def get_ethers(data):
    desired = []
    for i in range(0, 100):
        desired.append((i * 'C') + 'COC')
        desired.append('COC' + (i * 'C'))
    return search(data, desired)


def get_nitriles(data):
    desired = []
    for i in range(0, 100):
        desired.append((i * 'C') + 'C#N')
        desired.append('N#C' + (i * 'C'))
    return search(data, desired)


def get_mercaptans(data):
    desired = []
    for i in range(1, 100):
        desired.append((i * 'C') + 'S')
        desired.append('S' + (i * 'C'))
    return search(data, desired)


def get_carboxylic_acids(data):
    desired = []
    for i in range(0, 100):
        desired.append((i * 'C') + 'COO')
        desired.append('OOC' + (i * 'C'))
    return search(data, desired)


def get_polycyclic_aromatics(data):
    desired = []

    for i in range(2, 9):
        c = 'c1ccc'
        for j in range(2, i):
            c += str(j) + 'cc'
        c += str(i) + 'ccccc' + str(i)
        for j in range(i - 1, 1, -1):
            c += 'cc' + str(j)
        c += 'c1'
        desired.append(c)
    return search(data, desired)


def get_homologuous_series(df: pd.DataFrame) -> pd.DataFrame:
    """Function that takes a datafranme of smiles and target property and returns a dataframe with values for the following homologuous series:
        - Allenes/Cummulenes: Carbon chain with only double bondéd carbons C=C
        - n-Alkenes: Carbon chain with all single bonds ending with a double bonded Carbon C=C
        - n- Alkanes: Carbon chain with only single bonds
        - n-Alkynes: Carbon chain with all single bonds ending with a triple bonded Carbon C#C
        - Cyclic Alkanes: Carbon cycles where all bonds are single bond and all carbons are in the same cycle
        - Benzenes: Carbon chain with all single bonds ending with a benzene ring
        - Primary alcohols: Carbon chain with all single bonds ending with CO
        - Ketones: Carbon chain with all single bonds ending with C(C)=O
        - Esters: Carbon chain with all single bonds ending with COC=O
        - Ethers: Carbon chain with all single bonds ending with COC
        - Nitriles: Carbon chain with all single bonds ending with C#N
        - Mercaptans: Carbon chain with all single bonds ending with CS
        - Carboxylic acids: Carbon chain with all single bonds ending with COO
        - Polycyclic Aromatics: At least two benzen rings connected at two atom creating a straight chain e.g. Naphthalene, Anthracene, Tetracene, Pentacene

    :parms df: dataframe, contains property datasets and a column named "SMILES" that contains the smiles of the compounds and columns Const_Value with the target property

    :returns: df_homo: dataframe containing a column with SMILES, Target_Porperty, Series and index

    """

    # Generate the SMILES from the rdkit algorithm
    rdkit_s = []
    for s in df.SMILES:
        mol = Chem.MolFromSmiles(s)
        rdkit_s.append(Chem.MolToSmiles(mol))

    # Include the new SMILES list to the dataframe
    df['rdkit_SMILES'] = rdkit_s

    # constructing a dataframe with the homologuos series tag as a column

    calls = [
        ('allenes', get_allenes_cummulenes),
        ('n-alkenes', get_n_alkenes),
        ('n-alkanes', get_n_alkanes),
        ('n-alkynes', get_n_alkynes),
        ('cyclic alkanes', get_cyclic_alkanes),
        ('benzenes', get_benzenes),
        ('Primary alcohols', get_primary_alcohols),
        ('ketones', get_ketones),
        ('esters', get_esters),
        ('ethers', get_ethers),
        ('nitriles', get_nitriles),
        ('mercaptans', get_mercaptans),
        ('carboxylic acids', get_carboxylic_acids),
        ('polycyclic aromatics', get_polycyclic_aromatics)]

    # Construct dataframe
    df_homo = pd.DataFrame()

    for homo_tag, homo_get in calls:
        a = homo_get(df[['rdkit_SMILES', 'Const_Value']].values)
        df_new = pd.DataFrame(a, columns=['SMILES', 'Const_Value'])
        df_new['series'] = homo_tag
        c_count = []
        for smi in list(df_new.SMILES):
            c_count.append(smi.count('C') + smi.count('c'))
        df_new['c_count'] = c_count
        df_new.sort_values(['series', 'c_count'], ignore_index=True, inplace=True)
        df_new = df_new[['SMILES', 'Const_Value', 'c_count', 'series']]
        df_homo = pd.concat([df_homo, df_new], axis=0, join='outer', ignore_index=True)

    name_list = get_iupac_name(df_homo.SMILES)
    df_homo['iupac_name'] = name_list

    return df_homo


def get_iupac_name(smiles: list) -> list:
    """ Function that retrieves the iupac name of a compound using STOUT:
    https://github.com/Kohulan/Smiles-TO-iUpac-Translator

    :param smiles: list of SMILES
    :return: list of iupac names
    """
    name_list = []
    for smi in smiles:
        #name_list.append(translate_forward(smi))
        name_list.append(cirpy.resolve(smi, 'iupac_name'))
    return name_list


def get_unit(property_name: str) -> str:
    """ Function that returns the unit of a property

    :param property_name: the name of the property
    :return: a str of the unit
    """

    # build dictionary for the units for each dataset
    unit_dict = {
        "BP": "[K]",
        "MP": "[K]",
        "LogP": "",
        "DSOLP": "[MPa¹/²]",
        "PSOLP": "[MPa¹/²]",
        "HSOLP": "[MPa¹/²]",
        "SOLP": "[MPa¹/²]",
        "LogWs": "[mol frac.]",
        "OMEGA": "",
        "TC": "[K]",
        "PC": "[MPa]",
        "VC": "[m³\u2022kmol\u207b¹]",
        "HVAP298": "[kJ\u2022mol\u207b¹]",
        "HFUS": "[kJ\u2022mol\u207b¹]",
        "HFOR": "[kJ\u2022mol\u207b¹]",
        "LVOL": "[m³\u2022kmol\u207b¹]",
        "ENT": "[kJ\u2022mol\u207b¹\u2022K\u207b¹]",
        "RI": "",
        "HCOM": "[kJ\u2022mol\u207b¹]",
        "AIT": "[K]",
        "FLVL": "[vol % in air]",
        "FLVU": "[vol % in air]",
        "FP": "[K]",
        "BCF": "",
        "PCO": "",
        "pka": "",
        "LD50": "[mol/l]",
        "LC50": "[mol/l]",
        "OSHA_TWA": "[mol/m³]",
        "BioD": "",
    }
    return unit_dict.get(property_name)


def get_name(property_name: str) -> str:
    """ Function that returns the unit of a property

    :param property_name: the name of the property
    :return: a str of the unit
    """

    # build dictionary for the units for each dataset
    name_dict = {
        "BP": r'Boiling point, $\mathrm{\mathbf{T_b}}$',
        "MP": r'Melting point, $\mathrm{\mathbf{T_m}}$',
        "LogP": r'Octanol/water partition coef. , $\mathrm{\mathbf{logK_{OW}}}$',
        "DSOLP": r'Hansen D-solubility parameter, $\mathrm{\mathbf{\delta_{D}}}$',
        "PSOLP": r'Hansen P-solubility parameter, $\mathrm{\mathbf{\delta_{P}}}$',
        "HSOLP": r'Hansen H-solubility parameter, $\mathrm{\mathbf{\delta_{H}}}$',
        "SOLP": r'Hildebrand solubility parameter, $\mathrm{\mathbf{\delta}}$',
        "LogWs": r'Water solubility, $\mathrm{\mathbf{logWs}}$',
        "OMEGA": r'Acentric factor, $\mathrm{\mathbf{\omega}}$',
        "TC": r'Critical temperature, $\mathrm{\mathbf{T_c}}$',
        "PC": r'Critical pressure, $\mathrm{\mathbf{P_c}}$',
        "VC": r'Critical volume, $\mathrm{\mathbf{V_c}}$',
        "HVAP298": r'Enthalpy of vaporization, $\mathrm{\mathbf{H_{vap}^{298K}}}}$',
        "HFUS": r'Enthalpy of fusion, $\mathrm{\mathbf{H_{fus}}}}$',
        "HFOR": r'Enthalpy of formation, $\mathrm{\mathbf{H_{for}}}}$',
        "LVOL": r'Liquid molar volume, $\mathrm{\mathbf{V_{m}}}}$',
        "ENT": r'Absolute entropy, $\mathrm{\mathbf{S}}}$',
        "RI": r'Refractive index, $\mathrm{\mathbf{RI}}}$',
        "HCOM": r'Enthalpy of combustion, $\mathrm{\mathbf{H_{com}}}}$',
        "AIT": r'Auto-ignition temperature, $\mathrm{\mathbf{AIT}}}$',
        "FLVL": r'Lower flammability limit, $\mathrm{\mathbf{LFL}}}$',
        "FLVU": r'Upper flammability limit, $\mathrm{\mathbf{UFL}}}$',
        "FP": r'Flash point temperature, $\mathrm{\mathbf{FP}}}$',
        "BCF": r'Bioconcentration factor, $\mathrm{\mathbf{BCF}}}$',
        "PCO": r'Photochemical oxidation, $\mathrm{\mathbf{PCO}}}$',
        "pka": r'Acid dissociation constant, $\mathrm{\mathbf{pka}}}$',
        "LD50": r'Lethal dosage, $\mathrm{\mathbf{LD_{50}}}}$',
        "LC50": r'Lethal concentration, $\mathrm{\mathbf{LC_{50}}}}$',
        "OSHA_TWA": r'Premissible exposure limit, $\mathrm{\mathbf{OSHA-TWA}}}$',
        "BioD": r'Biodegradability, $\mathrm{\mathbf{BioD}}}$',
    }

    return name_dict.get(property_name)



# checks
assert (are_equal('CCCCN', 'CCCCN') == True)
assert (are_equal('NCCCCN', 'CCCCN') == False)
assert (len(get_unique(['CC', 'CC', 'CCC', 'CCCC'])) == 3)
