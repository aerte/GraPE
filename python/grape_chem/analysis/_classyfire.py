import os
import time
import json
import urllib.request
import math

from tqdm import tqdm
import pandas as pd
from rdkit import Chem

__all__ = [
    "classyfire",
    "classyfire_result_analysis",
]

def print_report(string, file=None):
    file.write('\n' + string)

def file_in_dir(directory, filename):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return True


def classyfire(smiles: list[str], path_to_export: str = None,
               record_log_file: bool = True, existing_log_file: str = None,
                log: bool = True) -> tuple[list[int], list[int]]:
    """Applies the classyfire procedure given in [1] to the input smiles and generates json files with the
    information. It can also generate a csv file recording the names of the json files and the corresponding SMILES to
    avoid retrieving the same information multiple times. The procedure will take about ~10min for 300 molecules. For
    large datasets, consider subsampling the smiles or using the less informative compound classifier.

    Parameters
    ----------
    smiles: list[str]
        The input smiles, their information will be attempted to retrieve from the classyfire database.
    path_to_export: str
        Path where the classyfire results should be stored.
    record_log_file: bool
        Decides if a log file should be created. Will check if an existing log file is given and use it if True.
        Default: True
    existing_log_file: str
        The path to a log file recording the json file names and the corresponding molecule Will be overridden with by
        a new file if record_lof_file is set to True. The existing file have to be saved with 'filename' and 'smiles'
        columns. Default: None
    log: bool
        Prints out issues in the classyfire procedure, fx. a failed attempt at retrieving a molecule's information.
        Default: True

    Returns
    -------
    tuple[list[int], list[int]]
        The indices of the SMILES where graphs was successfully retrieved. Could be used for an output analysis. As well
        as the indices of all the smiles that could be retrieved *relative to the input list*.

    References
    ----------
    [1] Djoumbou Feunang et al., ClassyFire: automated chemical classification with a comprehensive, computable taxonomy.,
    2016, https://doi.org/10.1186/s13321-016-0174-y

    """
    index_in = range(len(smiles))
    mols = list(map(lambda x: Chem.MolFromSmiles(x), smiles))
    standard_smiles = list(map(lambda  x: Chem.MolToSmiles(x), mols))
    ids = []

    if path_to_export is None:

        path_to_export = os.getcwd() + '/analysis_results'

        if not os.path.exists(path_to_export):
            os.mkdir(path_to_export)

        if existing_log_file is None:
            if file_in_dir(path_to_export, 'recorded_SMILES.csv'):
                print('Found log file in working directory.')
                existing_log_file = os.path.join(path_to_export,'recorded_SMILES.csv')

    if record_log_file and existing_log_file is None:
        log_file = os.path.join(path_to_export, 'recorded_SMILES.csv')
        log_frame = pd.DataFrame({'filename':[], 'smiles':[]})

    elif existing_log_file is not None:
        try:
            log_frame = pd.read_csv(existing_log_file)
            log_file = existing_log_file

            if log:
                print('Loaded existing log_file. Here is the graphs head:')
                print(log_frame.head(10))
        except:
            print('Passed log_file is not valid, creating a new one.')
            log_file = os.path.join(path_to_export, 'recorded_SMILES.csv')
            log_frame = pd.DataFrame({'filename': [], 'smiles': []})

    ids_out = []
    input_output_ids = []
    index_test = []

    inchikey_rdkit = []
    existing_log = 0
    for idx, mol in enumerate(mols):
        #print(standard_smiles[idx])
        if standard_smiles[idx] in log_frame.smiles.values:
            existing_log += 1
            ids_out.append(list(log_frame.smiles.values).index(standard_smiles[idx]))
            input_output_ids.append(idx)
            if log:
                print(f'Mol {idx} in log datafile, so skipping it.')
            continue
        try:
            inchikey_rdkit.append(Chem.inchi.MolToInchiKey(mol))
            ids.append(idx)
            index_test.append(idx)

        except:
            if log:
                print(f'No inchikey generated from SMILE index {idx}, possibly due to a faulty SMILE.')
                del standard_smiles[idx]
            inchikey_rdkit.append('')

    if existing_log == len(standard_smiles):
        print('All passed smiles are already in the passed log_file.')
        return ids_out, input_output_ids

    # download classification using inchikey
    path_folder = os.path.join(path_to_export,'classyfire')
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    # Output report
    missing_keys = False
    path_report = os.path.join(path_to_export,'missing_keys.txt')
    report = open(path_report, 'w')

    if len(log_frame) == 0:
        max_list = 0
    else:
        max_list = max(log_frame.index)

    # Generating the classyfire files
    for i in tqdm(range(len(inchikey_rdkit))):
        key = inchikey_rdkit[i]
        idx = ids[i]
        url = 'https://cfb.fiehnlab.ucdavis.edu/entities/' + str(key) + '.json'
        try:
            with urllib.request.urlopen(url) as webpage:
                data = json.loads(webpage.read().decode())

            filename = str(max_list+i) + '.json'
            with open(path_folder + '/' + filename, 'w') as f:
                json.dump(data, f)
            log_frame = pd.concat([log_frame, pd.DataFrame({'filename': [filename], 'smiles': [standard_smiles[idx]]})])
            ids_out.append(log_frame.smiles.values.tolist().index(standard_smiles[idx]))
            input_output_ids.append(idx)

        except:
            if log:
                print(f'Failure to retrieve information for SMILE at index {idx}')
            if key is None:
                print_report(str(i) + '    ' + 'NULL KEY', file=report)
            else:
                print_report(str(i) + '    ' + str(key), file=report)
            missing_keys = True
            pass


        time.sleep(math.ceil(len(inchikey_rdkit) / 12 / 60))

    report.close()

    if missing_keys:
        print('Some InChikeys were not available. Please check "Missing_ichikeys.txt" file.')
    else:
        os.remove(path_report)

    log_frame.to_csv(log_file,index=False)

    return ids_out, input_output_ids




def classyfire_result_analysis(path_to_classyfire: str = None, idx: list[int] = None,
                               layer:int = 1, return_relative_ids:bool = False) -> tuple[dict,dict]:
    """Uses the json files generated through the classyfire procedure to perform a 1st layer graphs analysis. It will
    return two dictionaries, one with the molecule class and the corresponding id, and the other with the class
    frequencies. It is assumed that one molecule corresponds to one json file.

    Parameters
    ----------
    path_to_classyfire: str
        The path to the directory containing the json files generated from the classyfire procedure. By default, it will
        assume that the json files are located in the working directory under '/analysis_results/classyfire'
    idx: list[int]
        Optional input to specify the SMILE indices used for performing the classyfire. Default: None
    layer: int
        Decides the layer of information accessed, see [1] for more information. The layers are: ``0``- Kingdom,
        ``1``- Super-Class, ``2``- Class. Default: 1.
    return_relative_ids: bool
        If true, will return the relative ids of the json files that worked in the same order as
        the ids were passed. Default: False

    Returns
    -------
    mols_class, class_freq, working_ids: tuple[dict,dict, list]
        The molecule-class, class frequency dictionaries and (optionally) a list over the indices of the json file
        indices that worked respectively.

    References
    ----------
    [1] Djoumbou Feunang et al., ClassyFire: automated chemical classification with a comprehensive, computable taxonomy.,
    2016, https://doi.org/10.1186/s13321-016-0174-y

    """

    if path_to_classyfire is None:
        path_to_classyfire = os.getcwd() + '/analysis_results' + '/classyfire'
    if idx is None:
        idx = range(len(os.listdir(path_to_classyfire)))

    if layer == 0:
        layer_name = 'kingdom'
    elif layer == 1:
        layer_name = 'superclass'
    elif layer == 2:
        layer_name = 'class'

    class_freq = dict()
    mols_class = dict()
    working_ids = []

    for id_mol, file, i in zip(idx, os.listdir(path_to_classyfire), range(len(idx))):
        file_path = os.path.join(path_to_classyfire, file)

        try:
            class_name = json.load(open(file_path))[layer_name]['name']
            if class_name is None:
                print(f'No class name in the first layer for file: {file} and index: {id_mol}')
                class_name = json.load(open(file_path))['alternative_parents']['name']

            if class_name in class_freq.keys():
                class_freq[class_name] += 1
            else:
                class_freq[class_name] = 1
            mols_class[id_mol] = class_name

            working_ids.append(i)

        except:
            print(f'Key error occurred using {layer_name} for file {file}.')
    if return_relative_ids:
        return mols_class, class_freq, working_ids
    return mols_class, class_freq