import os
from setuptools import setup, find_packages

#Install grape_chem in edit mode in order to import grape_chem saved in src folder in branch
#in src folder: pip install -e . --no-deps --config-settings editable_mode=compat -v --use-pep517 --config-settings=setup-file=setup_grape_src.py

setup(
    name='grape_chem',
    version='1.0.3',
    author='Felix Ærtebjerg',
    description='Tools for computational chemistry and deep learning.',
    packages=find_packages(where="src"),  # This line is important
    package_dir={"": "src"},  # This line tells setuptools where to find the packages
    url='https://github.com/aerte/GraPE',
    install_requires=[
        'torch == 2.1.0',
        'torch_geometric >= 2.1',
        'torch-scatter',
        'rdkit~=2023.9.5',
        'dgl~=1.1.3',
        'pandas~=2.2.0',
        'numpy~=1.26.3',
        'pyarrow',
        'openpyxl',
        'dgllife~=0.3.2',
        'matplotlib~=3.8.3',
        'seaborn~=0.13.2',
        'CIRpy~=1.0.2',
        'tqdm~=4.66.1',
        'scikit-learn'
    ],
)