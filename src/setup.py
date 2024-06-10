import os
from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='grape_chem',
    version='1.0.1',
    author='Felix Ærtebjerg',
    description='Tools for computational chemistry and deep learning.',
    packages=['grape_chem'],  # List of packages to be included
    url='https://github.com/aerte/GraPE',
    #packages = [package for packages in find_packages()
    #          if package.startswith('grape_chem')],
    install_requires=[
        'torch == 2.1.0',
        'torch_geometric >= 2.1',
        'torch-scatter == 2.1.2',
        'rdkit~=2023.9.5',
        'dgl~=2.0.0',
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
    long_description=long_description,
    long_description_content_type='text/markdown',
)
