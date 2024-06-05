GraPE-Chem - Graph-based Property Estimation for Chemistry
===========================================================

This is a python package to support Chemical property prediction using [PyTorch](https://pytorch.org/docs/stable/index)
and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).
The ambition of this project is to build a flexible pipeline that lets users go from molecule
descriptors (SMILES) to a fully functioning Graph Neural Network and allow for useful customization
at every step.

For more information, please check out the [docs](https://grape-chem.readthedocs.io/en/latest/).



Installing the toolbox
----------------------

To use the package, please run the following inside a terminal:

``pip install grape-chem``


Demonstrations and Use
-----------------------
After installing, the package will work like any other. See ``Demo``
and ``Advanced Demo`` inside of [docs](https://grape-chem.readthedocs.io/en/latest/) 
for an introduction of how the toolbox can be used.



Note
-----
If optimization is run on hpc using `GraPE` and the optimization procedure outlined in
the ``Advanced Demonstration``, the following requirements need to be met:

``
python==3.9
``
``
cuda==12.1
``

and the following package need to be re-installed using the correct cuda-version:

``
torch==2.1.2
``
``
dgl~=1.1.3
``
``
torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
``
``
ray
``
``
ConfigSpace==0.4.18
``
``
hpbandster==0.7.4
``

The reason for the particular python version is a subpackage in ``hpbandster``.



