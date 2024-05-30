# GraPE - Graph-based Property Estimation

A chemical property prediction pipeline.

Download using:
``
git clone https://github.com/aerte/GraPE.git
cd GraPE/python
python setup.py install
``



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



