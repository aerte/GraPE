Graph Dataset Classes
======================

The implemented Graph Dataset classes ``DataSet`` and ``GraphDataSet`` serve
as quick molecule dataset loading and easy management helpers. Given a set of
SMILES and a target of same length, will *filter* the SMILES and *create* the PyTorch
Geometric ``Data`` objects (molecule graphs).

Other features include the fast loading of a saved ``DataSet``, easy saving and some
data analysis features.

.. automodule:: grape.utils.data
   :members:
   :undoc-members:
   :show-inheritance: