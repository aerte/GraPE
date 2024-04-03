Datasets
==================

Inspired by the PyTorch dataset like MNIST and CIFAR, we implemented a few common
chemical properties datasets that use the ``DataSet`` class.

This means, that for each of the datasets, the data is downloaded from the web (if not easily found
in the working directory). It is also filtered (through ``DataSet`` and a functional graph ``Data`` set
is generated.


Bradley Double Plus
------------------------------------------------------------

.. automodule:: grape.datasets.jean_claude_bradley_double_plus
   :members:
   :undoc-members:
   :show-inheritance:

LogP
---------------------------------------

.. automodule:: grape.datasets.logP_dataset
   :members:
   :undoc-members:
   :show-inheritance:
