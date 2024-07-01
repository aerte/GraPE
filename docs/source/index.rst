GraPE-Chem - Graph-based Property Estimation for Chemistry
===========================================================

This is a python package to support Chemical property prediction using `PyTorch <https://pytorch.org/docs/stable/index.
html>`_ and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_.
The ambition of this project is to build a flexible pipeline that lets users go from molecule
descriptors (SMILES) to a fully functioning Graph Neural Network and allow for useful customization
at every step.



Installing the toolbox
----------------------

To use the package, please run the following inside of a terminal:

``pip install grape-chem``


Demonstrations and Use
-----------------------
After installing, the package will work like any other. See ``Demo``
and ``Advanced Demo`` for an introduction of how the toolbox can be used.



.. toctree::
    :maxdepth: 2
    :caption: Demonstrations
    :hidden:
    :glob:

    GraPE-Chem Demonstration
    Advanced GraPE-Chem Demonstration

.. toctree::
    :maxdepth: 2
    :caption: API Reference
    :hidden:
    :glob:

    api/models
    api/dataset
    api/analysis
    api/splitting
    api/plots
    api/utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`