Featurizers
============

Inspired by the `DGL-LifeSci <https://github.com/awslabs/dgl-lifesci>`_ featurizer, the ``AtomFeaturizer``
and ``BondFeaturizer`` allow for flexible molecule featurization using a list of feature names respectively.
This gives the user the room to customize the featurization step using any arbitrary combination of the in-build
feature functions.

Additionally, the classes offer the option of extending the featurizer with a *new* function.

.. automodule:: grape.utils.featurizer
   :members:
   :undoc-members:
   :show-inheritance:


Feature Functions
------------------

This is the collection of feature functions that can be used in the featurization (or separately).

.. automodule:: grape.utils.feature_func
   :members:
   :undoc-members:
   :show-inheritance: