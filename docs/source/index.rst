Fermi documentation
===================

**Fermi** (FitnEss, Relatedness, and other MetrIcs) is a Python toolkit for
economic-complexity analysis on bipartite matrices. It provides sparse matrix
preprocessing, comparative-advantage transformations, Fitness--Complexity and
ECI/PCI metrics, relatedness networks, statistical validation through WBNM
null models, forecasting, and prediction evaluation.

The distribution is named ``fermi-cref``; the import package is ``fermi``::

   import fermi
   print(fermi.__version__)

Fermi 0.2 uses :mod:`wbnm` for all supported bipartite null models. This
includes the binary :class:`wbnm.BiCM` as well as the weighted models
:class:`wbnm.BiWCM`, :class:`wbnm.BiECM`, :class:`wbnm.BiPECM`, and
:class:`wbnm.BiCReMA`.

Start here
----------

* :doc:`installation` explains installation, upgrades, and verification.
* :doc:`quickstart` gives an end-to-end working example.
* :doc:`data_and_preprocessing` documents accepted data and ICA/RCA.
* :doc:`null_models` explains how to choose a binary or weighted null model.
* :doc:`api` is the generated API reference.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   data_and_preprocessing
   null_models
   economic_complexity
   relatedness
   prediction
   validation
   migration

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
   references

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
