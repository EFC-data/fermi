.. fermi documentation master file, created by
   sphinx-quickstart on Tue May 27 10:59:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fermi’s documentation!
=================================
`fermi` is a modular Python framework for analyzing the main Economic Complexity metrics and features.
It provides tools to explore the hidden structure of economies through:

- 📊 **Matrix preprocessing**: raw cleaning, sparse conversion, Comparative advantage RCA/ICA, transformation and thresholding.
- 🧠 **Fitness & complexity**: compute Fitness, Complexity ECI, PCI and other metrics via multiple methods.
- 🌐 **Relatedness metrics**: product space, taxonomy, assist matrix.
- 📈 **Prediction models**: GDP forecasting, density models, XGBoost.
- ✅ **Validation metrics**: AUC, confusion matrix, prediction@k.


Basic functionalities: Fitness and Complexity module
====================================================


The main module to generate an Economic Complexity object and initialize it (with a biadjacency matrix):

    import fermi
    myefc = fermi.efc()
    myefc.load(my_biadjacency_matrix, *possible kwargs*)

To compute the Revealed Comparative Advantage (Balassa index) and binarize its value

    myefc.compute_rca().binarize()

To compute the Fitness and the Complexity (using the original [Tacchella2012] algorithm)

    fitness, complexity = myefc.get_fitness_complexity()

To compute the diversification and the ubiquity

    div, ubi = myefc.get_diversification_ubiquity()

To compute the ECI index (using the eigenvalue method)

    eci, pci = myefc.get_eci_pci()


Basic functionalities: Relatedness module
=========================================
The module to generate cooccurrences and similar relatedness measures is

    myproj = fermi.RelatednessMetrics()
    myproj.load(my_biadjacency_matrix, *possible kwargs*)

The cooccurrence can be evaluated using

    relatedness = myproj.get_projection(projection_method="cooccurrence")
    validated_relatedness, validated_values = myproj.get_bicm_projection(projection_method="cooccurrence", validation_method="fdr")


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
