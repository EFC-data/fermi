Quickstart
==========

This example starts from a weighted actor--activity matrix, computes ICA,
creates a binary specialization matrix, evaluates economic-complexity metrics,
and constructs a relatedness network. ICA and RCA are both comparative-
advantage methods; Fermi presents ICA as the standard workflow and RCA as an
alternative.

Prepare data
------------

Pandas labels are preserved by Fermi::

   import pandas as pd

   exports = pd.DataFrame(
       [
           [8.0, 2.0, 0.0, 1.0],
           [1.0, 7.0, 3.0, 0.0],
           [0.0, 2.0, 6.0, 5.0],
           [3.0, 0.0, 1.0, 7.0],
       ],
       index=["A", "B", "C", "D"],
       columns=["p1", "p2", "p3", "p4"],
   )

Compute ICA and binarize
------------------------

The standard ICA workflow uses BiWCM. Transformations are chainable and
operate on the current processed matrix::

   from fermi import MatrixProcessorCA

   processor = MatrixProcessorCA().load(exports)
   specialization = (
       processor.copy()
       .compute_ica(model="biwcm")
       .binarize(threshold=1.0)
       .get_matrix(aspandas=True)
   )

``get_matrix()`` returns CSR by default. Use ``dense=True`` for a NumPy array
or ``aspandas=True`` for a labeled DataFrame.

RCA as an alternative
---------------------

RCA provides an alternative comparative-advantage transformation. The rest
of the workflow is unchanged::

   specialization_rca = (
       processor.copy()
       .compute_rca()
       .binarize(threshold=1.0)
       .get_matrix(aspandas=True)
   )

Fitness, Complexity, ECI, and PCI
---------------------------------

The :class:`fermi.efc` class inherits the preprocessing operations. Here the
standard ICA transformation is applied before computing the metrics::

   from fermi import efc

   economy = efc(exports).compute_ica(model="biwcm").binarize()
   fitness, complexity = economy.get_fitness_complexity(aspandas=True)
   eci, pci = economy.get_eci_pci(aspandas=True)
   diversification, ubiquity = economy.get_diversification_ubiquity(
       aspandas=True
   )

Relatedness
-----------

Use the binary specialization matrix for co-occurrence, proximity, and
taxonomy projections::

   from fermi import RelatednessMetrics

   relatedness = RelatednessMetrics(specialization.to_numpy())
   product_space = relatedness.get_projection(
       rows=False,
       projection_method="proximity",
   )

Validate links against the binary configuration model::

   significant, pvalues = relatedness.get_bicm_projection(
       rows=False,
       projection_method="cooccurrence",
       validation_method="fdr",
       num_iterations=1_000,
       seed=42,
   )

For exploratory tests use fewer Monte Carlo samples. Published analyses
should use enough iterations to resolve the desired significance threshold.

Choose the ICA null model
-------------------------

BiWCM is the standard ICA model. A different WBNM model can be selected when
the scientific null hypothesis requires different constraints::

   ica = (
       MatrixProcessorCA()
       .load(exports)
       .compute_ica(model="biecm")
       .get_matrix(aspandas=True)
   )

See :doc:`null_models` before changing the model: BiCM is binary, whereas the
other models impose different constraints on weighted networks.
