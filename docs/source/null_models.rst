Null models and ICA
===================

Fermi delegates bipartite null-model fitting and sampling to WBNM. A model
name can be passed to :meth:`fermi.MatrixProcessorCA.compute_ica` and
:meth:`fermi.RelatednessMetrics.get_null_model_projection`.

Choosing a model
----------------

The models express different null hypotheses; they are not interchangeable
solver choices.

.. list-table::
   :header-rows: 1
   :widths: 16 18 32 34

   * - Name
     - Data
     - Constraints reproduced in expectation
     - Typical use
   * - ``bicm``
     - Binary
     - Row and column degrees
     - Test whether topology is explained by node degrees. Weighted input is
       binarized as nonzero/zero.
   * - ``biwcm``
     - Weighted
     - Row and column strengths
     - Standard Fermi ICA and weighted networks where strengths are the
       relevant constraints.
   * - ``biecm``
     - Weighted
     - Degrees and strengths
     - Separate topological and weight effects with the most constrained
       enhanced model.
   * - ``bipecm``
     - Weighted
     - Strengths and total edge count
     - Intermediate model when the total density matters but every node
       degree need not be constrained.
   * - ``bicrema``
     - Weighted
     - Degree stage followed by conditional strength reconstruction
     - Faster approximation to BiECM for larger systems.

Model names are case-insensitive and may contain ``-`` or ``_``. A WBNM
model class can also be supplied directly::

   from wbnm import BiECM

   processor.compute_ica(model=BiECM)

ICA
---

``compute_ica()`` fits the selected model and evaluates

.. math::

   ICA_{ij} = \frac{X_{ij}}{\mathbb{E}_{model}[X_{ij}]}.

BiWCM is the standard ICA model and the default::

   processor.compute_ica()

An explicit weighted model and solver configuration can be used as follows::

   processor.compute_ica(
       model="biecm",
       solve_kwargs={
           "method": "fixed-point",
           "tol": 1e-8,
           "max_iter": 10_000,
       },
       device="cpu",
   )

``solve_kwargs`` is forwarded unchanged to the selected WBNM model. Consult
the WBNM API for model-specific solver options. The fitted object is retained
as ``processor.null_model_`` for diagnostics such as convergence state,
expected matrices, likelihoods, and information criteria.

Isolated nodes
--------------

WBNM models cannot fit a node with zero degree or strength. Fermi handles this
at its boundary: empty rows and columns are removed before fitting and
restored as zeros in ICA matrices and samples. If every row or every column is
empty, ICA returns an all-zero matrix and no model is fitted.

Projection validation
---------------------

``get_null_model_projection()`` estimates projection p-values by Monte Carlo
sampling::

   validated, validated_pvalues = metrics.get_null_model_projection(
       null_model="biecm",
       projection_method="cooccurrence",
       validation_method="fdr",
       alpha=0.05,
       num_iterations=10_000,
       seed=42,
   )

The empirical projection is compared element-wise with each sampled
projection. The smallest nonzero p-value resolvable with ``N`` samples is
``1 / N``; an estimate can be zero when no sampled statistic reaches the
empirical value. Increase ``num_iterations`` for small significance levels.

The first returned array is a binary adjacency matrix of validated links. The
second contains p-values at validated positions and zeros elsewhere; it is
not the complete raw p-value matrix.

Reproducibility and devices
---------------------------

Set ``seed`` when sampling. Fermi derives a deterministic seed for each Monte
Carlo iteration. Use ``device="cpu"`` for portable reproducibility or a WBNM
supported accelerator for larger fits. Small numerical differences between
devices and solvers are expected.

BiCM compatibility
------------------

The backward-compatible wrapper remains available::

   validated, values = metrics.get_bicm_projection(
       projection_method="cooccurrence",
       validation_method="bonferroni",
   )

It is equivalent to calling ``get_null_model_projection(null_model="bicm",
...)`` and uses :class:`wbnm.BiCM`; the separate ``bicm`` distribution is not
required by Fermi 0.2.
