Economic-complexity metrics
===========================

The :class:`fermi.efc` class combines preprocessing and economic-complexity
metrics. Its input is interpreted as an actor--activity matrix, with actors on
rows and products, technologies, or other activities on columns.

Recommended workflow: ICA
-------------------------

Fitness--Complexity, ECI/PCI, diversification, ubiquity, and NODF are normally
computed on a binary specialization matrix. Fermi recommends computing
comparative advantage with ICA and the standard BiWCM model before
binarization::

   from fermi import efc

   model = efc(weighted_dataframe)
   model.compute_ica(model="biwcm").binarize(threshold=1.0)

   fitness, complexity = model.get_fitness_complexity(aspandas=True)
   eci, pci = model.get_eci_pci(aspandas=True)

Alternative workflow: RCA
-------------------------

RCA is available as an alternative comparative-advantage method. Once the
matrix is binarized, the downstream workflow is identical::

   rca_model = efc(weighted_dataframe)
   rca_model.compute_rca().binarize(threshold=1.0)

   fitness_rca, complexity_rca = rca_model.get_fitness_complexity(
       aspandas=True
   )

When the input is already a binary specialization matrix, construct ``efc``
directly from it and skip both ICA and RCA.

Fitness and Complexity
----------------------

The default is the nonlinear Tacchella algorithm::

   fitness, complexity = model.get_fitness_complexity(
       method="tacchella",
       max_iteration=1_000,
       min_distance=1e-14,
       normalization="sum",
       aspandas=True,
   )

The public wrapper caches results. Set ``force=True`` after changing options
or input state. Important keyword arguments forwarded to the solver include:

``method``
   ``"tacchella"`` or ``"servedio"``.

``check_stop``
   ``"distance"`` for an L1 convergence threshold or ``"crossing time"`` for
   the rank-crossing criterion.

``normalization``
   ``"sum"``, ``"max"``, ``"min"``, ``"mean"``, ``"none"``, or ``"zscore"``.

``fit_ic`` and ``com_ic``
   Optional initial conditions for row Fitness and column Complexity.

``removelowdegrees``
   Zero out columns whose degree is not above the supplied threshold.

``redundant``
   Use the just-updated complexity vector immediately in the Fitness update.

``delta``
   Positive shift used by the Servedio variant.

Dummy nodes
-----------

``add_dummy()`` can add an all-one reference row or column without modifying
the original object::

   with_reference = model.add_dummy(dummy_row=True, inplace=False)
   fitness, complexity = with_reference.get_fitness_complexity(
       with_dummy=True,
   )

ECI and PCI
-----------

The default eigenvalue method uses the second eigenvectors of normalized
row--row and column--column transition matrices::

   eci, pci = model.get_eci_pci(
       method="eigenvalue",
       norm="zscore",
       aspandas=True,
   )

The iterative Method of Reflections is also available::

   eci, pci = model.get_eci_pci(
       method="reflections",
       max_iterations=18,
       force=True,
   )

Degree and structural metrics
-----------------------------

::

   diversification, ubiquity = model.get_diversification_ubiquity(
       aspandas=True
   )
   density = model.get_density()
   nodf = model.get_nodf(nodf_method="nodf")
   stable_nodf = model.get_nodf(force=True, nodf_method="s_nodf")

Diversification is the row sum and ubiquity is the column sum. Density is the
fraction of nonzero cells. ``nodf`` selects standard NODF; ``s_nodf`` selects
Stable NODF.

Plotting
--------

``plot_matrix()`` reorders and displays the processed matrix::

   fig, ax = model.plot_matrix(index="fitness", cmap="Blues")

Available orderings include ``fitness``, ``eci``, ``degree``, ``custom``, and
``no``. With ``index="custom"``, pass row and column permutations through
``user_set0`` and ``user_set1``.
