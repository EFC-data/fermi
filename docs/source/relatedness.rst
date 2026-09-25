Relatedness and projections
===========================

:class:`fermi.RelatednessMetrics` constructs one-mode networks from a
bipartite matrix and validates their links against WBNM null models.

Projection methods
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Method
     - Meaning
     - Output
   * - ``cooccurrence``
     - Raw overlap between pairs of nodes.
     - Symmetric projection.
   * - ``proximity``
     - Co-occurrence normalized by the larger degree of each pair.
     - Symmetric projection.
   * - ``taxonomy``
     - Degree-normalized two-step transitions inspired by the taxonomy
       network.
     - Symmetric projection.
   * - ``assist``
     - Directed relation between two bipartite matrices sharing the
       intermediate layer.
     - Generally asymmetric projection.

``rows=True`` projects the row layer; ``rows=False`` projects the column
layer::

   from fermi import RelatednessMetrics

   metrics = RelatednessMetrics(binary_matrix)
   countries = metrics.get_projection(
       rows=True,
       projection_method="cooccurrence",
   )
   products = metrics.get_projection(
       rows=False,
       projection_method="proximity",
   )

For ``assist``, provide ``second_matrix`` with a compatible shared dimension::

   assist = metrics.get_projection(
       second_matrix=next_period_matrix,
       projection_method="assist",
   )

Statistical validation
----------------------

Validation fits a WBNM model, samples bipartite matrices, recomputes the
projection, and estimates the probability of observing a value at least as
large as the empirical one::

   links, values = metrics.get_null_model_projection(
       null_model="bicm",
       projection_method="cooccurrence",
       validation_method="fdr",
       rows=False,
       alpha=0.05,
       num_iterations=10_000,
       seed=42,
   )

Available corrections are:

``direct``
   Compare every estimated p-value directly with ``alpha``.

``bonferroni``
   Divide ``alpha`` by the number of tested links.

``fdr``
   Apply the Benjamini--Hochberg false-discovery-rate procedure.

For symmetric projections only the upper triangle excluding the diagonal is
tested, and validated links are mirrored. Assist projections are tested as
directed matrices. See :doc:`null_models` for model choice and return-value
details.

Graph conversion
----------------

Convert a biadjacency matrix to a NetworkX bipartite graph::

   graph = RelatednessMetrics.mat_to_network(
       binary_matrix,
       projection=False,
       row_names=country_names,
       col_names=product_names,
   )

Convert a projection to an undirected or directed graph::

   graph = RelatednessMetrics.mat_to_network(
       products.toarray(),
       projection=True,
       node_names=product_names,
   )

Symmetric matrices produce ``networkx.Graph`` objects; asymmetric matrices
produce ``networkx.DiGraph`` objects. ``plot_graph()`` creates an interactive
Bokeh figure and supports layouts, labels, centrality coloring, modularity,
and maximum spanning trees.
