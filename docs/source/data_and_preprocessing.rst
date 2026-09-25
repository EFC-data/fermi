Data and preprocessing
======================

The :class:`fermi.MatrixProcessorCA` class is the common preprocessing base
used by the economic-complexity and relatedness modules. Internally, matrices
are stored as SciPy CSR sparse matrices.

Accepted inputs
---------------

``load()`` accepts:

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Input
     - Behaviour
     - Labels
   * - ``pandas.DataFrame``
     - Values are converted to CSR.
     - Non-default index and columns are preserved.
   * - Two-dimensional NumPy array or list
     - Converted to CSR.
     - No labels; use a DataFrame when labeled output is required.
   * - SciPy sparse matrix
     - Converted to CSR without densifying.
     - No labels; ``efc`` generates positional labels for metric output.
   * - CSV, TSV, TXT, or DAT path
     - Loaded with :func:`pandas.read_csv`; keyword arguments are forwarded.
     - A mostly non-numeric first column is inferred as the index.
   * - XLSX path
     - Loaded with :func:`pandas.read_excel`.
     - DataFrame labels are preserved.
   * - MTX/MM, NPZ, or NPY path
     - Loaded with the corresponding SciPy or NumPy reader.
     - No external labels.

Example with explicit CSV options::

   processor = MatrixProcessorCA().load(
       "exports.csv",
       index_col=0,
       dtype=float,
   )

CSV, TSV, and DAT have built-in default separators. Pass ``sep`` explicitly
for TXT files, for example ``sep=r"\s+"`` for arbitrary whitespace. XLSX
input requires a pandas Excel engine such as ``openpyxl``.

State and copies
----------------

Methods such as ``compute_ica()``, ``compute_rca()``, and ``binarize()``
replace the current processed matrix and return ``self`` for chaining. Use
``copy()`` when two transformations must start from the same data::

   source = MatrixProcessorCA().load(exports)
   ica = source.copy().compute_ica().get_matrix()
   rca = source.copy().compute_rca().get_matrix()

ICA
---

Inferred comparative advantage divides each observed value by the
corresponding expectation of a fitted null model::

   ica = MatrixProcessorCA().load(exports).compute_ica(
       model="biwcm",
       solve_kwargs={"tol": 1e-8, "max_iter": 10_000},
       device="cpu",
   )

Fermi removes isolated rows and columns before fitting and restores them as
zeros afterward. The fitted WBNM object is available as ``null_model_``.

BiWCM is the standard ICA model. See :doc:`null_models` before selecting a
different set of constraints.

RCA
---

Revealed comparative advantage is available as an alternative comparative-
advantage transformation. For a non-negative matrix :math:`X`,

.. math::

   RCA_{ij} = \frac{X_{ij}/\sum_j X_{ij}}
                    {\sum_i X_{ij}/\sum_{ij}X_{ij}}.

::

   rca = MatrixProcessorCA().load(exports).compute_rca()
   binary = rca.binarize(threshold=1.0)

Empty rows and columns remain zero. Negative values are not meaningful for
RCA and should be cleaned before loading.

Binarization
------------

``binarize(threshold)`` maps values greater than or equal to the threshold to
one and all other stored values to zero::

   binary = processor.binarize(threshold=1.0).get_matrix()

Output formats
--------------

::

   sparse = processor.get_matrix()             # scipy.sparse.csr_matrix
   array = processor.get_matrix(dense=True)    # numpy.ndarray
   frame = processor.get_matrix(aspandas=True) # labeled pandas.DataFrame

Do not pass both ``dense=True`` and ``aspandas=True``: ``dense`` takes
precedence. DataFrame output from ``MatrixProcessorCA`` requires labels from a
DataFrame or labeled file input. The ``efc`` class supplies positional labels
when its input has none.
