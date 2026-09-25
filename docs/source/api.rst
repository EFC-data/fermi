API reference
=============

This reference is generated from the public classes and functions in the
source code. See the user-guide pages for workflows and model-selection
guidance.

Matrix preprocessing
--------------------

.. autoclass:: fermi.MatrixProcessorCA
   :members:
   :show-inheritance:

Economic complexity
-------------------

.. autoclass:: fermi.efc
   :members:
   :show-inheritance:

Relatedness
-----------

.. autoclass:: fermi.RelatednessMetrics
   :members:
   :show-inheritance:

Prediction
----------

.. autoclass:: fermi.ECPredictor
   :members:

.. autoclass:: fermi.SPS
   :members:

Validation
----------

.. autoclass:: fermi.ValidationMetrics
   :members:

Null-model helpers
------------------

These helpers implement Fermi's adapter layer around WBNM. Most users should
call ``compute_ica()`` or ``get_null_model_projection()`` instead.

.. automodule:: fermi.null_models
   :members:
   :member-order: bysource
