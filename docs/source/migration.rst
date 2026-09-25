Migration to Fermi 0.2
======================

Fermi 0.2 replaces its direct use of the external ``bicm`` package with the
unified WBNM interface. WBNM includes both binary and weighted bipartite null
models, so Fermi no longer needs ``bicm`` as a direct dependency.

What remains compatible
-----------------------

The default ICA call is unchanged::

   processor.compute_ica()

It still means the standard BiWCM model. The backward-compatible projection
wrapper is also retained::

   metrics.get_bicm_projection(
       projection_method="cooccurrence",
       validation_method="fdr",
   )

The wrapper now fits :class:`wbnm.BiCM` internally.

New model selection
-------------------

ICA can select any WBNM model::

   processor.compute_ica(
       model="biecm",
       solve_kwargs={"method": "fixed-point"},
   )

Projection validation uses the generic method::

   metrics.get_null_model_projection(
       null_model="biecm",
       projection_method="cooccurrence",
       validation_method="fdr",
   )

Supported names are ``bicm``, ``biwcm``, ``biecm``, ``bipecm``, and
``bicrema``.

Binary versus weighted data
---------------------------

Do not mechanically replace BiCM with a weighted model. BiCM constrains binary
degrees and binarizes nonzero weighted observations. BiWCM constrains
strengths; BiECM constrains both degrees and strengths; BiPECM and BiCReMA
encode still different hypotheses. Revisit the scientific null hypothesis
when migrating an analysis. See :doc:`null_models`.

Solver and sampling changes
---------------------------

Solver options now go in ``solve_kwargs`` and are forwarded to WBNM::

   processor.compute_ica(
       model="bicm",
       solve_kwargs={"method": "coordinate", "tol": 1e-8},
   )

Projection sampling supports ``seed`` for deterministic runs and ``device``
for WBNM device selection. Fermi removes isolated nodes before fitting and
restores the original shape in expectations and samples.

Dependency update
-----------------

Remove an explicit ``bicm`` dependency only if no other part of the project
imports it directly. Fermi itself now requires ``wbnm>=0.1.0``::

   python -m pip uninstall bicm  # optional, after checking your own imports
   python -m pip install --upgrade "wbnm>=0.1.0" "fermi-cref>=0.2.0"
