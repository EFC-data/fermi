Prediction
==========

Fermi contains two prediction interfaces: :class:`fermi.ECPredictor` for link
scores on bipartite matrices and :class:`fermi.SPS` for trajectory-based
forecasting in a multidimensional state space.

Network-based link prediction
-----------------------------

Given an actor--activity matrix :math:`M` and an activity--activity similarity
matrix :math:`B`, network prediction evaluates :math:`MB`::

   import numpy as np
   from fermi import ECPredictor

   M = np.array([[1, 0, 1], [0, 1, 1]])
   B = np.array(
       [[0.0, 0.8, 0.2], [0.8, 0.0, 0.4], [0.2, 0.4, 0.0]]
   )

   predictor = ECPredictor(M, mode="network", normalize=True)
   scores = predictor.predict_network(B)

With ``normalize=True``, each output column is divided by the corresponding
column sum of ``B``. Zero sums are treated as one.

Machine-learning link prediction
--------------------------------

Supply an estimator implementing ``fit`` and ``predict_proba``::

   from sklearn.ensemble import RandomForestClassifier

   predictor = ECPredictor(
       M_train[0],
       mode="ml",
       model=RandomForestClassifier(random_state=42),
   )
   future_scores = predictor.predict_ml_by_rowstack(
       M_list_train=M_train,
       Y_list_train=Y_train,
       M_test=M_future,
   )

Training matrices are stacked by row, and one binary classifier is fitted per
target column. Columns without a positive training example receive zero
scores. ``predict_ml_crossval()`` accepts a scikit-learn splitter and returns
out-of-fold scores with the shape of the stacked targets.

SPS trajectory forecasting
--------------------------

SPS expects one actor-by-year DataFrame per state-space dimension::

   import pandas as pd
   from fermi import SPS

   fitness = pd.DataFrame(
       {2018: [1.0, 0.8], 2019: [1.1, 0.9], 2020: [1.2, 1.0]},
       index=["A", "B"],
   )
   log_gdp = pd.DataFrame(
       {2018: [10.0, 9.0], 2019: [10.1, 9.2], 2020: [10.3, 9.3]},
       index=["A", "B"],
   )

   forecaster = SPS(
       {"fitness": fitness, "log_gdp": log_gdp},
       delta_t=1,
       sigma=0.5,
       n_boot=1_000,
       seed=42,
   )

The constructor aligns actors and years and builds ``state_matrix`` with a
``(actor, year)`` MultiIndex. Forecast an actor from historical analogues::

   forecaster.predict_actor("A", 2019, method="nw")
   result = forecaster.nw_actor

Use ``method="boot"`` for bootstrap forecasts; ``return_samples=True`` also
returns the sampled future positions. ``get_analogues()`` exposes the eligible
historical states used for a forecast. ``predict_actor_velocity()`` provides a
trajectory-only baseline, while ``sps_plus_velocity()`` combines SPS and
velocity estimates by inverse-variance weighting.

SPS requires enough complete historical observations for the selected actor,
year, horizon, and dimensions. Missing or non-consecutive histories can make
a requested forecast undefined.
