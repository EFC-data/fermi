Prediction validation
=====================

:class:`fermi.ValidationMetrics` evaluates a matrix of prediction scores
against a binary ground-truth matrix of identical shape.

Basic use
---------

::

   import numpy as np
   from fermi import ValidationMetrics

   truth = np.array([[0, 1, 1], [1, 0, 0]])
   scores = np.array([[0.1, 0.8, 0.7], [0.9, 0.3, 0.2]])
   metrics = ValidationMetrics(truth, scores)

   roc_auc = metrics.roc_auc()
   pr_auc = metrics.pr_auc()
   threshold, f1 = metrics.best_f1()
   confusion = metrics.confusion_scores(threshold=threshold)

The matrices are flattened internally for global binary-classification
metrics.

Available metrics
-----------------

``roc_auc()``
   Area under the receiver-operating-characteristic curve.

``pr_auc()``
   Average precision, used as area under the precision--recall curve.

``best_f1()``
   Prediction threshold and maximum F1 across precision--recall thresholds.

``area_under_curves()``
   Convenience tuple ``(roc_auc, pr_auc)``.

``confusion_scores(threshold)``
   Counts and derived statistics including TPR, TNR, PPV, NPV, F1, accuracy,
   balanced accuracy, MCC, Fowlkes--Mallows, informedness, and markedness.

Ranking metrics
---------------

::

   precision_3 = metrics.precision_at_k(3)
   ap_3 = metrics.ap_at_k(3)
   map_3 = metrics.map_at_k(3)
   recall_3 = metrics.recall_at_k(3, threshold=0.5)

Scores are sorted in descending order before applying the cutoff. ``K=None``
or ``K=-1`` uses all elements. ``ndcg_at_k()`` is a static helper for batches
of recommendation rankings; its ``r`` argument must already be ordered by
rank and contain relevance values in the first ``k`` columns.

Edge cases
----------

ROC-AUC requires both classes to be present in the ground truth. Very small
top-k subsets can contain only one class; AP helpers return explicit boundary
values for all-negative and all-positive subsets. Always evaluate the metric
appropriate to the class balance and intended decision rule.

Entropy sampling
----------------

``hist_entropies()`` is an experimental diagnostic that repeatedly optimizes
random matrices with row and column totals close to the prediction matrix,
plots the entropy distribution, and returns the sampled entropies. It is more
computationally expensive than the deterministic validation metrics.
