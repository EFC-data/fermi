import numpy as np

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ValidationMetrics:
    """
    Evaluate a *prediction* matrix P against a *ground-truth* matrix M.

    Both arrays must have identical shape.
    The class flattens them internally so all metrics work on a 1-D view.
    """

    # ──────────────────────────────────────────────────────────────────
    def __init__(self, M, P):
        """
        Inizialise the class TestingPredictions.
        Parameters
        ----------
          - M : numpy.ndarray of 0/1 (int or bool)
              Ground-truth labels.
          - P : numpy.ndarray of floats
              Prediction scores (probabilities, logits, etc.).

        Raises
        ------
          - ValueError
              If the two matrices do not share the same shape.
        """
        if M.shape != P.shape:
            raise ValueError(
                "shape mismatch: ground-truth matrix has shape %s but prediction "
                "matrix has shape %s" % (M.shape, P.shape)
            )

        # store originals
        self.M = M.astype(int, copy=False)
        self.P = P.astype(float, copy=False)

        # flattened views for metric functions
        self.m_flat = self.M.ravel()
        self.p_flat = self.P.ravel()

    # ──────────────────────────── aucs ─────────────────────────────────
    def roc_auc(self):
        """
        Compute the **area under the ROC curve** (AUC-ROC).

        Returns
        -------
        float
            AUC-ROC score in the range [0, 1].
        """
        return roc_auc_score(self.m_flat, self.p_flat)

    def pr_auc(self):
        """
        Compute the **area under the Precision–Recall curve**
        (also called *average precision*).

        Returns
        -------
        float
            PR-AUC score in the range [0, 1].
        """
        return average_precision_score(self.m_flat, self.p_flat)

    # ─────────────────────── best-F1 across thresholds ──────────────────────
    def best_f1(self):
        """
        Find the decision threshold that maximises the F1 score.

        Returns
        -------
        tuple (threshold, best_f1)
            * **threshold** – score cut-off that yields the highest F1.
            * **best_f1**   – corresponding F1 value.
        """
        precision, recall, thresh = precision_recall_curve(
            self.m_flat, self.p_flat
        )

        # precision_recall_curve appends an extra entry; drop it
        precision, recall = precision[:-1], recall[:-1]

        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        idx = np.nanargmax(f1)
        return float(thresh[idx]), float(f1[idx])

    # ──────────────────────────── ranking helpers ───────────────────────────
    @staticmethod
    def _sorted_true_pred(true, pred):
        """
        Helper — sort *both* arrays by descending prediction score.

        Returns
        -------
        tuple (true_sorted, pred_sorted) – both 1-D NumPy arrays.
        """
        order = np.argsort(-pred)
        return true[order], pred[order]

    # ───────────────────────────── precision@K ──────────────────────────────
    def precision_at_k(self, K=None):
        """
        Precision among the **top-K highest-scoring items**.

        Parameters
        ----------
        K : int or None
            Number of items to keep.
            *K=None* or *K=-1* means “use all items”.

        Returns
        -------
        float
            Precision@K.
        """
        t, _ = self._sorted_true_pred(self.m_flat, self.p_flat)
        K = len(t) if K in (None, -1) else K
        return precision_score(t[:K], np.ones(K), zero_division=0)

    # ─────────────────────────── MAP@K (macro) ──────────────────────────────
    def map_at_k(self, K=None):
        """
        **Mean-Average-Precision** at cut-off *K*.

        This is the macro-average of AP for the positive and negative
        classes, matching the behaviour in your original function.

        Returns
        -------
        float
            MAP@K score in [0, 1].
        """
        t, p = self._sorted_true_pred(self.m_flat, self.p_flat)
        K = len(t) if K in (None, -1) else K
        t, p = t[:K], p[:K]

        if t.sum() == 0:
            return 0.0
        if t.sum() == K:
            return 1.0

        ap_pos = average_precision_score(t, p, pos_label=1)
        ap_neg = average_precision_score(t, p, pos_label=0)
        return float((ap_pos + ap_neg) / 2)

    # ───────────────────────────── AP@K (PR-AUC) ────────────────────────────
    def ap_at_k(self, K=None):
        """
        **Average-Precision** (area under PR curve) restricted to top-K.

        Returns
        -------
        float
            AP@K in [0, 1].
        """
        t, p = self._sorted_true_pred(self.m_flat, self.p_flat)
        K = len(t) if K in (None, -1) else K
        t, p = t[:K], p[:K]

        if t.sum() == 0:
            return 0.0
        if t.sum() == K:
            return 1.0

        return float(average_precision_score(t, p, pos_label=1))

    # ─────────────────────────── recall@K helper ────────────────────────────
    def recall_at_k(self, K, threshold):
        """
        Recall of the **top-K items** relative to recall at a fixed cut-off.

        Parameters
        ----------
        K : int (-1 or None → all items)
            How many high-score items to keep.
        threshold : float
            Global decision threshold used to compute TP and FN.

        Returns
        -------
        float
            Recall@K value.
        """
        t, p = self._sorted_true_pred(self.m_flat, self.p_flat)
        K = len(t) if K in (None, -1) else K
        TP_K = t[:K].sum()

        pred_bin = (self.p_flat >= threshold).astype(int)
        TP = ((self.m_flat == 1) & (pred_bin == 1)).sum()
        FN = ((self.m_flat == 1) & (pred_bin == 0)).sum()
        return TP_K / (TP + FN + 1e-12)

    # ───────────────────── confusion-matrix derived bundle ──────────────────
    def confusion_scores(self, threshold=0.5):
        """
        Compute a **rich set of confusion-matrix statistics** at
        the given threshold.

        Parameters
        ----------
        threshold : float, default 0.5
            Score cut-off for binarising predictions.

        Returns
        -------
        dict
            Dictionary with keys such as 'TP', 'FP', 'F1', 'ACC', 'MCC', etc.
        """
        y_pred = (self.p_flat >= threshold).astype(int)
        cm = confusion_matrix(self.m_flat, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        scores = {}

        # ──────────  raw confusion-matrix counts  ──────────
        scores["TP"] = float(tp)                           # true-positives
        scores["FP"] = float(fp)                           # false-positives
        scores["FN"] = float(fn)                           # false-negatives
        scores["TN"] = float(tn)                           # true-negatives

        # ──────────  primary rates and proportions  ───────
        scores["TPR"] = tp / (tp + fn + 1e-12)             # true-positive rate  (recall / sensitivity)
        scores["TNR"] = tn / (tn + fp + 1e-12)             # true-negative rate  (specificity)
        scores["PPV"] = tp / (tp + fp + 1e-12)             # positive predictive value (precision)
        scores["NPV"] = tn / (tn + fn + 1e-12)             # negative predictive value

        scores["FNR"] = fn / (tp + fn + 1e-12)             # false-negative rate
        scores["FPR"] = fp / (tn + fp + 1e-12)             # false-positive rate
        scores["FDR"] = fp / (tp + fp + 1e-12)             # false discovery rate
        scores["FOR"] = fn / (fn + tn + 1e-12)             # false omission rate

        # ──────────  composite / summary metrics  ─────────
        scores["PT"] = (                                   # prevalence-threshold
            (scores["TPR"] * (1 - scores["TNR"])) ** 0.5   # see Fluss et al. (2005)
            + scores["TNR"] - 1
        ) / (scores["TPR"] + scores["TNR"] - 1 + 1e-12)

        scores["TS"]  = tp / (tp + fn + fp + 1e-12)        # threat score (Jaccard index)
        scores["ACC"] = (tp + tn) / (tp + tn + fp + fn + 1e-12)  # accuracy
        scores["BA"]  = (scores["TPR"] + scores["TNR"]) / 2       # balanced accuracy
        scores["F1"]  = 2 * tp / (2 * tp + fp + fn + 1e-12)       # F1-measure (harmonic mean of precision & recall)
        scores["MCC"] = matthews_corrcoef(self.m_flat, y_pred)    # Matthews correlation coefficient
        scores["FM"]  = (scores["PPV"] * scores["TPR"]) ** 0.5    # Fowlkes–Mallows index
        scores["BM"]  = scores["TPR"] + scores["TNR"] - 1         # informedness / Bookmaker
        scores["MK"]  = scores["PPV"] + scores["NPV"] - 1         # markedness

        return scores

    # ───────────────────── quick bundle of both AUCs ───────────────────────
    def area_under_curves(self):
        """
        Convenience wrapper.

        Returns
        -------
        tuple (roc_auc, pr_auc)
        """
        return self.roc_auc(), self.pr_auc()

    # ─────────────────────── NDCG@K for recommendation ─────────────────────
    @staticmethod
    def ndcg_at_k(test_data, r, k):
        """
        Normalised Discounted Cumulative Gain for top-K recommendation.

        Parameters
        ----------
        test_data : list of list<int>
            For each user, the indices of relevant items.
        r : numpy.ndarray (users × items)
            Ranking matrix – **higher scores must be leftmost**.
        k : int
            Cut-off position.

        Returns
        -------
        float
            Sum of NDCG values over all users (same convention as your code).
        """
        assert len(test_data) == len(r)

        pred_data = r[:, :k]

        test_matrix = np.zeros_like(pred_data, dtype=float)
        for i, items in enumerate(test_data):
            length = k if k <= len(items) else len(items)
            test_matrix[i, :length] = 1.0

        idcg = np.sum(test_matrix / np.log2(np.arange(2, k + 2)), axis=1)
        dcg = (pred_data * (1.0 / np.log2(np.arange(2, k + 2)))).sum(axis=1)

        idcg[idcg == 0.0] = 1.0  # avoid division by zero
        ndcg = dcg / idcg
        return float(np.nan_to_num(ndcg).sum())


####################### FOR THE WORKSHOP SUMMER SCHOOL#########################
    # ──────────────────────────────────────────────────────────────────
    #  E N T R O P Y   S A M P L I N G   U N D E R   R O W / C O L   C O N S T R A I N T S
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _objective(flat_matrix, target):
        """
        Helper for `scipy.optimize.minimize`.

        Parameters
        ----------
        flat_matrix : 1-D numpy.ndarray
            Candidate matrix flattened. Will be reshaped inside.
        target : numpy.ndarray
            Target matrix whose row/column totals we want to match.

        Returns
        -------
        float
            L-1 distance between row-sums and column-sums of the candidate
            matrix and those of *target*.
        """
        mat = flat_matrix.reshape(target.shape)
        return (
            np.abs(mat.sum(axis=0) - target.sum(axis=0)).sum()
            + np.abs(mat.sum(axis=1) - target.sum(axis=1)).sum()
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(flat_matrix):
        """
        Shannon entropy of a flattened probability matrix.

        Parameters
        ----------
        flat_matrix : 1-D numpy.ndarray
            Matrix entries flattened.

        Returns
        -------
        float
            −Σ p log p   (ignoring zeros safely).
        """
        p = flat_matrix.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.nansum(p * np.log(p + 1e-12))

    # ------------------------------------------------------------------
    def hist_entropies(self, n_samples=100, progress=True):
        """
        Sample matrices whose **row and column totals** mimic matrix *P*,
        record their entropies, and plot a histogram.

        The optimisation treats each candidate entry as a free variable in
        [0, 1] and minimises the L-1 distance between its marginals and those
        of *P*.

        Parameters
        ----------
        n_samples : int, default 100
            How many random restarts / solutions to collect.
        progress : bool, default True
            Show a tqdm progress-bar if available.

        Returns
        -------
        list of float
            Entropy value for every sampled matrix.
        """
        from scipy.optimize import minimize
        import seaborn as sns

        iterator = range(n_samples)
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator)
            except ImportError:
                pass  # tqdm not installed → silently skip progress bar

        entropies = []
        bounds = [(0.0, 1.0)] * self.P.size  # each entry constrained to [0,1]

        for _ in iterator:
            res = minimize(
                fun=self._objective,
                x0=np.random.rand(self.P.size),  # random initial matrix
                args=(self.P,),
                bounds=bounds,
                method="L-BFGS-B",
            )
            entropies.append(self._entropy(res.x))

        sns.histplot(entropies)
        return entropies

