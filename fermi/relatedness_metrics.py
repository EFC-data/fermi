# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import trange
from typing import Union, List, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix

# BICM library
from bicm import BipartiteGraph
from bicm.network_functions import sample_bicm

# Bokeh - core functions
import bokeh
from bokeh.io import output_file, output_notebook, save, show
from bokeh.plotting import figure, from_networkx

# Bokeh - models
from bokeh.models import (
    Circle, ColumnDataSource, EdgesAndLinkedNodes, GraphRenderer,
    LabelSet, MultiLine, NodesAndLinkedEdges, StaticLayoutProvider
)
from bokeh.palettes import Spectral4

# NetworkX algorithms
from networkx.algorithms import bipartite

from fermi.matrix_processor import MatrixProcessorCA

class RelatednessMetrics(MatrixProcessorCA):
    """
    A class representing a bipartite network with statistical validation
    and relatedness computation.
    """
    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix] = None):
        """
        Initialize the bipartite network with a given binary matrix.

        Parameters
        ----------
          - matrix : np.ndarray or scipy.sparse.spmatrix
              Input binary matrix (dense or sparse) representing the biadjacency matrix.
          - hardcopy : bool, optional
              Whether to create a copy of the input matrix (default is False).
        """
        super().__init__()

        if matrix is not None:
            self.load(matrix.copy())
            
    def load(
            self,
            input_data: Union[str, Path, pd.DataFrame, np.ndarray, List[Any]],
            **kwargs
    ):
        super().load(input_data, **kwargs)
        self._processed_dense = self._processed.toarray()

    
########################################
########## INTERNAL METHODS ############
########################################

    def _cooccurrence_dense(self, rows: bool = True) -> np.ndarray:
        """
        Compute the cooccurrence matrix for one layer of the bipartite network.

        Parameters
        ----------
          - rows : bool, optional
              If True, compute cooccurrence on the row-layer; if False, on the column-layer.

        Returns
        -------
          - np.ndarray
              Cooccurrence matrix (square) of dimensions depending on the chosen layer.
        """
        if rows:
            return self._processed_dense.dot(self._processed_dense.T)
        else:
            return self._processed_dense.T.dot(self._processed_dense)

    def _cooccurrence(self, rows: bool = True) -> csr_matrix:
        """
        Compute the cooccurrence matrix for one layer of the bipartite network.

        Parameters
        ----------
        - rows : bool, optional
            If True, compute cooccurrence on the row-layer; if False, on the column-layer.

        Returns
        -------
        - csr_matrix
            Cooccurrence matrix (square, sparse) of dimensions depending on the chosen layer.
        """
        if rows:
            return self._processed.dot(self._processed.T)
        else:
            return self._processed.T.dot(self._processed)

    def _proximity_dense(self, rows: bool = True) -> np.ndarray:
        """
        Compute the proximity network from a bipartite network.
        Introduced by Hidalgo et al. (2007)

        Parameters
        ----------
          - rows : bool, optional
              If True, compute proximity for row-layer; if False, for column-layer.

        Returns
        -------
          - np.ndarray
              Proximity matrix where elements are cooccurrence weighted by inverse ubiquity.
        """
        if rows:
            cooc = self._processed_dense.dot(self._processed_dense.T)
            ubiquity = self._processed_dense.sum(axis=1)
        else:
            cooc = self._processed_dense.T.dot(self._processed_dense)
            ubiquity = self._processed_dense.sum(axis=0)

        ubi_mat = np.tile(ubiquity, (len(ubiquity), 1))
        ubi_max = np.maximum(ubi_mat, ubi_mat.T).astype(float)
        np.divide(np.ones_like(ubi_max, dtype=float), ubi_max, out=ubi_max, where=ubi_max != 0)
        return np.multiply(cooc, ubi_max)

    def _proximity(self, rows: bool = True) -> csr_matrix:
        """
        Compute the proximity network from a bipartite network.
        Introduced by Hidalgo et al. (2007)

        Parameters
        ----------
        - rows : bool, optional
            If True, compute proximity for row-layer; if False, for column-layer.

        Returns
        -------
        - csr_matrix
            Proximity matrix (sparse) where elements are cooccurrence weighted by inverse ubiquity.
        """
        if rows:
            A = self._processed
        else:
            A = self._processed.T

        cooc = A.dot(A.T).tocoo()
        ubiquity = np.array(A.sum(axis=1)).flatten()

        # Calcola inverso del massimo tra ubiquity[i] e ubiquity[j]
        row = cooc.row
        col = cooc.col
        data = cooc.data

        ubi_max = np.maximum(ubiquity[row], ubiquity[col])
        with np.errstate(divide='ignore'):
            weights = np.where(ubi_max != 0, 1.0 / ubi_max, 0.0)

        proximity_data = data * weights
        proximity = csr_matrix((proximity_data, (row, col)), shape=cooc.shape)

        return proximity

    def _taxonomy_dense(self, rows: bool = True) -> np.ndarray:
        """
        Compute the taxonomy network from a bipartite network.
        Introduced by Zaccaria et al. (2014)

        Parameters
        ----------
          - rows : bool, optional
              If True, compute taxonomy based on row to column to row transitions; otherwise column.

        Returns
        -------
          - np.ndarray
              Taxonomy matrix reflecting normalized transitions between nodes.
        """
        network = self._processed_dense.T if rows else self._processed_dense
        diversification = network.sum(axis=1)
        div_mat = np.tile(diversification, (network.shape[1], 1)).T
        m_div = np.divide(network, div_mat, where=div_mat != 0)
        intermediate = network.T.dot(m_div)

        ubiquity = network.sum(axis=0)
        ubi_mat = np.tile(ubiquity, (network.shape[1], 1))
        ubi_max = np.maximum(ubi_mat, ubi_mat.T).astype(float)
        np.divide(np.ones_like(ubi_max, dtype=float), ubi_max, out=ubi_max, where=ubi_max != 0)
        return np.multiply(intermediate, ubi_max)

    def _taxonomy(self, rows: bool = True) -> csr_matrix:
        """
        Compute the taxonomy network from a bipartite network.
        Introduced by Zaccaria et al. (2014)

        Parameters
        ----------
        - rows : bool, optional
            If True, compute taxonomy based on row to column to row transitions; otherwise column.

        Returns
        -------
        - csr_matrix
            Taxonomy matrix reflecting normalized transitions between nodes.
        """
        if rows:
            network = self._processed.T
        else:
            network = self._processed

        # Step 1: diversificazione (normalizzazione righe)
        diversification = np.array(network.sum(axis=1)).flatten()
        with np.errstate(divide='ignore'):
            inv_div = np.where(diversification != 0, 1.0 / diversification, 0.0)
        div_diag = csr_matrix((inv_div, (np.arange(len(inv_div)), np.arange(len(inv_div)))), shape=(len(inv_div), len(inv_div)))
        m_div = div_diag.dot(network)

        # Step 2: prodotto intermedio
        intermediate = network.T.dot(m_div)

        # Step 3: normalizzazione per ubiquità – TUTTE LE COPPIE (i,j)
        n = intermediate.shape[0]
        ubiquity = np.array(network.sum(axis=0)).flatten()

        # Costruzione esplicita di matrice dei pesi 1 / max(ubiq_i, ubiq_j)
        row_idx, col_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        max_ubiq = np.maximum(ubiquity[row_idx], ubiquity[col_idx])
        with np.errstate(divide='ignore'):
            weights = np.where(max_ubiq != 0, 1.0 / max_ubiq, 0.0)

        # Applichiamo la moltiplicazione elemento per elemento
        taxonomy_dense = intermediate.toarray() * weights

        return csr_matrix(taxonomy_dense)


    def _assist_dense(self, second_matrix: np.ndarray, rows: bool = True) -> np.ndarray:
        """
        Compute the assist matrix from a bipartite network.
        Introduced by Zaccaria et al. (2014)

        Parameters
        ----------
          - second_matrix : csr_matrix
              Second matrix for the assist matrix computation.
          - rows : bool, optional
              If True, compute assist matrix based on row to column to row transitions; otherwise column.

        Returns
        -------
          - np.ndarray
              Assist matrix reflecting normalized transitions between nodes.
        """
        m_0 = self._processed_dense.T if rows else self._processed_dense
        m_1 = second_matrix.T if rows else second_matrix

        d_1=m_1.sum(1)
        # let us automatically select the non-empty rows in m_1
        # and consider the summation just on them
        _m_0=m_0[d_1>0]
        _m_1=m_1[d_1>0]
        d_1=d_1[d_1>0]
        # let conside the second matrix term of the assist matrix
        mm_1=np.divide(_m_1.T, d_1).T

        # regarding the first term, there is the issue that if I have a product with
        # zero ubiquity, then the assist matrix will explode. In that case all
        # M_{cp} will be zero. Let's use a trick, then: let's calculate all ubiquities and
        # manually set to 1 all the 0 ones: their contribution will be still 0 (due to the numerator),
        # but we will avoid dividing by 0 in the following (and making things explode)
        u_0=_m_0.sum(0)
        # manually set to 1 all the 0s
        u_0[u_0==0]=1
        mm_0=np.divide(_m_0, u_0)
        return np.dot(mm_0.T,mm_1)

    def _assist(self, second_matrix: csr_matrix, rows: bool = True) -> csr_matrix:
        
        M = self._processed.T if rows else self._processed
        M_prime = second_matrix.T if rows else second_matrix

        d_prime = np.array(M_prime.sum(axis=1)).flatten()
        d_prime[d_prime == 0] = 1

        D_inv = csr_matrix((1.0 / d_prime, (np.arange(len(d_prime)), np.arange(len(d_prime)))), shape=(len(d_prime), len(d_prime)))
        M_prime_norm = D_inv @ M_prime

        u = np.array(M.sum(axis=0)).flatten()
        u[u == 0] = 1

        U_inv = csr_matrix((1.0 / u, (np.arange(len(u)), np.arange(len(u)))), shape=(len(u), len(u)))

        result = M.T @ M_prime_norm
        result = U_inv @ result

        return result

    def _bonferroni_threshold_dense(self, test_pvmat: np.ndarray, interval: float) -> Tuple[List[Tuple[int, int]], List[float], float]:
        """
        Calculates the Bonferroni threshold for a bipartite matrix of p-values and returns
        the positions and p-values that satisfy the condition:

        p_value < interval / D

        where D is the total number of tested hypotheses (n * m).

        Parameters
        ----------
          - test_pvmat : np.ndarray
              Square matrix of p-values (n x n).
          - interval : float
              Significance level alpha to be divided by the number of hypotheses.

        Returns
        -------
          - positionvalidated : list of tuple
              List of (i, j) indices where p-value < interval/D.
          - pvvalidated : list of float
              List of p-values satisfying the threshold.
          - threshold : float
              Computed threshold value (interval / D).
        """
        # Compute total number of tested hypotesis (D)
        D = test_pvmat.shape[0] * (test_pvmat.shape[0] - 1) / 2 #square matrix of pvalues/projection: does not consider diagonal and symmetrical terms
        #D = test_pvmat.shape[0] * test_pvmat.shape[1] #rectangular bipartite matrix: general case
        threshold = interval / D

        positionvalidated = []
        pvvalidated = []

        # Iterate on the whole matrix and select the positions with p-values less than threshold
        for i in range(test_pvmat.shape[0]):
            #for j in range(test_pvmat.shape[1]): #rectangular bipartite
            for j in range(i + 1, test_pvmat.shape[0]): #validated projection
                if test_pvmat[i, j] < threshold:
                    positionvalidated.append((i, j))
                    pvvalidated.append(test_pvmat[i, j])

        if not positionvalidated:
            print("No value satisfies the condition.")

        return positionvalidated, pvvalidated, threshold

    def _bonferroni_threshold(self, test_pvmat: csr_matrix, interval: float) -> Tuple[List[Tuple[int, int]], List[float], float]:
        """
        Calculates the Bonferroni threshold for a bipartite matrix of p-values and returns
        the positions and p-values that satisfy the condition:

        p_value < interval / D

        where D is the total number of tested hypotheses (n * (n - 1) / 2 for symmetric case).

        Parameters
        ----------
        - test_pvmat : csr_matrix
            Square matrix of p-values (n x n).
        - interval : float
            Significance level alpha to be divided by the number of hypotheses.

        Returns
        -------
        - positionvalidated : list of tuple
            List of (i, j) indices where p-value < interval/D.
        - pvvalidated : list of float
            List of p-values satisfying the threshold.
        - threshold : float
            Computed threshold value (interval / D).
        """
        n = test_pvmat.shape[0]
        D = n * (n - 1) / 2  # symmetric case (no diagonal, no duplicates)
        threshold = interval / D

        positionvalidated = []
        pvvalidated = []

        # Usando formato COO per iterare solo sugli elementi non zero
        coo = test_pvmat.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j and v < threshold:
                positionvalidated.append((i, j))
                pvvalidated.append(v)

        if not positionvalidated:
            print("No value satisfies the condition.")

        return positionvalidated, pvvalidated, threshold

    def _fdr_threshold_dense(self, test_pvmat, interval):
        D = test_pvmat.shape[0] * (test_pvmat.shape[0] - 1) / 2 #square matrix of pvalues/projection: does not consider diagonal and symmetrical terms
        #D = test_pvmat.shape[0] * test_pvmat.shape[1] #rectangular bipartite matrix: general case
        sorted_indices = []
        sortedpvaluesfdr = []

        for i in range(test_pvmat.shape[0]):
            for j in range(i + 1, test_pvmat.shape[0]): #rectangular bipartite
            #for j in range(test_pvmat.shape[1]): #validated projection
                sortedpvaluesfdr.append(test_pvmat[i][j])
                sorted_indices.append((i, j))

        sorted_pairs = sorted(zip(sortedpvaluesfdr, sorted_indices))  # Joint ordering
        sortedpvaluesfdr, sorted_indices = zip(*sorted_pairs)

        if len(sortedpvaluesfdr) == 0:
            print("No value satisfies the condition.")
            return [], [], None

        sortedpvaluesfdr = np.array(sortedpvaluesfdr)
        thresholds = np.arange(1, len(sortedpvaluesfdr) + 1) * interval / D
        valid_indices = np.where(sortedpvaluesfdr <= thresholds)[0]

        if len(valid_indices) == 0:
            print("No value satisfies the condition.")
            return [], [], None

        thresholdpos = valid_indices[-1]
        threshold = (thresholdpos + 1) * interval / D

        positionvalidated = []
        pvvalidated = []

        for i in range(len(sortedpvaluesfdr)):
            if sortedpvaluesfdr[i] <= threshold:
                positionvalidated.append(sorted_indices[i])
                pvvalidated.append(sortedpvaluesfdr[i])
            else:
                break

        if threshold is None:
            threshold = 0

        return positionvalidated, pvvalidated, threshold

    def _fdr_threshold(self, test_pvmat: csr_matrix, interval: float) -> Tuple[List[Tuple[int,int]], List[float], Optional[float]]:
        """
        Compute the FDR threshold for a bipartite matrix of p-values and return positions and values that satisfy the condition.

        Parameters
        ----------
        - test_pvmat : csr_matrix
            Square matrix of p-values.
        - interval : float
            Significance level alpha.

        Returns
        -------
        - positionvalidated : list of tuple
            List of (i, j) indices where p-values satisfy FDR threshold.
        - pvvalidated : list of float
            List of p-values passing the threshold.
        - threshold : float or None
            Computed threshold value or None if no values satisfy the condition.
        """
        n = test_pvmat.shape[0]
        D = n * (n - 1) / 2  # symmetric case: no diagonal, no duplicates

        # Estrai tutti i valori nella metà superiore
        coo = test_pvmat.tocoo()
        filtered = [(v, (i,j)) for i,j,v in zip(coo.row, coo.col, coo.data) if i < j]

        if not filtered:
            print("No value satisfies the condition.")
            return [], [], None

        sortedpvaluesfdr, sorted_indices = zip(*sorted(filtered, key=lambda x: x[0]))
        sortedpvaluesfdr = np.array(sortedpvaluesfdr)

        thresholds = (np.arange(1, len(sortedpvaluesfdr) + 1) * interval) / D
        valid_indices = np.where(sortedpvaluesfdr <= thresholds)[0]

        if len(valid_indices) == 0:
            print("No value satisfies the condition.")
            return [], [], None

        thresholdpos = valid_indices[-1]
        threshold = (thresholdpos + 1) * interval / D

        positionvalidated = []
        pvvalidated = []

        for i, pval in enumerate(sortedpvaluesfdr):
            if pval <= threshold:
                positionvalidated.append(sorted_indices[i])
                pvvalidated.append(pval)
            else:
                break

        return positionvalidated, pvvalidated, threshold

    def _direct_threshold_dense(self, test_pvmat, alpha):
        """
        Select the positions in the p-value matrix that meet the threshold specified by alpha.

        Args:
            test_pvmat (np.ndarray): P-value matrix.
            alpha (float): Fixed threshold to apply.

        Returns:
            positionvalidated (list of tuple): Indices (i,j) of the p-values that satisfy p <= alpha.
            pvvalidated (list of float): Corresponding p-values.
            threshold (float): The alpha threshold used.
        """
        sorted_indices = []
        sortedpvalues = []

        for i in range(test_pvmat.shape[0]):
            for j in range(i + 1, test_pvmat.shape[0]):  # parte superiore della matrice, escludendo diagonale
                sortedpvalues.append(test_pvmat[i][j])
                sorted_indices.append((i, j))

        if len(sortedpvalues) == 0:
            print("No value satisfies the condition.")
            return [], [], None

        positionvalidated = []
        pvvalidated = []

        for pv, idx in zip(sortedpvalues, sorted_indices):
            if pv <= alpha:
                positionvalidated.append(idx)
                pvvalidated.append(pv)

        if len(pvvalidated) == 0:
            print("No value satisfies the condition.")
            return [], [], None

        return positionvalidated, pvvalidated, alpha

    def _direct_threshold(self, test_pvmat: csr_matrix, alpha: float) -> Tuple[List[Tuple[int, int]], List[float], Optional[float]]:
        """
        Select the positions in the p-value matrix that meet the threshold specified by alpha.

        Args:
            test_pvmat (csr_matrix): P-value sparse matrix.
            alpha (float): Fixed threshold to apply.

        Returns:
            positionvalidated (list of tuple): Indices (i,j) of p-values satisfying p <= alpha.
            pvvalidated (list of float): Corresponding p-values.
            threshold (float or None): The alpha threshold used or None if no values satisfy.
        """
        coo = test_pvmat.tocoo()
        positionvalidated = []
        pvvalidated = []

        # Consider only upper triangular (i < j)
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j and v <= alpha:
                positionvalidated.append((i, j))
                pvvalidated.append(v)

        if not pvvalidated:
            print("No value satisfies the condition.")
            return [], [], None

        return positionvalidated, pvvalidated, alpha

    def _bonferroni_threshold_nonsymmetric_dense(self, test_pvmat, interval):
        """
        Bonferroni for non symmetrical matrix (es. bipartite): p_ij < alpha / D con D = n * m.
        """
        D = test_pvmat.shape[0] * test_pvmat.shape[1]
        threshold = interval / D

        positionvalidated = []
        pvvalidated = []

        for i in range(test_pvmat.shape[0]):
            for j in range(test_pvmat.shape[1]):
                if test_pvmat[i, j] < threshold:
                    positionvalidated.append((i, j))
                    pvvalidated.append(test_pvmat[i, j])

        if not positionvalidated:
            print("No value satisfies the condition.")

        return positionvalidated, pvvalidated, threshold

    def _bonferroni_threshold_nonsymmetric(self, test_pvmat: csr_matrix, interval: float) -> Tuple[List[Tuple[int,int]], List[float], float]:
        """
        Bonferroni correction for non-symmetric matrix (e.g. bipartite): 
        selects positions where p_ij < alpha / D with D = n * m.

        Parameters
        ----------
        - test_pvmat : csr_matrix
            Rectangular sparse matrix of p-values.
        - interval : float
            Significance level alpha.

        Returns
        -------
        - positionvalidated : list of tuple
            List of (i,j) indices where p-value < threshold.
        - pvvalidated : list of float
            Corresponding p-values.
        - threshold : float
            Computed threshold (interval / D).
        """
        n, m = test_pvmat.shape
        D = n * m
        threshold = interval / D

        positionvalidated = []
        pvvalidated = []

        coo = test_pvmat.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if v < threshold:
                positionvalidated.append((i, j))
                pvvalidated.append(v)

        if not positionvalidated:
            print("No value satisfies the condition.")

        return positionvalidated, pvvalidated, threshold

    def _fdr_threshold_nonsymmetric_dense(self, test_pvmat, interval):
        """
        False Discovery Rate for non symmetrical matrix.
        """
        D = test_pvmat.shape[0] * test_pvmat.shape[1]

        sortedpvalues = []
        sorted_indices = []

        for i in range(test_pvmat.shape[0]):
            for j in range(test_pvmat.shape[1]):
                sortedpvalues.append(test_pvmat[i, j])
                sorted_indices.append((i, j))

        if not sortedpvalues:
            print("No value satisfies the condition.")
            return [], [], None

        sorted_pairs = sorted(zip(sortedpvalues, sorted_indices))
        sortedpvalues, sorted_indices = zip(*sorted_pairs)
        sortedpvalues = np.array(sortedpvalues)

        thresholds = np.arange(1, len(sortedpvalues) + 1) * interval / D
        valid_indices = np.where(sortedpvalues <= thresholds)[0]

        if len(valid_indices) == 0:
            print("No value satisfies the condition.")
            return [], [], None

        thresholdpos = valid_indices[-1]
        threshold = (thresholdpos + 1) * interval / D

        positionvalidated = []
        pvvalidated = []

        for i in range(len(sortedpvalues)):
            if sortedpvalues[i] <= threshold:
                positionvalidated.append(sorted_indices[i])
                pvvalidated.append(sortedpvalues[i])
            else:
                break

        return positionvalidated, pvvalidated, threshold

    def _fdr_threshold_nonsymmetric(self, test_pvmat: csr_matrix, interval: float) -> Tuple[List[Tuple[int,int]], List[float], Optional[float]]:
        """
        False Discovery Rate for non-symmetrical matrix (e.g., bipartite).

        Parameters
        ----------
        - test_pvmat : csr_matrix
            Rectangular sparse matrix of p-values.
        - interval : float
            Significance level alpha.

        Returns
        -------
        - positionvalidated : list of tuple
            List of (i,j) indices passing the FDR threshold.
        - pvvalidated : list of float
            Corresponding p-values.
        - threshold : float or None
            Threshold value or None if none satisfies.
        """
        n, m = test_pvmat.shape
        D = n * m

        coo = test_pvmat.tocoo()
        sortedpvalues = []
        sorted_indices = []

        # Consider all entries (including zeros if present; if zeros mean no test, consider only nonzeros)
        for i, j, v in zip(coo.row, coo.col, coo.data):
            sortedpvalues.append(v)
            sorted_indices.append((i, j))

        if not sortedpvalues:
            print("No value satisfies the condition.")
            return [], [], None

        sorted_pairs = sorted(zip(sortedpvalues, sorted_indices))
        sortedpvalues, sorted_indices = zip(*sorted_pairs)
        sortedpvalues = np.array(sortedpvalues)

        thresholds = np.arange(1, len(sortedpvalues) + 1) * interval / D
        valid_indices = np.where(sortedpvalues <= thresholds)[0]

        if len(valid_indices) == 0:
            print("No value satisfies the condition.")
            return [], [], None

        thresholdpos = valid_indices[-1]
        threshold = (thresholdpos + 1) * interval / D

        positionvalidated = []
        pvvalidated = []

        for i in range(len(sortedpvalues)):
            if sortedpvalues[i] <= threshold:
                positionvalidated.append(sorted_indices[i])
                pvvalidated.append(sortedpvalues[i])
            else:
                break

        return positionvalidated, pvvalidated, threshold

    def _direct_threshold_nonsymmetric_dense(self, test_pvmat, alpha):
        """
        Filtra p-value for non symmetrical matrix with fixed threshold.
        """
        positionvalidated = []
        pvvalidated = []

        for i in range(test_pvmat.shape[0]):
            for j in range(test_pvmat.shape[1]):
                pv = test_pvmat[i, j]
                if pv <= alpha:
                    positionvalidated.append((i, j))
                    pvvalidated.append(pv)

        if not positionvalidated:
            print("No value satisfies the condition.")
            return [], [], None

        return positionvalidated, pvvalidated, alpha

    def _direct_threshold_nonsymmetric(self, test_pvmat: csr_matrix, alpha: float) -> Tuple[List[Tuple[int,int]], List[float], Optional[float]]:
        """
        Filter p-values for non-symmetrical matrix with fixed threshold alpha.

        Parameters
        ----------
        - test_pvmat : csr_matrix
            Rectangular sparse matrix of p-values.
        - alpha : float
            Fixed threshold.

        Returns
        -------
        - positionvalidated : list of tuple
            List of (i,j) indices with p-value <= alpha.
        - pvvalidated : list of float
            Corresponding p-values.
        - alpha : float or None
            Threshold used, or None if no values found.
        """
        positionvalidated = []
        pvvalidated = []

        coo = test_pvmat.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if v <= alpha:
                positionvalidated.append((i, j))
                pvvalidated.append(v)

        if not positionvalidated:
            print("No value satisfies the condition.")
            return [], [], None

        return positionvalidated, pvvalidated, alpha

    def _validation_threshold(self, test_pvmat, interval, method=None):
        if method=="bonferroni":
            return self._bonferroni_threshold(test_pvmat, interval)
        elif method=="fdr":
            return self._fdr_threshold(test_pvmat, interval)
        elif method=="direct":
            return self._direct_threshold(test_pvmat, interval)
        #return positionvalidated, pvvalidated, threshold
        else:
            raise ValueError(
            f"Unsupported method {method}. Please enter one of: bonferroni, fdr or direct.")

    def _validation_threshold_non_symm(self, test_pvmat, interval, method=None):
        if method=="bonferroni":
            return self._bonferroni_threshold_nonsymmetric(test_pvmat, interval)
        elif method=="fdr":
            return self._fdr_threshold_nonsymmetric(test_pvmat, interval)
        elif method=="direct":
            return self._direct_threshold_nonsymmetric(test_pvmat, interval)
        #return positionvalidated, pvvalidated, threshold
        else:
            raise ValueError(
            f"Unsupported method {method}. Please enter one of: bonferroni, fdr or direct.")

############################################
########    Projection wrappers    #########
############################################

    def get_projection_dense(self, second_matrix=None, rows=True, method="cooccurrence"):
        """
        Compute projection matrix, given binary bipartite input.

        Parameters:
            second_matrix: Second binary matrix if method == "assist"
            rows: boolean, if True processes rows, if False processes columns
            methods: string, ['cooccurrence', 'proximity', 'taxonomy', 'assist']

        Returns:
            numpy.ndarray: The projection matrix representing relationships between matrices
        """

        if method == "cooccurrence":
            return self._cooccurrence_dense(rows=rows)

        elif method == "proximity":
            return self._proximity_dense(rows=rows)

        elif method == "taxonomy":
            return self._taxonomy_dense(rows=rows)

        elif method == "assist":
            if second_matrix is None:
                raise ValueError("Second matrix is required for assist method.")
            if sp.issparse(second_matrix):
                second_matrix = second_matrix.toarray()
            return self._assist_dense(second_matrix, rows=rows)
        else:
            raise ValueError(
            f"Unsupported method {method}. Please enter one of: cooccurrence, proximity, taxonomy, assist.")

    def get_projection(self, second_matrix=None, rows=True, method="cooccurrence"):
        """
        Compute projection matrix, given binary bipartite input.

        Parameters:
            second_matrix: Second binary matrix if method == "assist"
            rows: boolean, if True processes rows, if False processes columns
            methods: string, ['cooccurrence', 'proximity', 'taxonomy', 'assist']

        Returns:
            numpy.ndarray: The projection matrix representing relationships between matrices
        """

        if method == "cooccurrence":
            return self._cooccurrence(rows=rows)

        elif method == "proximity":
            return self._proximity(rows=rows)

        elif method == "taxonomy":
            return self._taxonomy(rows=rows)

        elif method == "assist":
            if second_matrix is None:
                raise ValueError("Second matrix is required for assist method.")
            if sp.issparse(second_matrix):
                second_matrix = second_matrix.toarray()
            return self._assist(second_matrix, rows=rows)
        else:
            raise ValueError(
            f"Unsupported method {method}. Please enter one of: cooccurrence, proximity, taxonomy, assist.")

    def get_bicm_projection(self, alpha = 5e-2, num_iterations=int(1e4), method=None, rows=True, second_matrix=None, validation_method=None):
        """
        Generate BICM samples and validate the network.
        """
        original_bipartite = self._processed_dense
        empirical_projection = self.get_projection(second_matrix=second_matrix, rows=rows, method=method)

        myGraph = BipartiteGraph()
        myGraph.set_biadjacency_matrix(self._processed_dense)
        my_probability_matrix = myGraph.get_bicm_matrix()
        pvalues_matrix = np.zeros_like(empirical_projection)

        if method=="assist":
            second_network = BipartiteGraph()
            second_network.set_biadjacency_matrix(second_matrix)
            other_probability_matrix = second_network.get_bicm_matrix()

        for _ in trange(num_iterations):
            if method=="assist":
                self._processed_dense = sample_bicm(my_probability_matrix)
                second_sample = sample_bicm(other_probability_matrix)
                pvalues_matrix = np.add(pvalues_matrix,np.where(self.get_projection(second_matrix=second_sample, rows=rows, method=method)>=empirical_projection, 1,0))

            else:
                self._processed_dense = sample_bicm(my_probability_matrix)
                pvalues_matrix = np.add(pvalues_matrix,np.where(self.get_projection(second_matrix=second_matrix, rows=rows, method=method)>=empirical_projection, 1,0))

        pvalues_matrix = np.divide(pvalues_matrix, num_iterations)
        self._processed_dense = original_bipartite #reset class network
        if method=="assist":
            positionvalidated, pvvalidated, pvthreshold = self._validation_threshold_non_symm(pvalues_matrix, alpha, method=validation_method)
            validated_relatedness = np.zeros_like(pvalues_matrix, dtype=int)
            validated_values = np.zeros_like(pvalues_matrix)

            # validated position: (i, j)
            if len(positionvalidated) > 0:
                rows, cols = zip(*positionvalidated)

                # Imposta 1 su (i,j) e (j,i)
                validated_relatedness[rows, cols] = 1
                # validated_relatedness[cols, rows] = 1

                # Copia i valori della proximity media anche su (j,i)
                validated_values[rows, cols] = pvalues_matrix[rows, cols]
                # validated_values[cols, rows] = pvalues_matrix[rows, cols]

            return validated_relatedness, validated_values

        else:
            positionvalidated, pvvalidated, pvthreshold = self._validation_threshold(pvalues_matrix, alpha, method=validation_method)
            validated_relatedness = np.zeros_like(pvalues_matrix, dtype=int)
            validated_values = np.zeros_like(pvalues_matrix)

            # validated position: (i, j)
            if len(positionvalidated) > 0:
                rows, cols = zip(*positionvalidated)

                # Imposta 1 su (i,j) e (j,i)
                validated_relatedness[rows, cols] = 1
                validated_relatedness[cols, rows] = 1

                # Copia i valori della proximity media anche su (j,i)
                validated_values[rows, cols] = pvalues_matrix[rows, cols]
                validated_values[cols, rows] = pvalues_matrix[rows, cols]

            return validated_relatedness, validated_values

############################################
#########      Static methods      #########
############################################

    @staticmethod
    def generate_binary_matrix(rows=20, cols=20, probability=0.8):
        """Generate a random binary matrix.
        Useful for tests, not in the final product."""
        return np.random.binomial(1, probability, size=(rows, cols))

    @staticmethod
    def mat_to_network(matrix, projection=None, row_names=None, col_names=None, node_names=None):

        if projection:
            # Check simmetry
            # if not np.allclose(matrix, matrix.T):
            #     raise ValueError("Matrix is not simmetric: not a valid bipartite projection.")

            G = nx.Graph()

            if node_names is None:
                node_names = [f"Node_{i}" for i in range(len(matrix))]

            if len(node_names) != len(matrix):
                print("The number of node names must be equal to the number of rows: default names assigned.")
                node_names = [f"Node_{i}" for i in range(len(matrix))]

            G.add_nodes_from(node_names)

            for i in range(len(matrix)):
                for j in range(i + 1, len(matrix)):
                    weight = matrix[i, j]
                    if weight != 0:
                        G.add_edge(node_names[i], node_names[j], weight=weight)

            return G

        elif not projection:
            rows, cols = matrix.shape
            G = nx.Graph()

            if row_names is None:
                row_names = [f"R_{i}" for i in range(rows)]  # Prefix "R" for rows
            if col_names is None:
                col_names = [f"C_{j}" for j in range(cols)]  # Prefix "C" for columns

            if len(row_names) != rows:
                print("The number of row names must be equal to the number of rows: default names assigned.")
                row_names = [f"R{i}" for i in range(rows)]  # Prefix "R" for rows
            if len(col_names) != cols:
                print("The number of column names must be equal to the number of columns: default names assigned.")
                col_names = [f"C{j}" for j in range(cols)]  # Prefix "C" for columns

            G.add_nodes_from(row_names, bipartite=0)  # Layer 1 (rows)
            G.add_nodes_from(col_names, bipartite=1)  # Layer 2 (columns)

            for i in range(rows):
                for j in range(cols):
                    weight = matrix[i, j]
                    if weight != 0:
                        G.add_edge(row_names[i], col_names[j], weight=weight)

            return G
        else:
            raise ValueError(
            f"Unsupported projection parameter {projection}. projection=True if you intend to plot the projection, and projection=False otherwise.")

    @staticmethod
    def plot_graph(G, node_size=5, weight=True, layout="", save=False,
                interaction=False, filename="graph.html", color=None, names=False):

        '''
        Parameters:

        '''

        if layout:
            layout_pos = getattr(nx, f"{layout}_layout")(G)
        else:
            if bipartite.is_bipartite(G):
                top_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
                layout_pos = nx.bipartite_layout(G, top_nodes)
            else:
                layout_pos = nx.spring_layout(G, k=None if len(G) <= 25 else 3 / len(G))

        # Recalculate the display space based on the nodes
        x_coords = [pos[0] for pos in layout_pos.values()]
        y_coords = [pos[1] for pos in layout_pos.values()]
        x_margin = (max(x_coords) - min(x_coords)) * 0.2 if x_coords else 1
        y_margin = (max(y_coords) - min(y_coords)) * 0.2 if y_coords else 1

        x_range = (min(x_coords) - x_margin, max(x_coords) + x_margin)
        y_range = (min(y_coords) - y_margin, max(y_coords) + y_margin)

        # Create adaptive figure
        plot = figure(
            title="Graph Visualization",
            x_range=x_range,
            y_range=y_range,
            tools="tap,box_select,lasso_select,reset,hover" if interaction else "",
            #tools="tap,box_select,lasso_select,reset,hover" if interaction else "reset",
            toolbar_location="above"
        )

        graph_renderer = GraphRenderer()

        # Mappatura nodi -> indici
        node_list = list(G.nodes)
        node_indices = list(range(len(node_list)))
        name_to_index = {name: idx for idx, name in enumerate(node_list)}

        is_bipartite = bipartite.is_bipartite(G)
        if is_bipartite:
            top_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
            bottom_nodes = set(G.nodes) - top_nodes

            if isinstance(color, dict) and all(n in color for n in node_list):
                fill_colors = [color[node] for node in node_list]
            else:
                # Warn if the provided color is invalid (internally only, without print)
                fill_colors = [
                    Spectral4[0] if node in top_nodes else Spectral4[1]
                    for node in node_list
                ]
        else:
            if isinstance(color, dict) and all(n in color for n in node_list):
                fill_colors = [color[node] for node in node_list]
            else:
                fill_colors = [Spectral4[0] for _ in node_list]


        # Node renderer with border and opacity
        graph_renderer.node_renderer.data_source.data = {
            "index": node_indices,
            "fill_color": fill_colors
        }

        # Main node
        graph_renderer.node_renderer.glyph = Circle(
            #size=node_size,
            radius=node_size / 100,

            fill_color="fill_color",
            line_color="dimgrey",   # black border
            line_width=2,         # border thickness
            fill_alpha=0.9,        # opacity

        )

        # Interaction: hover & selection (same aesthetic features)
        if interaction:
            graph_renderer.node_renderer.selection_glyph = Circle(
                #size=node_size,
                radius=node_size / 100,
                fill_color=Spectral4[2],
                line_color="dimgrey",
                line_width=2,
                fill_alpha=0.9,

            )
            graph_renderer.node_renderer.hover_glyph = Circle(
                #size=node_size,
                radius=node_size / 100,
                fill_color=Spectral4[1],
                line_color="dimgrey",
                line_width=2,
                fill_alpha=0.9,
            )

        # Edge renderer
        start_indices = []
        end_indices = []
        line_widths = []

        for start_node, end_node, data in G.edges(data=True):
            start_indices.append(name_to_index[start_node])
            end_indices.append(name_to_index[end_node])
            line_widths.append(data.get("weight", 1) if weight else 1)

        graph_renderer.edge_renderer.data_source.data = {
            "start": start_indices,
            "end": end_indices,
            "line_width": line_widths
        }

        graph_renderer.edge_renderer.glyph = MultiLine(
            line_color="#CCCCCC", line_alpha=0.95, line_width="line_width" #line_alpha generale
        )

        if interaction:
            graph_renderer.edge_renderer.selection_glyph = MultiLine(
                line_color=Spectral4[2], line_width="line_width"
            )
            graph_renderer.edge_renderer.hover_glyph = MultiLine(
                line_color=Spectral4[1], line_width="line_width"
            )
            graph_renderer.selection_policy = NodesAndLinkedEdges()
            graph_renderer.inspection_policy = EdgesAndLinkedNodes()

        # Layout (spring, bipartite, ecc.)
        if layout:
            layout_pos = getattr(nx, f"{layout}_layout")(G)
        else:
            if bipartite.is_bipartite(G):
                top_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
                layout_pos = nx.bipartite_layout(G, top_nodes)
            else:
                layout_pos = nx.spring_layout(G)

        graph_layout = {name_to_index[node]: pos for node, pos in layout_pos.items()}
        graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        plot.renderers.append(graph_renderer)

        # Labels (only if names=True)
        if names:
            x, y, labels = [], [], []
            for node in G.nodes():
                idx = name_to_index[node]
                x_pos, y_pos = graph_layout[idx]
                x.append(x_pos)
                y.append(y_pos)
                labels.append(str(node))

            label_source = ColumnDataSource(data=dict(x=x, y=y, name=labels))
            labels = LabelSet(x="x", y="y", text="name", source=label_source,
                            text_align="center", text_baseline="middle", text_font_size="10pt")
            plot.add_layout(labels)

        #centering the graph: plot in margin
        coords = list(graph_layout.values())
        if coords:
            x_coords = [x for x, _ in coords]
            y_coords = [y for _, y in coords]

            x_margin = (max(x_coords) - min(x_coords)) * 0.2 if x_coords else 1
            y_margin = (max(y_coords) - min(y_coords)) * 0.2 if y_coords else 1

            plot.x_range.start = min(x_coords) - x_margin
            plot.x_range.end   = max(x_coords) + x_margin
            plot.y_range.start = min(y_coords) - y_margin
            plot.y_range.end   = max(y_coords) + y_margin


        if save==False:
            output_notebook()
            show(plot, notebook_handle=True)
            #print("not saved")

        if save:
            show(plot, notebook_handle=False)
            #output_notebook()
            #save(filename)
            output_file(filename)
            print("saved")
