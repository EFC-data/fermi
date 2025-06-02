# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import trange
from typing import Union, List, Tuple, Any, Optional
from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

# BICM library
from bicm import BipartiteGraph
from bicm.network_functions import sample_bicm

# Bokeh - core functions
import bokeh
from bokeh.io import output_file, output_notebook, save, show
from bokeh.plotting import figure, from_networkx
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256, Inferno256, Plasma256, Magma256,Turbo256, Spectral4

# Bokeh - models
from bokeh.models import (
    Circle, ColumnDataSource, EdgesAndLinkedNodes, GraphRenderer,
    LabelSet, MultiLine, NodesAndLinkedEdges, StaticLayoutProvider, ColorBar, BasicTicker
)
from bokeh.palettes import Spectral4

# NetworkX algorithms
from networkx.algorithms import bipartite

from fermi.matrix_processor import MatrixProcessorCA

class RelatednessMetrics(MatrixProcessorCA):
    """
    Main relatedness methods for binary matrices, with optional statistical validation.

    This class provides tools to compute and validate projection networks
    derived from a binary (typically sparse) matrix.

    Features
    --------
    Relatedness metrics
        - Cooccurrence matrix
        - Proximity network
        - Taxonomy network
        - Assist network (from a second bipartite matrix)

    Statistical validation
        - Bonferroni correction
        - False Discovery Rate (FDR)
        - Direct thresholding

    Additional functionality
        - BICM sampling for statistical validation of projections
        - Matrix visualization with customizable sorting
        - Support for sparse matrices, configurable initial conditions,
          and custom row/column labels
    """

    def __init__(self, matrix: Union[np.ndarray, sp.spmatrix] = None):
        """
        Initializes the RelatednessMetrics class with a given binary matrix.
        The matrix is loaded into the class, and the internal state is set up for further processing.
        If the matrix is not provided, an empty instance is created.

        Parameters
        ----------
          - matrix : np.ndarray or scipy.sparse.spmatrix
              Input binary matrix (dense or sparse) representing the biadjacency matrix.
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

    
    ########################################
    ########## Internal Methods ############
    ########################################

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

        # Step 1: compute cooccurrence matrix
        # Convert to COO format for efficient row/column access
        cooc = A.dot(A.T).tocoo()
        ubiquity = np.array(A.sum(axis=1)).flatten()

        row = cooc.row
        col = cooc.col
        data = cooc.data

        ubi_max = np.maximum(ubiquity[row], ubiquity[col])
        with np.errstate(divide='ignore'):
            weights = np.where(ubi_max != 0, 1.0 / ubi_max, 0.0)
        # Step 2: compute proximity matrix by normalizing cooccurrence with ubiquity weights
        # Convert to CSR format for efficient matrix operations
        proximity_data = data * weights
        proximity = csr_matrix((proximity_data, (row, col)), shape=cooc.shape)

        return proximity

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

        # Step 1: diversification (rows norm)
        diversification = np.array(network.sum(axis=1)).flatten()
        with np.errstate(divide='ignore'):
            inv_div = np.where(diversification != 0, 1.0 / diversification, 0.0)
        div_diag = csr_matrix((inv_div, (np.arange(len(inv_div)), np.arange(len(inv_div)))), shape=(len(inv_div), len(inv_div)))
        m_div = div_diag.dot(network)

        # Step 2: intermediate product
        intermediate = network.T.dot(m_div)

        # Step 3: ubiquity normalization
        n = intermediate.shape[0]
        ubiquity = np.array(network.sum(axis=0)).flatten()

        # explicit meshgrid to create row and column indices and compute maximum ubiquity
        row_idx, col_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        max_ubiq = np.maximum(ubiquity[row_idx], ubiquity[col_idx])
        with np.errstate(divide='ignore'):
            weights = np.where(max_ubiq != 0, 1.0 / max_ubiq, 0.0)

        # compute taxonomy matrix
        taxonomy_dense = intermediate.toarray() * weights

        return csr_matrix(taxonomy_dense)

    def _assist(self, second_matrix: csr_matrix, rows: bool = True) -> csr_matrix:

        M = self._processed.T if rows else self._processed
        M_prime = second_matrix.T if rows else second_matrix
        # Step 1: compute normalization factors
        d_prime = np.array(M_prime.sum(axis=1)).flatten()
        d_prime[d_prime == 0] = 1

        # Step 2: compute the normalized second matrix
        D_inv = csr_matrix((1.0 / d_prime, (np.arange(len(d_prime)), np.arange(len(d_prime)))), shape=(len(d_prime), len(d_prime)))
        M_prime_norm = D_inv @ M_prime

        u = np.array(M.sum(axis=0)).flatten()
        u[u == 0] = 1
        # Step 3: compute the inverse of the ubiquity matrix
        U_inv = csr_matrix((1.0 / u, (np.arange(len(u)), np.arange(len(u)))), shape=(len(u), len(u)))
        # Step 4: compute the assist matrix
        result = M.T @ M_prime_norm
        result = U_inv @ result

        return result

    def _bonferroni_threshold(self, test_pvmat: np.ndarray, interval: float = 0.05, symmetry: bool = None) -> Tuple[List[Tuple[int, int]], List[float], float]:
        """
        Calculates the Bonferroni threshold for a bipartite matrix of p-values and returns
        the positions and p-values that satisfy the condition:

        p_value < interval / D

        where D is the total number of tested hypotheses (n * m).

        Parameters
        ----------
          - test_pvmat : np.ndarray
              Square matrix of p-values (N x N).
          - interval : float
              Significance level alpha to be divided by the number of hypotheses.
          - symmetry : bool
            If True, the matrix is considered symmetric (e.g., projection matrix),
            and only the upper triangle is considered for validation.
            If False, the entire matrix is considered (e.g., bipartite matrix).
            If None, the method will raise an error.
            
        Returns
        -------
          - positionvalidated : list of tuple
              List of (i, j) indices where p-value < interval/D.
          - pvvalidated : list of float
              List of p-values satisfying the threshold.
          - threshold : float
              Computed threshold value (interval / D).
        """

        if symmetry:
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
        elif not symmetry:
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

        else:
            raise ValueError(
            f"Unsupported symmetry parameter {symmetry}. Please enter True for symmetric matrix or False for non-symmetric matrix.")

    def _fdr_threshold(self, test_pvmat: np.ndarray, interval: float = 0.05, symmetry: bool = None) -> Tuple[List[Tuple[int, int]], List[float], float]:

        """
        Calculates the False Rate Discovery (FDR) threshold for a bipartite matrix of p-values and returns
        the positions and p-values that satisfy the condition:

        p_value < alpha_{FDR} = k * interval / D

        where D is the total number of tested hypotheses (n * m) and k is the highest index i that satisfies the relationship:

        p_value_i < i * interval / D

        Parameters
        ----------
          - test_pvmat : np.ndarray
              Square matrix of p-values (N x N).
          - interval : float
              Significance level alpha to be divided by the number of hypotheses.
          - symmetry : bool
            If True, the matrix is considered symmetric (e.g., projection matrix),
            and only the upper triangle is considered for validation.
            If False, the entire matrix is considered (e.g., bipartite matrix).
            If None, the method will raise an error.
            
        Returns
        -------
          - positionvalidated : list of tuple
              List of (i, j) indices where p-value < interval/D.
          - pvvalidated : list of float
              List of p-values satisfying the threshold.
          - threshold : float
              Computed threshold value alpha_{FDR}.
        """

        if symmetry:
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
        elif not symmetry:
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
        else:
            raise ValueError(
            f"Unsupported symmetry parameter {symmetry}. Please enter True for symmetric matrix or False for non-symmetric matrix.")


    def _direct_threshold(self, test_pvmat: np.ndarray, alpha: float=0.05, symmetry: bool=None) -> Tuple[List[Tuple[int, int]], List[float], float]:
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
        if symmetry:
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
        elif not symmetry:
        # Non-symmetric case: iterate through the entire matrix
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
        else:
            raise ValueError(
            f"Unsupported symmetry parameter {symmetry}. Please enter True for symmetric matrix or False for non-symmetric matrix.")

    def _validation_threshold(self, test_pvmat: np.ndarray, interval: float = 0.05, validation_method: Optional[str] = None, symmetry: Optional[bool] = None) -> Tuple[List[Tuple[int, int]], List[float], float]:

        """
        Validate the p-value matrix using the specified validation method.
        Parameters:
            test_pvmat: np.ndarray
                Matrix of p-values to validate.
            interval: float
                Significance level alpha to be divided by the number of hypotheses.
            validation_method: str, optional
                Method for validation, one of ['bonferroni', 'fdr', 'direct'].
            symmetry: bool, optional
                If True, the matrix is considered symmetric (e.g., projection matrix),
                and only the upper triangle is considered for validation.
                If False, the entire matrix is considered (e.g., bipartite matrix).
                If None, the method will raise an error.
        Returns:
            positionvalidated: list of tuple
                List of (i, j) indices where p-value satisfies the validation condition.
            pvvalidated: list of float
                List of p-values satisfying the validation condition.
            threshold: float
                Computed threshold value based on the validation method.
        """
        if validation_method is None:
            raise ValueError("Validation method must be specified. Choose from: bonferroni, fdr, direct.")
        elif validation_method not in ["bonferroni", "fdr", "direct"]:
            raise ValueError(f"Unsupported validation method {validation_method}. Choose from: bonferroni, fdr, direct.")
        if validation_method=="bonferroni":
            return self._bonferroni_threshold(test_pvmat, interval, symmetry)
        elif validation_method=="fdr":
            return self._fdr_threshold(test_pvmat, interval, symmetry)
        elif validation_method=="direct":
            return self._direct_threshold(test_pvmat, interval, symmetry)

    ############################################
    ########    Projection wrappers    #########
    ############################################

    def get_projection(self, second_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None, rows: bool = True, projection_method: Optional[str] = None) -> csr_matrix:
        """
        Compute a one-mode projection from a binary bipartite matrix.

        Depending on the chosen method, returns different networks of relationships
        either between rows or between columns of the input.

        Parameters
        ----------
        second_matrix : csr_matrix or ndarray, optional
            A second binary bipartite matrix required only if
            `projection_method == "assist"`. Must have the same shape
            (in the non-projected dimension) as the primary matrix.
        rows : bool, default=True
            If True, project onto the row-space (i.e., produce a row×row matrix);
            if False, project onto the column-space (column×column matrix).
        projection_method : {'cooccurrence', 'proximity', 'taxonomy', 'assist'}, default='cooccurrence'
            The algorithm to use for projection:
            - `'cooccurrence'` : raw cooccurrence counts.
            - `'proximity'`    : normalized cooccurrence (e.g., phi-coefficient).
            - `'taxonomy'`     : hierarchical or minimum-spanning projection.
            - `'assist'`       : cross-assist network using `second_matrix`.

        Returns
        -------
        projection : ndarray
            A square matrix of shape (n, n) where n is the number of
            rows if `rows=True` or number of columns if `rows=False`,
            containing the computed projection values.

        Raises
        ------
        ValueError
            If `projection_method` is not one of the supported options
            or if `second_matrix` is missing when required.
        """

        if projection_method == "cooccurrence":
            return self._cooccurrence(rows=rows)

        elif projection_method == "proximity":
            return self._proximity(rows=rows)

        elif projection_method == "taxonomy":
            return self._taxonomy(rows=rows)

        elif projection_method == "assist":
            if second_matrix is None:
                raise ValueError("Second matrix is required for assist method.")
            if not sp.issparse(second_matrix):
                second_matrix = csr_matrix(second_matrix)
            return self._assist(second_matrix, rows=rows)
        
        else:
            raise ValueError(
            f"Unsupported method {projection_method}. Choose from: cooccurrence, proximity, taxonomy, assist.")

    def get_bicm_projection(self, alpha: float = 0.05, num_iterations: int = 10000, projection_method: Optional[str] = None, rows: bool = True, second_matrix: Optional[Union[np.ndarray, sp.spmatrix]] = None, validation_method: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform BICM sampling and statistically validate a projection network.

        This method generates BICM samples to build a null distribution for
        a chosen one-mode projection, then applies a multiple‐comparison
        correction or threshold to identify significant links.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for validation (p-value threshold).
        num_iterations : int, default=10000
            Number of BICM random samples to generate.
        projection_method : {'cooccurrence', 'proximity', 'taxonomy', 'assist'}, optional
            Projection algorithm to apply:
            - 'cooccurrence' : raw cooccurrence counts.
            - 'proximity'    : normalized cooccurrence.
            - 'taxonomy'     : hierarchical projection.
            - 'assist'       : assist network (requires `second_matrix`).
        rows : bool, default=True
            If True, project onto rows (row×row matrix); if False, onto columns.
        second_matrix : ndarray or sparse matrix, optional
            Secondary bipartite matrix for 'assist' projection.
        validation_method : {'bonferroni', 'fdr', 'direct'}, optional
            Statistical validation procedure:
            - 'bonferroni' : Bonferroni correction.
            - 'fdr'        : False Discovery Rate.
            - 'direct'     : direct p-value threshold at `alpha`.

        Returns
        -------
        validated_relatedness : ndarray
            Binary matrix (0/1) indicating which links are significant.
        p_values : ndarray
            Matrix of p-values from the validation step.

        Raises
        ------
        ValueError
            If `projection_method` or `validation_method` is missing or unsupported,
            or if `second_matrix` is required but not provided.
        """
        if validation_method is None:
            raise ValueError("Validation method must be specified. Choose from: bonferroni, fdr, direct.")
        elif validation_method not in ["bonferroni", "fdr", "direct"]:
            raise ValueError(f"Unsupported validation method {validation_method}. Choose from: bonferroni, fdr, direct.")
        if projection_method is None:
            raise ValueError("Projection method must be specified. Choose from: cooccurrence, proximity, taxonomy, assist.")
        elif projection_method not in ["cooccurrence", "proximity", "taxonomy", "assist"]:
            raise ValueError(
            f"Unsupported projection method {projection_method}. Choose from: cooccurrence, proximity, taxonomy, assist.")
        
        original_bipartite = self._processed.copy()
        empirical_projection = self.get_projection(second_matrix=second_matrix, rows=rows, projection_method=projection_method)

        my_graph = BipartiteGraph()
        my_graph.set_biadjacency_matrix(self._processed)
        my_probability_matrix = my_graph.get_bicm_matrix()

        shape = empirical_projection.shape
        pvalues_matrix = np.zeros(shape, dtype=float)

        if projection_method == "assist":
            second_network = BipartiteGraph()
            second_network.set_biadjacency_matrix(second_matrix)
            other_probability_matrix = second_network.get_bicm_matrix()

            for _ in trange(num_iterations):
                self._processed = csr_matrix(sample_bicm(my_probability_matrix))
                second_sample = csr_matrix(sample_bicm(other_probability_matrix))
                pvalues_matrix = np.add(pvalues_matrix,np.where(self.get_projection(second_matrix=second_sample, rows=rows, projection_method=projection_method).toarray()>=empirical_projection, 1,0))

        else:
            for _ in trange(num_iterations):
                self._processed = csr_matrix(sample_bicm(my_probability_matrix))
                pvalues_matrix = np.add(pvalues_matrix,np.where(self.get_projection(rows=rows, projection_method=projection_method).toarray()>=empirical_projection, 1, 0))

        # after the iterations, we normalize the p-values matrix
        pvalues_matrix = pvalues_matrix / num_iterations

        self._processed = original_bipartite  # reset class network

        if projection_method == "assist":
            positionvalidated, pvvalidated, pvthreshold = self._validation_threshold(pvalues_matrix, alpha, validation_method=validation_method)
            validated_relatedness = np.zeros_like(pvalues_matrix, dtype=int)
            validated_values = np.zeros_like(pvalues_matrix)

            if len(positionvalidated) > 0:
                rows_idx, cols_idx = zip(*positionvalidated)
                validated_relatedness[rows_idx, cols_idx] = 1
                validated_values[rows_idx, cols_idx] = pvalues_matrix[rows_idx, cols_idx]

            return validated_relatedness, validated_values

        else:
            positionvalidated, pvvalidated, pvthreshold = self._validation_threshold(pvalues_matrix, alpha, validation_method=validation_method)
            validated_relatedness = np.zeros_like(pvalues_matrix, dtype=int)
            validated_values = np.zeros_like(pvalues_matrix)

            if len(positionvalidated) > 0:
                rows_idx, cols_idx = zip(*positionvalidated)
                validated_relatedness[rows_idx, cols_idx] = 1
                validated_relatedness[cols_idx, rows_idx] = 1
                validated_values[rows_idx, cols_idx] = pvalues_matrix[rows_idx, cols_idx]
                validated_values[cols_idx, rows_idx] = pvalues_matrix[rows_idx, cols_idx]

            return validated_relatedness, validated_values

    ##############################################################
    #########      Static methods for graph plotting     #########
    ##############################################################

    @staticmethod
    def mat_to_network(matrix: Union[np.ndarray, sp.spmatrix], projection: bool = None, row_names: Optional[List[str]] = None, col_names: Optional[List[str]] = None, node_names: Optional[List[str]] = None) -> nx.Graph:

        """
        Convert a bipartite or projected matrix into a NetworkX graph.

        Parameters
        ----------
        matrix : ndarray or sparse matrix
            The input matrix. If `projection` is False or None, this is
            the biadjacency (bipartite) matrix. If True, this is a
            one-mode projection matrix.
        projection : bool, optional
            Whether `matrix` is already a one-mode projection (`True`)
            or a bipartite adjacency (`False` or None).
        row_names : list of str, optional
            Labels for rows (used as node names in bipartite graph).
        col_names : list of str, optional
            Labels for columns (used as node names in bipartite graph).
        node_names : list of str, optional
            Custom names for nodes in a projected graph; must match
            the dimension of `matrix`.

        Returns
        -------
        graph : networkx.Graph
            A NetworkX graph representing either the bipartite structure
            or the one-mode projection.

        Raises
        ------
        ValueError
            If `matrix` is not 2-D, or if labels/names lengths do not match.
        """

        if projection:
            # For projection=True, we handle non-symmetric matrices by creating directed or undirected graphs
            # that capture all non-zero entries

            # Check if matrix is symmetric
            is_symmetric = np.allclose(matrix, matrix.T)

            if is_symmetric:
                # If symmetric, create undirected graph (more efficient)
                G = nx.Graph()
            else:
                # If non-symmetric, create directed graph to preserve all connections
                G = nx.DiGraph()

            if node_names is None:
                node_names = [f"{i}" for i in range(len(matrix))]

            if len(node_names) != len(matrix):
                print("The number of node names must be equal to the number of rows: default names assigned.")
                node_names = [f"{i}" for i in range(len(matrix))]

            G.add_nodes_from(node_names)

            if is_symmetric:
                # For symmetric matrices, only check upper triangle to avoid duplicate edges
                for i in range(len(matrix)):
                    for j in range(i + 1, len(matrix)):
                        weight = matrix[i, j]
                        if weight != 0:
                            G.add_edge(node_names[i], node_names[j], weight=weight)
            else:
                # For non-symmetric matrices, check all entries
                for i in range(len(matrix)):
                    for j in range(len(matrix)):
                        if i != j:  # Skip diagonal (self-loops)
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
    def plot_graph(G: nx.Graph, node_size: int = 5, weight: bool = True, layout: str = "", save: bool=False, 
                   interaction: bool = False, filename: str="graph.html", color: Optional[dict] = None, names: bool = False, 
                   projection: bool = False, centrality_metric: Optional[str] = None, spanning_tree: bool = False, 
                   seed: int = 42, modularity: bool = False) -> bokeh.plotting.figure:

        """
        Plot a network graph with various visualization options and centrality metrics.

        Parameters:
        -----------
        G : networkx.Graph or networkx.DiGraph
            The graph to visualize
        node_size : int
            Size of the nodes in the visualization
        weight : bool
            Whether to consider edge weights for line thickness
        layout : str
            Layout algorithm to use (spring, circular, kamada_kawai, etc.)
        save : bool
            Whether to save the plot to a file
        interaction : bool
            Whether to enable interactive features
        filename : str
            Name of the file to save the plot
        color : dict, list, or numpy.array
            Custom color mapping for nodes. Can be:
            - dict: {node_name: color_value}
            - list/array: color values in same order as graph nodes
            - numpy array: numeric values to be mapped to colors (like PCI values)
        names : bool, str, list, or numpy.array
            Whether to show node labels. Options:
            - bool: True shows original node names, False shows no labels
            - str: 'degree', 'closeness', 'betweenness' shows centrality values
            - list/array: custom names in same order as graph nodes
        projection : bool
            If True, treats the network as monopartite and enables centrality-based coloring.
            For directed graphs (non-symmetric adjacency matrices), each non-zero entry creates
            a connection with the corresponding weight.
        centrality_metric : str
            Centrality metric to use for coloring nodes ('degree', 'closeness', 'betweenness')
        spanning_tree : bool
            If True and projection is True, computes and displays the maximum spanning tree
            with optimized tree layout and automatic exclusion of isolated nodes
        seed : int
            Random seed for layout reproducibility (default: 42)
        modularity : bool
            If True and projection is True, computes modularity, detects communities,
            colors nodes by community, and adds a legend
        """

        def calculate_tree_layout(mst_graph: nx.Graph, layout_type: str = 'auto', root: Optional[str] = None, seed: int = 42) -> dict:
           
            """Calculate optimized layout for tree structures.
            Parameters:
                mst_graph (networkx.Graph): The graph to calculate layout for.
                layout_type (str): Type of layout to use ('auto', 'hierarchical', 'radial', 'spring_tree').
                root (str, optional): The root node for hierarchical layout.
                seed (int): Random seed for layout reproducibility.
            Returns:
                dict: A dictionary mapping node names to (x, y) coordinates.
            Raises:
                ValueError: If an invalid layout_type is provided.
            Notes:
                - 'auto': Automatically chooses the best layout based on tree characteristics.
                - 'hierarchical': Uses a hierarchical tree layout.
                - 'radial': Uses a radial tree layout.
                - 'spring_tree': Uses a spring layout for the tree structure.
            """
            import math
            from collections import deque

            if layout_type == 'auto':
                # Choose best layout based on tree characteristics
                num_nodes = len(mst_graph.nodes())
                if num_nodes < 20:
                    layout_type = 'radial'
                elif num_nodes < 100:
                    layout_type = 'hierarchical'
                else:
                    layout_type = 'spring_tree'

            if layout_type == 'hierarchical':
                # Hierarchical tree layout
                if root is None:
                    # Choose root as the node with highest degree in MST
                    root = max(mst_graph.degree(), key=lambda x: x[1])[0]

                pos = {}
                levels = {}

                # BFS to assign levels
                queue = deque([(root, 0)])
                visited = {root}
                levels[0] = [root]
                max_level = 0

                while queue:
                    node, level = queue.popleft()
                    max_level = max(max_level, level)

                    for neighbor in mst_graph.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, level + 1))
                            if level + 1 not in levels:
                                levels[level + 1] = []
                            levels[level + 1].append(neighbor)

                # Assign positions with better spacing
                for level, nodes in levels.items():
                    y = (max_level - level) * 2  # Increase vertical spacing
                    if len(nodes) == 1:
                        pos[nodes[0]] = (0, y)
                    else:
                        for i, node in enumerate(nodes):
                            x = (i - (len(nodes) - 1) / 2) * 1.5  # Increase horizontal spacing
                            pos[node] = (x, y)

            elif layout_type == 'radial':
                # Radial tree layout
                if root is None:
                    root = max(mst_graph.degree(), key=lambda x: x[1])[0]

                pos = {}
                levels = {}

                # BFS to assign levels
                queue = deque([(root, 0)])
                visited = {root}
                levels[0] = [root]

                while queue:
                    node, level = queue.popleft()

                    for neighbor in mst_graph.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, level + 1))
                            if level + 1 not in levels:
                                levels[level + 1] = []
                            levels[level + 1].append(neighbor)

                # Assign radial positions
                pos[root] = (0, 0)
                for level, nodes in levels.items():
                    if level == 0:
                        continue
                    radius = level * 2.5  # Increase radius spacing
                    if len(nodes) == 1:
                        angle = 0
                    else:
                        angle_step = 2 * math.pi / len(nodes)
                        for i, node in enumerate(nodes):
                            angle = i * angle_step
                            x = radius * math.cos(angle)
                            y = radius * math.sin(angle)
                            pos[node] = (x, y)

            elif layout_type == 'spring_tree':
                # Spring layout optimized for trees
                pos = nx.spring_layout(mst_graph, k=3.0, iterations=150, seed=seed)

            else:
                # Kamada-Kawai layout (good for revealing structure)
                try:
                    pos = nx.kamada_kawai_layout(mst_graph)
                except:
                    pos = nx.spring_layout(mst_graph, k=2.5, iterations=100, seed=seed)

            return pos

        # If projection is True, ensure we handle the graph as monopartite
        if projection:
            # Check if we're dealing with a directed graph (non-symmetric adjacency matrix)
            is_directed = G.is_directed()

            # If it's a directed graph, we'll work with it as-is
            # If it's undirected but we want projection, we still treat it as monopartite
            if bipartite.is_bipartite(G) and not is_directed:
                # We'll use the original graph for layout and visualization
                # but calculate centrality metrics on it directly
                is_bipartite = False
            else:
                is_bipartite = False
        else:
            is_bipartite = bipartite.is_bipartite(G)
            is_directed = G.is_directed()

        # Community detection and modularity calculation
        communities = None
        community_colors = None
        community_legend_data = None
        modularity_score = None

        if projection and modularity:
            try:
                # For directed graphs, convert to undirected for community detection
                graph_for_communities = G.to_undirected() if is_directed else G

                # Detect communities using the Louvain method
                communities = nx.community.greedy_modularity_communities(graph_for_communities, weight='weight' if weight else None)

                # Calculate modularity score
                modularity_score = nx.community.modularity(graph_for_communities, communities, weight='weight' if weight else None)
                print(f"Network Modularity: {modularity_score:.4f}")
                print(f"Number of communities detected: {len(communities)}")

                # Create node to community mapping
                node_to_community = {}
                for i, community in enumerate(communities):
                    for node in community:
                        node_to_community[node] = i

                # Generate distinct colors for communities
                from bokeh.palettes import Category20, Set3, Paired
                if len(communities) <= 20:
                    palette = Category20[max(3, len(communities))] if len(communities) > 2 else Category20[3]
                elif len(communities) <= 12:
                    palette = Set3[12]
                else:
                    palette = Paired[12]
                    # Extend palette if more communities than colors
                    while len(palette) < len(communities):
                        palette.extend(palette)

                community_colors = {i: palette[i % len(palette)] for i in range(len(communities))}

                # Prepare legend data
                community_legend_data = []
                for i, community in enumerate(communities):
                    community_legend_data.append({
                        'community': f'Community {i+1}',
                        'color': community_colors[i],
                        'size': len(community),
                        'nodes': sorted(list(community))[:5]  # Show first 5 nodes as example
                    })

                print("\nCommunity Summary:")
                for item in community_legend_data:
                    nodes_preview = ', '.join(str(n) for n in item['nodes'])
                    if item['size'] > 5:
                        nodes_preview += ', ...'
                    print(f"  {item['community']}: {item['size']} nodes ({nodes_preview})")

            except Exception as e:
                print(f"Error in community detection: {str(e)}")
                modularity = False

        # Check if graph has any edges before attempting to compute spanning tree
        has_edges = G.number_of_edges() > 0
        has_isolated_nodes = any(d == 0 for n, d in G.degree())

        # Calculate MAXIMUM spanning tree if requested and possible
        mst_edges = set()
        mst_graph = None
        connected_nodes = set()  # Track nodes that are part of connected components

        if projection and spanning_tree:
            if not has_edges:
                print("Unable to plot the maximum spanning tree: graph has no edges.")
                spanning_tree = False
            else:
                try:
                    # For directed graphs, convert to undirected for spanning tree calculation
                    if is_directed:
                        # Convert directed graph to undirected
                        # If there are edges in both directions, keep the one with maximum weight
                        undirected_G = G.to_undirected()

                        # Ensure weights are properly handled when converting
                        for u, v, data in undirected_G.edges(data=True):
                            if G.has_edge(u, v) and G.has_edge(v, u):
                                # Take maximum weight if edges exist in both directions
                                weight_uv = G[u][v].get('weight', 1)
                                weight_vu = G[v][u].get('weight', 1)
                                undirected_G[u][v]['weight'] = max(weight_uv, weight_vu)
                            elif G.has_edge(u, v):
                                undirected_G[u][v]['weight'] = G[u][v].get('weight', 1)
                            elif G.has_edge(v, u):
                                undirected_G[u][v]['weight'] = G[v][u].get('weight', 1)

                        graph_for_mst = undirected_G
                        print("Using undirected version of directed graph for maximum spanning tree calculation.")
                    else:
                        graph_for_mst = G

                    # Get all connected components and track connected nodes
                    components = list(nx.connected_components(graph_for_mst))

                    if nx.is_connected(graph_for_mst):
                        # Use maximum_spanning_tree for maximum spanning tree
                        mst_graph = nx.maximum_spanning_tree(graph_for_mst, weight='weight' if weight else None)
                        mst_edges = set(mst_graph.edges())
                        connected_nodes = set(graph_for_mst.nodes())

                        # Print MST statistics
                        total_weight = sum(data.get('weight', 1) for u, v, data in mst_graph.edges(data=True))
                        print(f"Maximum Spanning Tree computed successfully:")
                        print(f"  - Edges in MST: {len(mst_edges)}")
                        print(f"  - Total weight: {total_weight:.2f}")

                    else:
                        # If graph is not connected, compute maximum spanning forest
                        print(f"Graph is not connected ({len(components)} components). Computing maximum spanning forest.")

                        mst_edges = set()
                        mst_subgraphs = []
                        total_weight = 0

                        for component in components:
                            if len(component) > 1:  # Only process components with more than 1 node
                                subgraph = graph_for_mst.subgraph(component)
                                component_mst = nx.maximum_spanning_tree(subgraph, weight='weight' if weight else None)
                                mst_edges.update(component_mst.edges())
                                mst_subgraphs.append(component_mst)
                                connected_nodes.update(component)

                                # Add to total weight
                                component_weight = sum(data.get('weight', 1) for u, v, data in component_mst.edges(data=True))
                                total_weight += component_weight

                        # Create a combined MST graph from all components
                        mst_graph = nx.Graph()
                        for subgraph in mst_subgraphs:
                            mst_graph = nx.union(mst_graph, subgraph)

                        print(f"Maximum Spanning Forest computed:")
                        print(f"  - Edges in MSF: {len(mst_edges)}")
                        print(f"  - Total weight: {total_weight:.2f}")

                        if has_isolated_nodes:
                            isolated_count = len(G.nodes()) - len(connected_nodes)
                            print(f"  - Excluding {isolated_count} isolated nodes from visualization")

                except Exception as e:
                    print(f"Error computing maximum spanning tree: {str(e)}")
                    spanning_tree = False
                    mst_edges = set()

        # Determine which nodes to include in the visualization
        # When spanning_tree=True, automatically exclude isolated nodes and focus on tree
        if projection and spanning_tree and connected_nodes:
            # Only include nodes that are part of connected components
            nodes_to_visualize = connected_nodes
            G_visual = G.subgraph(nodes_to_visualize).copy()
        else:
            # Include all nodes
            nodes_to_visualize = set(G.nodes())
            G_visual = G

        # Calculate layout with seed for reproducibility
        # Use MST-optimized layout when showing spanning tree
        if projection and spanning_tree and mst_graph:
            if layout == "":
                # Auto-select best tree layout
                layout_pos = calculate_tree_layout(mst_graph, 'auto')
            elif layout in ['hierarchical', 'radial', 'spring_tree']:
                layout_pos = calculate_tree_layout(mst_graph, layout)
            elif layout == 'kamada_kawai':
                try:
                    layout_pos = nx.kamada_kawai_layout(mst_graph)
                except:
                    layout_pos = calculate_tree_layout(mst_graph, 'spring_tree')
            else:
                # Use specified layout on MST
                try:
                    layout_func = getattr(nx, f"{layout}_layout")
                    layout_pos = layout_func(mst_graph, seed=seed)
                except (AttributeError, TypeError):
                    layout_pos = calculate_tree_layout(mst_graph, 'auto')
        else:
            # Standard layout calculation for non-spanning-tree cases
            if layout:
                # Check if the layout algorithm supports seed parameter
                layout_func = getattr(nx, f"{layout}_layout")
                try:
                    # Try to pass seed parameter (works for spring_layout and some others)
                    layout_pos = layout_func(G_visual, seed=seed)
                except TypeError:
                    # If seed is not supported by this layout, use without seed
                    layout_pos = layout_func(G_visual)
            else:
                if is_bipartite and not projection:
                    top_nodes = {n for n, d in G_visual.nodes(data=True) if d.get("bipartite") == 0}
                    layout_pos = nx.bipartite_layout(G_visual, top_nodes)
                else:
                    # For spring layout, always use seed for reproducibility when projection=True
                    if projection:
                        layout_pos = nx.spring_layout(G_visual, k=None if len(G_visual) <= 25 else 3 / np.sqrt(len(G_visual)), seed=seed)
                    else:
                        layout_pos = nx.spring_layout(G_visual, k=None if len(G_visual) <= 25 else 3 / np.sqrt(len(G_visual)))

        # Calculate margins for visualization
        # When showing spanning tree, focus on connected nodes only
        if projection and spanning_tree and connected_nodes:
            # Only consider positions of connected nodes for plot range
            relevant_positions = {node: pos for node, pos in layout_pos.items() if node in connected_nodes}
            if relevant_positions:
                x_coords = [pos[0] for pos in relevant_positions.values()]
                y_coords = [pos[1] for pos in relevant_positions.values()]
            else:
                x_coords = [pos[0] for pos in layout_pos.values()]
                y_coords = [pos[1] for pos in layout_pos.values()]
        else:
            x_coords = [pos[0] for pos in layout_pos.values()]
            y_coords = [pos[1] for pos in layout_pos.values()]

        x_margin = (max(x_coords) - min(x_coords)) * 0.15 if x_coords else 1  # Reduced margin for better focus
        y_margin = (max(y_coords) - min(y_coords)) * 0.15 if y_coords else 1

        x_range = (min(x_coords) - x_margin, max(x_coords) + x_margin)
        y_range = (min(y_coords) - y_margin, max(y_coords) + y_margin)

        # Create adaptive figure
        title_suffix = ""
        if is_directed:
            title_suffix += " (Directed)"
        if projection and spanning_tree and mst_edges:
            title_suffix += " - Maximum Spanning Tree"
            if has_isolated_nodes:
                title_suffix += " (Tree Structure Optimized)"

        plot = figure(
            title="Graph Visualization" + title_suffix,
            x_range=x_range,
            y_range=y_range,
            tools="tap,box_select,lasso_select,reset,hover" if interaction else "",
            toolbar_location="above"
        )

        graph_renderer = GraphRenderer()

        # Node mapping - use only nodes that are being visualized
        node_list = list(G_visual.nodes)
        node_indices = list(range(len(node_list)))
        name_to_index = {name: idx for idx, name in enumerate(node_list)}

        # Calculate centrality metrics if projection is True or centrality_metric is provided
        centrality_values = {}
        if projection or (isinstance(names, str) and names in ['degree', 'closeness', 'betweenness']):
            metric_to_use = centrality_metric if centrality_metric else 'degree'
            if isinstance(names, str) and names in ['degree', 'closeness', 'betweenness']:
                metric_to_use = names

            # For spanning tree, calculate centrality on the MST instead of full graph
            graph_for_centrality = mst_graph if (projection and spanning_tree and mst_graph) else G_visual

            if metric_to_use == 'degree':
                if is_directed:
                    # For directed graphs, you might want to use in_degree, out_degree, or total degree
                    # Here we use total degree (in + out)
                    centrality_values = {node: (graph_for_centrality.in_degree(node) + graph_for_centrality.out_degree(node)) / (2 * (len(graph_for_centrality) - 1))
                                        for node in graph_for_centrality.nodes()}
                else:
                    centrality_values = nx.degree_centrality(graph_for_centrality)
            elif metric_to_use == 'closeness':
                centrality_values = nx.closeness_centrality(graph_for_centrality)
            elif metric_to_use == 'betweenness':
                centrality_values = nx.betweenness_centrality(graph_for_centrality)


    # ===== Main methods to manage colors =====

        # Determine if we're using custom colors from the color parameter
        use_custom_colors = False
        use_color_mapper = False
        custom_color_values = None
        mapper = None

        # First, check if we have custom color data
        if color is not None:
            import numpy as np

            if isinstance(color, (list, np.ndarray)):
                # Handle both numeric arrays and string color arrays
                if len(color) >= len(G.nodes()):
                    use_custom_colors = True

                    # Check if we have string colors or numeric values
                    sample_value = color[0] if len(color) > 0 else None

                    if isinstance(sample_value, str):
                        # Handle string colors (hex codes, color names, etc.)
                        use_color_mapper = False

                        # Create mapping from original graph nodes to color strings
                        original_nodes = list(G.nodes())
                        node_color_mapping = {}

                        for i, node in enumerate(original_nodes):
                            if i < len(color):
                                node_color_mapping[node] = str(color[i])

                        # Get color values for nodes we're actually visualizing
                        fill_colors = [node_color_mapping.get(node, "#1f77b4") for node in node_list]

                        print(f"Using custom string color mapping with {len(fill_colors)} colors")
                        print(f"Sample colors: {fill_colors[:3]}...")

                    else:
                        # Handle numeric arrays (like PCI values) - existing functionality
                        use_color_mapper = True

                        # Create mapping from original graph nodes to color values
                        original_nodes = list(G.nodes())
                        node_color_mapping = {}

                        for i, node in enumerate(original_nodes):
                            if i < len(color):
                                node_color_mapping[node] = float(color[i])

                        # Get color values for nodes we're actually visualizing
                        custom_color_values = [node_color_mapping.get(node, 0) for node in node_list]

                        # Create color mapper
                        mapper = linear_cmap(field_name='custom_color', palette=Viridis256,
                                            low=min(custom_color_values), high=max(custom_color_values))

                        print(f"Using custom numeric color mapping with {len(custom_color_values)} values")
                        print(f"Color range: {min(custom_color_values):.3f} to {max(custom_color_values):.3f}")

            elif isinstance(color, dict):
                # Handle dictionary mapping
                if all(n in color for n in node_list):
                    use_custom_colors = True
                    # Check if values are numeric (for color mapping) or direct colors
                    sample_value = next(iter(color.values()))
                    if isinstance(sample_value, (int, float)):
                        use_color_mapper = True
                        custom_color_values = [float(color[node]) for node in node_list]
                        mapper = linear_cmap(field_name='custom_color', palette=Turbo256,
                                            low=min(custom_color_values), high=max(custom_color_values))
                    else:
                        # Direct color mapping (strings)
                        use_color_mapper = False
                        fill_colors = [str(color[node]) for node in node_list]

        # Determine node colors based on priority - CUSTOM COLORS OVERRIDE EVERYTHING
        if use_custom_colors and use_color_mapper:
            # Custom numeric colors take HIGHEST priority - override centrality/modularity
            fill_colors = ["#1f77b4"] * len(node_list)  # Default, will be overridden by mapper
            print("Using custom numeric colors - overriding any centrality/modularity coloring")

        elif use_custom_colors and not use_color_mapper:
            # Direct color mapping (string colors) takes HIGHEST priority - override centrality/modularity
            # fill_colors already set above
            print("Using custom string colors - overriding any centrality/modularity coloring")

        elif projection and modularity and communities:
            # Use community membership for coloring (normal functionality when no custom colors)
            fill_colors = []
            for node in node_list:
                community_id = node_to_community.get(node, 0)
                fill_colors.append(community_colors[community_id])

        elif projection and centrality_metric in ['degree', 'closeness', 'betweenness']:
            # Use centrality for coloring (normal functionality when no custom colors)
            centrality_vals = [centrality_values[node] for node in node_list]

            # Create the color mapper - we'll use it directly in the glyph rather than pre-computing colors
            mapper = linear_cmap(field_name='centrality', palette=Viridis256,
                                low=min(centrality_vals), high=max(centrality_vals))
            # Default fill colors - will be overridden by the mapper
            fill_colors = ["#1f77b4"] * len(node_list)

        elif is_bipartite and not projection:
            top_nodes = {n for n, d in G_visual.nodes(data=True) if d.get("bipartite") == 0}
            bottom_nodes = set(G_visual.nodes) - top_nodes

            fill_colors = [
                Spectral4[0] if node in top_nodes else Spectral4[1]
                for node in node_list
            ]
        else:
            # Default coloring
            fill_colors = [Spectral4[0] for _ in node_list]

        # Node renderer data
        node_data = {
            "index": node_indices,
            "fill_color": fill_colors,
            "name": [str(node) for node in node_list]
        }

        # Add custom color values to node data if using color mapper
        if use_custom_colors and use_color_mapper:
            node_data["custom_color"] = custom_color_values

        # Add centrality values to node data if calculated
        if centrality_values and not (use_custom_colors and use_color_mapper):
            node_data["centrality"] = [centrality_values.get(node, 0) for node in node_list]

        # Add community information to node data
        if projection and modularity and communities:
            node_data["community"] = [f"Community {node_to_community.get(node, 0) + 1}" for node in node_list]

        graph_renderer.node_renderer.data_source.data = node_data

        # Node glyph - handle custom colors with mapper
        if use_custom_colors and use_color_mapper:
            # Use custom color mapper
            graph_renderer.node_renderer.glyph = Circle(
                radius=node_size / 100,
                fill_color=mapper,  # Use the custom color mapper
                line_color="dimgrey",
                line_width=2,
                fill_alpha=0.9,
            )
        elif projection and centrality_metric in ['degree', 'closeness', 'betweenness'] and not (modularity and communities) and not use_custom_colors:
            # Use centrality mapper (only if not using custom colors or modularity)
            graph_renderer.node_renderer.glyph = Circle(
                radius=node_size / 100,
                fill_color=mapper,  # Use the centrality mapper
                line_color="dimgrey",
                line_width=2,
                fill_alpha=0.9,
            )
        else:
            # Standard fill color (including modularity coloring and direct color mapping)
            graph_renderer.node_renderer.glyph = Circle(
                radius=node_size / 100,
                fill_color="fill_color",
                line_color="dimgrey",
                line_width=2,
                fill_alpha=0.9,
            )

        # Interaction: hover & selection
        if interaction:
            graph_renderer.node_renderer.selection_glyph = Circle(
                radius=node_size / 100,
                fill_color=Spectral4[2],
                line_color="dimgrey",
                line_width=2,
                fill_alpha=0.9,
            )
            graph_renderer.node_renderer.hover_glyph = Circle(
                radius=node_size / 100,
                fill_color=Spectral4[1],
                line_color="dimgrey",
                line_width=2,
                fill_alpha=0.9,
            )

        # Edge renderer - handle directed graphs properly
        start_indices = []
        end_indices = []
        line_widths = []
        edge_colors = []
        edge_alphas = []

        # Determine which edges to show based on spanning tree setting
        if projection and spanning_tree and mst_edges:
            # Show only maximum spanning tree edges, and only if both nodes are in visualization
            edges_to_show = []
            for start_node, end_node, data in G.edges(data=True):
                # Check if this edge is in the MST and both nodes are being visualized
                if ((start_node, end_node) in mst_edges or
                    (end_node, start_node) in mst_edges) and \
                    start_node in nodes_to_visualize and end_node in nodes_to_visualize:
                    edges_to_show.append((start_node, end_node, data))

            print(f"Displaying {len(edges_to_show)} edges from maximum spanning tree")
        else:
            # Show all edges between visualized nodes
            edges_to_show = [(start, end, data) for start, end, data in G_visual.edges(data=True)]

        for start_node, end_node, data in edges_to_show:
            start_indices.append(name_to_index[start_node])
            end_indices.append(name_to_index[end_node])
            edge_weight = data.get("weight", 1) if weight else 1
            # Scala i pesi per la visualizzazione
            visual_width = min(3, max(0.5, edge_weight * 0.1)) if weight else 1
            line_widths.append(visual_width)  # Ensure minimum line width of 1

            # Color spanning tree edges differently
            if projection and spanning_tree and mst_edges:
                # All displayed edges are MST edges, so color them prominently
                edge_colors.append("#FF4500")  # Orange-red for MST edges
                edge_alphas.append(0.95)
            else:
                # Regular edges
                edge_colors.append("#CCCCCC")  # Light grey for regular edges
                edge_alphas.append(0.6)

        graph_renderer.edge_renderer.data_source.data = {
            "start": start_indices,
            "end": end_indices,
            "line_width": line_widths,
            "line_color": edge_colors,
            "line_alpha": edge_alphas
        }

        # Use different glyph for directed graphs to show direction
        if is_directed:
            # For directed graphs, you might want to add arrows
            # This is a basic implementation - you could enhance with actual arrow heads
            graph_renderer.edge_renderer.glyph = MultiLine(
                line_color="line_color", line_alpha="line_alpha", line_width="line_width"
            )
        else:
            graph_renderer.edge_renderer.glyph = MultiLine(
                line_color="line_color", line_alpha="line_alpha", line_width="line_width"
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

        # Use the already calculated layout
        graph_layout = {name_to_index[node]: pos for node, pos in layout_pos.items()}
        graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        plot.renderers.append(graph_renderer)

        # Add a color bar if using custom colors or centrality metrics for coloring
        if use_custom_colors and use_color_mapper:
            color_bar = ColorBar(
                color_mapper=mapper['transform'],
                title="Custom Color Values (PCI)",
                ticker=BasicTicker(),
                location=(0, 0),
                orientation="vertical"
            )
            plot.add_layout(color_bar, "right")
        elif projection and centrality_metric in ['degree', 'closeness', 'betweenness'] and not (modularity and communities) and not use_custom_colors:
            color_bar = ColorBar(
                color_mapper=mapper['transform'],
                title=f"{centrality_metric.capitalize()} Centrality",
                ticker=BasicTicker(),
                location=(0, 0),
                orientation="vertical"
            )
            plot.add_layout(color_bar, "right")

        # Add community legend if modularity is enabled
        if projection and modularity and community_legend_data and not use_custom_colors:
            from bokeh.layouts import column, row

            # Create legend as a separate plot
            legend_y_pos = list(range(len(community_legend_data)))
            legend_y_pos.reverse()  # Reverse to show communities from top to bottom
            legend_colors = [item['color'] for item in community_legend_data]
            legend_labels = [f"{item['community']} ({item['size']} nodes)" for item in community_legend_data]

            legend_source = ColumnDataSource(data=dict(
                x=[0] * len(community_legend_data),
                y=legend_y_pos,
                color=legend_colors,
                label=legend_labels
            ))

            legend_plot = figure(
                width=280,
                height=max(200, len(community_legend_data) * 30 + 50),
                title="Communities",
                toolbar_location=None,
                x_range=(-0.5, 3),
                y_range=(-0.5, len(community_legend_data) - 0.5)
            )

            # Add colored circles for legend
            legend_plot.circle(x='x', y='y', size=20, color='color', source=legend_source,
                            line_color="dimgrey", line_width=1)

            # Add labels
            legend_labels_source = ColumnDataSource(data=dict(
                x=[0.4] * len(legend_labels),
                y=legend_y_pos,
                text=legend_labels
            ))

            legend_label_set = LabelSet(
                x='x', y='y', text='text', source=legend_labels_source,
                text_align="left", text_baseline="middle",
                text_font_size="10pt"
            )
            legend_plot.add_layout(legend_label_set)

            # Remove axes and grid
            legend_plot.xaxis.visible = False
            legend_plot.yaxis.visible = False
            legend_plot.xgrid.visible = False
            legend_plot.ygrid.visible = False
            legend_plot.outline_line_color = None

            # Create a layout with the main plot and legend side by side
            final_plot = row(plot, legend_plot)
        else:
            final_plot = plot

        # Node labels
        if names:
            x, y, labels = [], [], []

            # Get original node order for mapping custom names
            original_nodes = list(G.nodes())

            for node in G_visual.nodes():
                idx = name_to_index[node]
                x_pos, y_pos = graph_layout[idx]
                x.append(x_pos)
                y.append(y_pos)

                # Format labels based on the type of names parameter
                if isinstance(names, bool):
                    # Show original node names
                    labels.append(str(node))

                elif isinstance(names, str) and names in ['degree', 'closeness', 'betweenness']:
                    # Show centrality values
                    if centrality_values:
                        labels.append(f"{float(centrality_values[node]):.2f}")
                    else:
                        labels.append(str(node))  # Fallback to node name

                elif isinstance(names, (list, tuple)) or (hasattr(names, '__array__') and hasattr(names, '__len__')):
                    # Handle custom names list/array
                    try:
                        # Find the position of this node in the original graph
                        original_node_index = original_nodes.index(node)
                        if original_node_index < len(names):
                            labels.append(str(names[original_node_index]))
                        else:
                            labels.append(str(node))  # Fallback if index out of range
                    except (ValueError, IndexError):
                        labels.append(str(node))  # Fallback if node not found

                else:
                    # Fallback for any other case
                    labels.append(str(node))

            label_source = ColumnDataSource(data=dict(x=x, y=y, name=labels))
            label_set = LabelSet(
                x="x", y="y", text="name", source=label_source,
                text_align="center", text_baseline="middle",
                text_font_size="10pt", background_fill_color="white",
                background_fill_alpha=0.7
            )
            plot.add_layout(label_set)

        # Output handling
        if save:
            output_file(filename)
            show(final_plot)
            print(f"Saved to {filename}")
        else:
            output_notebook()
            show(final_plot, notebook_handle=True)
