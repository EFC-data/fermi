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
    This class implements the main relatedness methods, with optional statistical 
    validation, from a binary (typically sparse) matrix.

    Main functionalities include:
    - Relatedness metrics computation:
        - Cooccurrence matrix
        - Proximity network
        - Taxonomy network
        - Assist network (from a second bipartite matrix)
    - Statistical validation of the projection matrices:
        - Bonferroni correction
        - False Discovery Rate (FDR)
        - Direct thresholding
    - BICM sampling for statistical validation of the projections
    - Matrix visualization with customizable sorting for comparative analysis

    The class supports sparse matrices, configurable initial conditions, and row/column labeling.

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
        Compute projection matrix, given binary bipartite input.
        The method supports different projection methods:
        - "cooccurrence": Computes the cooccurrence matrix.
        - "proximity": Computes the proximity network.
        - "taxonomy": Computes the taxonomy network.
        - "assist": Computes the assist network from a second binary matrix.
        If the method is "assist", a second matrix must be provided.
        If the method is not specified, it raises a ValueError.

        Parameters:
            second_matrix: Second binary matrix if method == "assist"
            rows: boolean, if True processes rows, if False processes columns
            projection_method: string, ['cooccurrence', 'proximity', 'taxonomy', 'assist']

        Returns:
            numpy.ndarray: The projection matrix representing relationships between matrices
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
        Generate BICM samples and validate the network using sparse matrices.
        This method performs statistical validation of the projection matrix using BICM sampling.
        It supports different validation methods:
        - "bonferroni": Bonferroni correction for multiple comparisons.
        - "fdr": False Discovery Rate (FDR) correction.
        - "direct": Direct thresholding based on a fixed alpha value.
        The method also supports different projection methods:
        - "cooccurrence": Computes the cooccurrence matrix.
        - "proximity": Computes the proximity network.
        - "taxonomy": Computes the taxonomy network.
        - "assist": Computes the assist network from a second binary matrix.
            If the method is "assist", a second matrix must be provided.
        Parameters:
            alpha: float, significance level for validation
            num_iterations: int, number of BICM samples to generate
            projection_method: string, ['cooccurrence', 'proximity', 'taxonomy', 'assist']
            rows: boolean, if True processes rows, if False processes columns
            second_matrix: Second binary matrix if method == "assist"
            validation_method: string, ['bonferroni', 'fdr', 'direct']
        Returns:
            validated_relatedness: numpy.ndarray
                Binary matrix indicating validated relationships (1 for validated, 0 otherwise).
            validated_values: numpy.ndarray
                Matrix of p-values corresponding to the validated relationships.
        Raises:
            ValueError: If the validation method or projection method is not specified or unsupported.
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
        Convert a bipartite matrix to a NetworkX graph.
        Parameters:
            matrix: np.ndarray or scipy.sparse.spmatrix
                Input bipartite matrix (dense or sparse) representing the biadjacency matrix.
            projection: bool, optional
                If True, returns the bipartite projection graph.
                If False, returns the bipartite graph.
            row_names: list of str, optional
                Names for the rows of the bipartite matrix.
            col_names: list of str, optional
                Names for the columns of the bipartite matrix.
            node_names: list of str, optional
                Names for the nodes in the bipartite projection graph.
        Returns:
            nx.Graph: A NetworkX graph representing the bipartite matrix.
        Raises:
            ValueError: If the input matrix is not a valid bipartite matrix or if the projection parameter is invalid.
        """

        #Control if matrix is a valid bipartite matrix in np ndarray format
        if not isinstance(matrix, (np.ndarray, sp.spmatrix)):
            raise ValueError("Input matrix must be a numpy ndarray or a scipy sparse matrix.")
        if isinstance(matrix, sp.spmatrix):
            matrix = matrix.toarray()
        if projection:
            # Check symmetry
            if not np.allclose(matrix, matrix.T):
                symmetry = False
            else:
                symmetry = True
                
            # Create a NetworkX graph from the adjacency matrix

            G = nx.Graph()

            if node_names is None:
                node_names = [f"Node_{i}" for i in range(len(matrix))]

            if len(node_names) != len(matrix):
                print("The number of node names must be equal to the number of rows: default names assigned.")
                node_names = [f"Node_{i}" for i in range(len(matrix))]

            G.add_nodes_from(node_names)
            if symmetry:
                for i in range(len(matrix)):
                    for j in range(i + 1, len(matrix)):
                        weight = matrix[i, j]
                        if weight != 0:
                            G.add_edge(node_names[i], node_names[j], weight=weight)
            elif not symmetry:
                for i in range(len(matrix)):
                    for j in range(len(matrix)):
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
                   interaction: bool = False, filename: str="graph.html", color: Optional[dict] = None, names: bool = False) -> bokeh.plotting.figure:

        """
        Plot a graph using NetworkX and Bokeh

        Parameters:
        - G : nx.Graph
            Input graph to be plotted.
            - node_size : int, optional
                Size of the nodes in the graph. Default is 5.
            - weight : bool, optional
                If True, edge weights are considered in the visualization. Default is True.
            - layout : str, optional
                Layout algorithm to use for positioning nodes. Default is "" (no specific layout).
            - save : bool, optional
                If True, saves the plot as an HTML file. Default is False.
            - interaction : bool, optional
                If True, enables interaction features like hover and selection. Default is False.
            - filename : str, optional
                Name of the HTML file to save the plot if `save` is True. Default is "graph.html".
            - color : dict, optional
                Dictionary mapping node names to colors. Default is None (uses default colors).
            - names : bool, optional
                If True, displays node names on the graph. Default is False.
        Returns:
            - plot : bokeh.plotting.figure
            Bokeh figure object containing the graph visualization.
        """
        # control if G is a valid graph
        if not isinstance(G, nx.Graph):
            raise ValueError("Input G must be a NetworkX Graph object. Please convert your matrix to a graph using mat_to_network() method.")
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
