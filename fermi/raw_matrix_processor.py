import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from typing import Any, List, Tuple, Union
from scipy.sparse import csr_matrix

class RawMatrixProcessor:

    """
        RawMatrixProcessor is a flexible utility for loading, managing, and aligning sparse matrices.

        What it can do:
        - Load a single matrix from a file or from Python objects (DataFrame, numpy array, list, edge list, etc.)
        - Automatically convert data into a sparse CSR matrix format
        - Extract row and column labels when available (e.g. from CSV or DataFrame)
        - Add multiple matrices with associated row/column labels
        - Align all added matrices to a shared global set of row/column labels
        - Fill in missing entries with zeros so all matrices have the same shape

        Example use cases:
        - Combining time series of bipartite networks with different sets of nodes
        - Preparing input matrices for machine learning or network analysis
        - Harmonizing real-world data from different sources

        Output:
        - Aligned sparse matrices ready for analysis
        - Global row and column labels for consistent indexing
    """

    def __init__(self) -> None:
        """
        Initialize the processor with empty internal storage for matrices and global labels
        """
        # This is the main list that will hold all matrices and their labels
        # Each element is a tuple: (matrix, row_labels, col_labels)
        self.matrices = []
        self.global_row_labels = []  # Combined list of all unique row labels
        self.global_col_labels = []  # Combined list of all unique column labels

    def load_to_sparse(
        self,
        input_data: Union[str, pd.DataFrame, np.ndarray, List[Any]],
        return_labels: bool = False
        ) -> Union[csr_matrix, Tuple[csr_matrix, List[str], List[str]]]:
        
        """
        Loads the input and converts it to a sparse CSR matrix.

        Parameters
        ----------
          - input_data:  str, pd.DataFrame, np.ndarray, or list
              Can be a file path (CSV, TSV), a pandas DataFrame, a numpy array, or other formats
          - return_labels: bool
              if True, and the input is a file or DataFrame, also return row and column labels

        Returns
        -------
          - scipy.sparse.csr_matrix 
              optionally row and column labels if return_labels is True
        """
        if isinstance(input_data, str) and os.path.exists(input_data):
            # If input is a file path, read it as a DataFrame
            df = self._read_file(input_data)
            matrix = sp.csr_matrix(df.values)
            if return_labels:
                return matrix, list(df.index.astype(str)), list(df.columns.astype(str))
            return matrix

        elif isinstance(input_data, pd.DataFrame):
            # If input is already a DataFrame
            matrix = sp.csr_matrix(input_data.values)
            if return_labels:
                return matrix, list(input_data.index.astype(str)), list(input_data.columns.astype(str))
            return matrix

        else:
            # For other types (numpy arrays, edge lists, etc.)
            matrix = self._convert_to_sparse(input_data)
            if return_labels:
                raise ValueError("Cannot extract labels from this data type.")
            return matrix

    def add_matrix(
        self,
        input_data: Any,
        row_labels: List[str],
        col_labels: List[str]
        ) -> None:
        """
        Add a matrix and its labels to the internal collection.

        Parameters:
        -----------
          - input_data: Any
               Data convertible to a sparse matrix.
          - row_labels: list[str]
              list of row label strings
          - col_labels: list[str]
              list of column label strings
        """
        matrix = self.load_to_sparse(input_data)
        self._update_union(self.global_row_labels, row_labels)
        self._update_union(self.global_col_labels, col_labels)
        self.matrices.append((matrix, row_labels, col_labels))

    def get_aligned_matrices(self) -> List[csr_matrix]:
        """
        Aligns all stored matrices to the same global row/column space.
        Missing values are filled with zeros.

        Returns
        -------
          - aligned: list[scipy.sparse.csr_matrix]
              list of aligned scipy.sparse.csr_matrix with uniform shape.
        """
        # Create a mapping from label to index for rows and columns
        row_index = {label: i for i, label in enumerate(self.global_row_labels)}
        col_index = {label: i for i, label in enumerate(self.global_col_labels)}

        aligned = []  # Will hold the aligned matrices

        for matrix, local_rows, local_cols in self.matrices:
            coo = matrix.tocoo()  # Convert to coordinate format for easy iteration
            new_rows, new_cols, new_data = [], [], []

            # Map each value from local position to global position
            for i, j, v in zip(coo.row, coo.col, coo.data):
                global_i = row_index[local_rows[i]]
                global_j = col_index[local_cols[j]]
                new_rows.append(global_i)
                new_cols.append(global_j)
                new_data.append(v)

            shape = (len(self.global_row_labels), len(self.global_col_labels))
            aligned_matrix = sp.csr_matrix((new_data, (new_rows, new_cols)), shape=shape)
            aligned.append(aligned_matrix)

        return aligned

    def get_global_labels(self) -> Tuple[List[str], List[str]]:
        """
        Get the unified set of row and column labels.

        Returns
        -------
          - tuple: (global_row_labels, global_col_labels)
              global_row_labels: all unique row labels collected across added matrices
              global_col_labels: all unique column labels collected across added matrices
        """
        return self.global_row_labels, self.global_col_labels
    
    # -----------------------------
    # Internal Methods
    # ----------------------------

    def _update_union(self, overall: List[str], new_labels: List[str]) -> None:
        # Add new labels to the global list if they are not already present
        for label in new_labels:
            if label not in overall:
                overall.append(label)


    def _read_file(self, filepath: str) -> pd.DataFrame:
        """
        Reads a file into a pandas DataFrame.
        Supports CSV and TSV formats.

        Parameters
        ----------
          - filepath: str
              path to the input file

        Returns
        -------
          - pd.DataFrame:
              DataFrame with the file contents
        """
        _, ext = os.path.splitext(filepath.lower())
        sep = '\t' if ext in ['.tsv'] else ','  # Use tab for .tsv, comma for others

        try:
            return pd.read_csv(filepath, sep=sep, index_col=0)
        except Exception as e:
            raise ValueError(f"Failed to read file {filepath}: {e}")

    def _convert_to_sparse(self, data: Any) -> csr_matrix:
        """
        Converts various types of input data to a sparse CSR matrix.

        Parameters
        ----------
          - data: various types
              Can be:
                - scipy sparse matrix
                - numpy array
                - list of lists (dense)
                - edge list: [(i, j), ...] or [(i, j, value), ...]

        Returns
        -------
          - scipy.sparse.csr_matrix:
              The converted sparse CSR matrix
        """
        if sp.issparse(data):
            return data.tocsr()
        elif isinstance(data, np.ndarray):
            return sp.csr_matrix(data)
        elif isinstance(data, list):
            # Case 1: dense matrix format [[1, 2], [3, 4]]
            if all(isinstance(row, list) for row in data):
                return sp.csr_matrix(np.array(data))
            # Case 2: edge list format [(i, j), ...] or [(i, j, value), ...]
            elif all(isinstance(item, (tuple, list)) for item in data):
                rows, cols, vals = [], [], []
                for item in data:
                    if len(item) == 2:
                        i, j = item
                        v = 1  # default weight = 1
                    elif len(item) == 3:
                        i, j, v = item
                    else:
                        raise ValueError("Each element must be a tuple/list of 2 or 3 elements.")
                    rows.append(i)
                    cols.append(j)
                    vals.append(v)
                n_rows = max(rows) + 1
                n_cols = max(cols) + 1
                return sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        else:
            raise TypeError(f"Unsupported input type: {type(data)}")