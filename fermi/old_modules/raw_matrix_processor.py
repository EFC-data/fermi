import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pathlib import Path
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

    def load_as_sparse(
        self,
        input_data: Union[str, Path, pd.DataFrame, np.ndarray, List[Any]],
        return_labels: bool = False,
        **kwargs
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
        if isinstance(input_data, (str, Path)):
            # If input is a file path
            print(f"Loading input file {input_data} (str or Path)")
            mat, row_labels, col_labels = self._load_from_path(Path(input_data), **kwargs)
        elif isinstance(input_data, pd.DataFrame):
            # If input is already a DataFrame
            print(f"Loading input file {input_data} (pd.Dataframe)")
            mat, row_labels, col_labels = self._load_from_dataframe(input_data, **kwargs)
        else:
            # For other types (numpy arrays, edge lists, etc.)
            print(f"Loading input file {input_data} (neither str nor Path nor pd.Dataframe but others)")
            mat = self._load_from_other(input_data, **kwargs)
            if return_labels:
                raise ValueError("Cannot extract labels from this data type.")
            return mat

        if return_labels:
            return mat.tocsr(), row_labels, col_labels
        else:
            return mat.tocsr()

    def add_matrix(
        self,
        input_data: Any,
        row_labels: List[str],
        col_labels: List[str],
        **kwargs
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
        matrix = self.load_as_sparse(input_data, **kwargs)
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
    def _load_from_path(self, path, **kwargs):
        path = Path(path).resolve()

        ext = path.suffix.lower()
        stem = path.stem    # name without extension, not necessary for the moment
        row_labels = col_labels = None
        dict_ext = {'.csv':',', '.tsv':'\t', '.dat':' '}

        if ext in ['.csv', '.tsv', '.dat', '.txt']:
            if kwargs.get('sep', None) is None:
                kwargs['sep'] = dict_ext.get(ext, ',')
            df = pd.read_csv(path, **kwargs)
            mat, row_labels, col_labels = self._load_from_dataframe(df)

        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(path, **kwargs)
            mat, row_labels, col_labels = self._load_from_dataframe(df)

        elif ext in ['.mtx', '.mm']:
            from scipy.io import mmread
            mat = mmread(str(path), **kwargs)

        elif ext in ['.npy', '.npz']:
            arr = np.load(path, **kwargs)
            if sp.issparse(arr):
                mat = arr
            else:
                mat = sp.csr_matrix(arr, **kwargs)

        else:
            raise ValueError(f"Format file not recognized: {ext}")
        
        return mat, row_labels, col_labels
    
    def _load_from_dataframe(self, df: pd.DataFrame):
        # If index/columns labels are present, store them
        if not df.index.equals(pd.RangeIndex(start=0, stop=len(df))):
            row_labels = df.index.tolist()
        else:
            row_labels = None

        if not df.columns.equals(pd.RangeIndex(start=0, stop=len(df.columns))):
            col_labels = df.columns.tolist()
        else:
            col_labels = None

        mat = sp.csr_matrix(df.values)
        return mat, row_labels, col_labels
         
    def _load_from_other(self, obj):
        """
        Converts various types of input data to a sparse CSR matrix.

        Parameters
        ----------
          - obj: various types
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
        if isinstance(obj, (np.ndarray, np.matrix)):
            return sp.csr_matrix(obj)
        elif sp.issparse(obj):
            return obj.tocsr()
        elif isinstance(obj, list):
            # Check if it's a dense matrix or an edge list
            if all(isinstance(row, (list, np.ndarray)) for row in obj):
                return sp.csr_matrix(np.array(obj))
            elif all(isinstance(item, (tuple, list)) for item in obj):
                # Edge list: [(i, j), ...] or [(i, j, weight), ...]
                rows, cols, vals = [], [], []
                for item in obj:
                    if len(item) == 2:
                        i, j = item
                        v = 1        # default weight = 1
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
            raise TypeError(f"Unsupported input type: {type(obj)}")

    def _update_union(self, overall: List[str], new_labels: List[str]) -> None:
        # Add new labels to the global list if they are not already present
        for label in new_labels:
            if label not in overall:
                overall.append(label)
