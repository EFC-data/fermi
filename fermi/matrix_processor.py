import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Any, List, Tuple, Union
from scipy.sparse import csr_matrix, diags, issparse
from bicm import BipartiteGraph
import copy

class MatrixProcessorCA:
    """
    Combined processor for loading, aligning sparse matrices and computing comparative advantage (RCA, ICA).

    Stores original and processed matrices internally. All methods modify internal state.
    Call get_matrices() to retrieve the current processed matrices.
    """
    def __init__(self) -> None:
        # Storage for raw and processed matrices and their labels
        self._original: Tuple[csr_matrix, List[str], List[str]] = None
        self._processed: csr_matrix = None
        self.global_row_labels: List[str] = []
        self.global_col_labels: List[str] = []

    # -----------------------------
    # Loading & Alignment Methods
    # -----------------------------
    def load(
        self,
        input_data: Union[str, Path, pd.DataFrame, np.ndarray, List[Any]],
        **kwargs
    ):
        """
        Load input_data as sparse matrix, store original and initialize processed as a copy.
        """
        mat, rows, cols = self._load_full(input_data, **kwargs)
        # update global labels
        if rows:
            self.global_row_labels =  rows
        if cols:
            self.global_col_labels =  cols
        # store original and initial processed
        self._original = (mat, rows or [], cols or [])
        self._processed = mat.copy()
        return self

    def align_matrices(self, ) -> None: # deprecated
        """
        Align all processed matrices to shared global labels.
        """
        row_index = {lab: i for i, lab in enumerate(self.global_row_labels)}
        col_index = {lab: i for i, lab in enumerate(self.global_col_labels)}
        aligned = []
        for mat, rows, cols in self._original:
            coo = mat.tocoo()
            r, c, d = [], [], []
            for i, j, v in zip(coo.row, coo.col, coo.data):
                r.append(row_index[rows[i]])
                c.append(col_index[cols[j]])
                d.append(v)
            shape = (len(self.global_row_labels), len(self.global_col_labels))
            aligned.append(sp.csr_matrix((d, (r, c)), shape=shape))
        self._processed = aligned
        return self

    def copy(self):  # ok with sparse
        """
        Copy routine
        :return: return the hard copy of the efc class
        """
        return copy.deepcopy(self)

    # -----------------------------
    # Comparative Advantage Methods
    # -----------------------------
    def compute_rca(self) -> None:
        """
        Replace each processed matrix with its RCA version.
        """
        mat = self._processed
        val = np.sqrt(mat.sum().sum())
        s0 = np.divide(val, mat.sum(0), where=mat.sum(0) > 0)
        s1 = np.divide(val, mat.sum(1), where=mat.sum(1) > 0)
        rca = mat.multiply(s0).multiply(s1)
        self._processed = rca.tocsr()
        return self

    def compute_ica(self) -> None:
        """
        Replace each processed matrix with its ICA version.
        """
        mat = self._processed
        # check rows or columns zeros
        row_sums = np.array(mat.sum(axis=1)).ravel()
        col_sums = np.array(mat.sum(axis=0)).ravel()
        row_mask = row_sums != 0
        col_mask = col_sums != 0
        submat = mat[row_mask][:, col_mask].tocsr()

        # compute the ica
        graph = BipartiteGraph()
        graph.set_biadjacency_matrix(submat)
        graph.solve_tool(linsearch=True, verbose=False, print_error=False, model='biwcm_c')
        avg = graph.avg_mat
        inv_avg = np.divide(np.ones_like(avg), avg, where=avg > 0)
        inv_avg[inv_avg == np.inf] = 0
        ica_sub = submat.multiply(sp.csr_matrix(inv_avg))

        # restore the original dimensions
        coo = ica_sub.tocoo()
        orig_rows = np.nonzero(row_mask)[0][coo.row]
        orig_cols = np.nonzero(col_mask)[0][coo.col]
        ica = csr_matrix((coo.data, (orig_rows, orig_cols)), shape=mat.shape)

        # append
        self._processed = ica
        return self

    # -----------------------------
    # Binarization
    # -----------------------------
    def binarize(self, threshold: float = 1) -> None:
        """
        Binarize each processed matrix in-place with given threshold.
        """
        mat = self._processed
        result = mat.tocsr()
        result.data = np.where(result.data >= threshold, 1, 0)
        result.eliminate_zeros()
        self._processed = result
        return self

    # -----------------------------
    # Accessor
    # -----------------------------
    def get_matrix(self) -> csr_matrix:
        """
        Return the list of current processed matrices.
        """
        return self._processed

    # -----------------------------
    # Internal Loading Helpers
    # -----------------------------
    def _load_full(self, input_data, **kwargs) -> Tuple[csr_matrix, List[str], List[str]]:
        # identify and load input, returning matrix and labels
        if isinstance(input_data, (str, Path)):
            return self._load_from_path(Path(input_data), **kwargs)
        if isinstance(input_data, pd.DataFrame):
            return self._load_from_dataframe(input_data)
        mat = self._load_from_other(input_data)
        return mat, [], []

    def _load_from_path(self, path: Path, **kwargs):
        path = path.resolve()
        ext = path.suffix.lower()
        dict_ext = {'.csv':',', '.tsv':'\t', '.dat':' '}
        if ext in ['.csv', '.tsv', '.dat', '.txt']:
            if 'sep' not in kwargs:
                kwargs['sep'] = dict_ext[ext]
            if kwargs.get('header', 0)==0 and ext in ['.dat', '.txt']:
                kwargs['header'] = None
            df = pd.read_csv(path, **kwargs)
            return self._load_from_dataframe(df)
        if ext in ['.xlsx','..xls']:
            df = pd.read_excel(path, **kwargs)
            return self._load_from_dataframe(df)
        if ext in ['.mtx','.mm']:
            from scipy.io import mmread
            mat = mmread(str(path))
            return mat.tocsr(), [], []
        if ext in ['.npz']:
            mat = sp.load_npz(path, **kwargs)
            return mat.tocsr(), [], []
        if ext in ['.npy']:
            arr = np.load(path, **kwargs)
            mat = arr if sp.issparse(arr) else sp.csr_matrix(arr)
            return mat, [], []
        raise ValueError(f"Unrecognized format: {ext}")

    def _load_from_dataframe(self, df: pd.DataFrame) -> Tuple[csr_matrix, List[str], List[str]]:
        rows = df.index.tolist() if not df.index.equals(pd.RangeIndex(len(df))) else []
        cols = df.columns.tolist() if not df.columns.equals(pd.RangeIndex(len(df.columns))) else []
        return sp.csr_matrix(df.values), rows, cols

    def _load_from_other(self, obj: Any) -> csr_matrix:
        if sp.issparse(obj):
            return obj.tocsr()
        arr = np.array(obj)
        if arr.ndim == 2:
            return sp.csr_matrix(arr)
        if isinstance(obj, list) and all(isinstance(el,(tuple,list)) for el in obj):
            rows, cols, vals = [], [], []
            for el in obj:
                i,j = el[:2]
                v = el[2] if len(el)==3 else 1
                rows.append(i); cols.append(j); vals.append(v)
            shape=(max(rows)+1, max(cols)+1)
            return sp.csr_matrix((vals,(rows,cols)), shape=shape)
        raise TypeError(f"Unsupported input type: {type(obj)}")

