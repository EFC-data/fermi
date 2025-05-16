import numpy as np
from scipy.sparse import diags, issparse, csr_matrix, spmatrix
from bicm import BipartiteGraph
from typing import Optional, Union

class ComparativeAdvantage:
    def __init__(self, mat: spmatrix, metric: str = 'rca') -> None:
        """
        Initializes the ComparativeAdvantage class with a sparse input matrix and a chosen metric.
        
        Parameters
        -----------------
          - mat: spmatrix  
              Input sparse matrix (e.g., a SciPy sparse matrix)

          - metric: str  
              The validation metric to use ('rca' or 'ica'). Default is 'rca'
        """
        self.mat = mat                      # Input sparse matrix
        self.metric = metric.lower()        # Chosen metric, should be 'rca' or 'ica'
        self.metric_matrix = None           # Will hold the computed metric matrix
        self.compute_metric()               # Compute the chosen metric at initialization

    def compute_rca(self) -> spmatrix:
        """
        Computes the Revealed Comparative Advantage (RCA) matrix for sparse input using a vectorized approach.
        RCA[i, j] = (mat[i, j] / row_sum[i]) * (total_sum / col_sum[j])

        The method computes row and column sums, uses safe versions to avoid division by zero,
        scales rows and columns via sparse multiplication, and finally restores zero rows/columns.

        Returns
        -------------
          - RCA: spmatrix  
              The computed RCA matrix as a sparse matrix
        """
        # Compute row sums and column sums as 1D numpy arrays.
        rows_sum = np.array(self.mat.sum(axis=1)).ravel()
        cols_sum = np.array(self.mat.sum(axis=0)).ravel()
        total_sum = rows_sum.sum()

        # Replace zero sums with 1 to avoid division by zero.
        safe_rows_sum = np.where(rows_sum == 0, 1, rows_sum)
        safe_cols_sum = np.where(cols_sum == 0, 1, cols_sum)

        # Scale each row by dividing by its safe sum.
        rca_temp = self.mat.multiply(1 / safe_rows_sum[:, None])
        # Create a diagonal matrix to scale each column.
        col_scaling = total_sum / safe_cols_sum
        D = diags(col_scaling)
        # Final RCA is the product of the row-scaled matrix and the column scaling diagonal matrix.
        RCA = rca_temp.dot(D)

        # Ensure that rows (or columns) with original zero sum remain zero.
        RCA = RCA.tocsr()
        for i, rsum in enumerate(rows_sum):
            if rsum == 0:
                RCA[i, :] = 0
        RCA = RCA.tocsc()
        for j, csum in enumerate(cols_sum):
            if csum == 0:
                RCA[:, j] = 0

        return RCA

    def compute_ica(self) -> spmatrix:
        """
        Computes the Inferred Compartive Advantage (ICA) matrix using the BipartiteGraph class.
        ICA is computed as the element-wise division: ICA = mat / BICM.

        To handle division by zero, the BICM matrix is converted to a dense array and zeros are
        replaced temporarily with 1 during division, then restored to 0.

        Returns
        -------------
          - ICA: spmatrix  
              The computed ICA matrix as a sparse matrix
        """
        myGraph = BipartiteGraph()
        myGraph.set_biadjacency_matrix(self.mat)
        BICM = myGraph.get_bicm_matrix()  # Expected to be a sparse matrix

        # Convert to dense arrays for element-wise operations.
        mat_dense = self.mat.toarray()
        bicm_dense = BICM.toarray() if issparse(BICM) else np.array(BICM)

        # Replace zeros in BICM to avoid division by zero.
        safe_bicm_dense = np.where(bicm_dense == 0, 1, bicm_dense)
        ica_dense = mat_dense / safe_bicm_dense
        # Reset values to zero where BICM was originally zero.
        ica_dense[bicm_dense == 0] = 0

        return csr_matrix(ica_dense)

    def compute_metric(self) -> Union[spmatrix, np.ndarray]:
        """
        Computes and stores the chosen metric (either RCA or ICA) in self.metric_matrix.

        Returns
        -------------
          - metric_matrix: spmatrix or ndarray  
              The computed metric matrix
        """
        if self.metric == 'rca':
            self.metric_matrix = self.compute_rca()
        elif self.metric == 'ica':
            self.metric_matrix = self.compute_ica()
        else:
            raise ValueError("Invalid metric: choose 'rca' or 'ica'")
        return self.metric_matrix

    def binarize(self, threshold: float) -> Union[spmatrix, np.ndarray]:
        """
        Binarizes the precomputed metric matrix using the provided threshold.
        For each element, if its value is greater than or equal to the threshold, it is set to 1; otherwise, 0.

        For sparse matrices, the binarization is performed on the nonzero data to preserve sparsity.

        Parameters
        -----------------
          - threshold: float  
              The threshold value for binarization

        Returns
        -------------
          - binarized_matrix: spmatrix or ndarray  
              The binarized matrix (sparse if the metric matrix is sparse)
        """
        if self.metric_matrix is None:
            self.compute_metric()

        if issparse(self.metric_matrix):
            matrix_csr = self.metric_matrix.tocsr()  # Ensure efficient row operations
            matrix_csr.data = np.where(matrix_csr.data >= threshold, 1, 0)
            return matrix_csr
        else:
            # For a dense matrix, use numpy's vectorized operation.
            return np.where(self.metric_matrix >= threshold, 1, 0)