import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Union
from pathlib import Path
from scipy.sparse import csr_matrix, diags, vstack, hstack, issparse
from scipy.sparse.linalg import eigs
from fermi.matrix_processor import MatrixProcessorCA

class efc(MatrixProcessorCA):
    """
    This class implements the core methods for computing Fitness and Complexity metrics
    in economic complexity analysis from a binary (typically sparse) matrix.

    Main functionalities include:
    - Fitness and Complexity computation (Tacchella-2012, Servedio-2018)
    - Economic Complexity Index (ECI) and Product Complexity Index (PCI)
    - Degree-based metrics: Diversification (country degree) and Ubiquity (product degree)
    - Structural metrics: Density and NODF (nestedness)
    - Matrix visualization with customizable sorting for comparative analysis

    The class supports sparse matrices, configurable initial conditions, and row/column labeling.
    """

    def __init__(self,
            input_data: Union[MatrixProcessorCA, str, Path, pd.DataFrame, np.ndarray, List[Any]] = None,
            **kwargs) -> None:
        """
        Inizializes the efc class with a binary country-product matrix.

        Parameters
        ----------
          - matrix : csr_matrix
              Binary country-product matrix (sparse format).
          - hardcopy : bool, default True
              If True, matrix is copied to avoid side effects.
          - global_row_labels : list, optional
              List of row labels (e.g. country names).
          - global_col_labels : list, optional
              List of column labels (e.g. product codes).

        """
        super().__init__()

        self.shape = None
        self._empty_metrics()

        if input_data is not None:
            if isinstance(input_data, MatrixProcessorCA):
                # shallow‐copy lists so you don’t accidentally share them
                self._original = (input_data._original[0].copy(), input_data._original[1].copy(), input_data._original[2].copy())
                self._processed = input_data._processed.copy()
                self.global_row_labels = input_data.global_row_labels
                self.global_col_labels = input_data.global_col_labels
                self.shape = self._processed.shape
                self._set_labels()

            else:
                self.load(input_data, **kwargs)

    def _empty_metrics(self):
        # Placeholders for fitness and complexity metrics
        self.fitness = None
        self.complexity = None

        # Placeholders for ubiquity and diversification
        self.ubiquity = None
        self.diversification = None

        # Placeholders for Economic Complexity Index (ECI) and Product Complexity Index (PCI)
        self.eci = None
        self.pci = None
        self.eci_eig = None
        self.pci_eig = None

        # Placeholders for structural metrics
        self.density = None
        self.nodf = None

    def _set_labels(self):
        if len(self.global_row_labels) == 0:
            self.global_row_labels = list(range(self.shape[0]))
        if len(self.global_col_labels) == 0:
            self.global_col_labels = list(range(self.shape[1]))

    def load(
            self,
            input_data: Union[str, Path, pd.DataFrame, np.ndarray, List[Any]],
            **kwargs
    ):
        super().load(input_data, **kwargs)

        self.shape = self._processed.shape
        self._empty_metrics()
        self._set_labels()

        return self

    def add_dummy(self, dummy_row: bool = True, dummy_col: bool = False, inplace: bool = False) -> "efc":
        """
        Adds a dummy country (row of 1s) and/or a dummy product (column of 1s) to the matrix.

        This operation sets a reference scale: the dummy country has maximum fitness,
        while the dummy product has minimum complexity. If inplace=False, returns a modified copy.

        Parameters
        ----------
          - dummy_row : bool
              Whether to add a dummy country (row of 1s).
          - dummy_col : bool
              Whether to add a dummy product (column of 1s).
          - inplace : bool
              If True, modifies the object in-place. If False, returns a new copy.
        Returns
        -------
          - efc
              Modified efc instance (self or a copy).
        """
        if not inplace:
            return self.copy().add_dummy(dummy_row=dummy_row, dummy_col=dummy_col, inplace=True)

        n_rows, n_cols = self.shape

        if dummy_row:
            dummy_row_vec = csr_matrix(np.ones((1, n_cols), dtype=self._processed.dtype))
            self._processed = vstack([self._processed, dummy_row_vec], format='csr')
            if self.global_row_labels is not None:
                self.global_row_labels = list(self.global_row_labels) + ['dummy_row']
            else:
                self.global_row_labels = [f'row_{i}' for i in range(n_rows)] + ['dummy_row']

        if dummy_col:
            new_n_rows = self.shape[0]
            dummy_col_vec = csr_matrix(np.ones((new_n_rows, 1), dtype=self._processed.dtype))
            self._processed = hstack([self._processed, dummy_col_vec], format='csr')
            if self.global_col_labels is not None:
                self.global_col_labels = list(self.global_col_labels) + ['dummy_col']
            else:
                self.global_col_labels = [f'col_{j}' for j in range(n_cols)] + ['dummy_col']

        self.shape = np.array(self._processed.shape)

        # Invalidate cached metrics
        self.fitness = None
        self.complexity = None
        self.eci = None
        self.pci = None
        self.diversification = None
        self.ubiquity = None

        return self

    ####################################
    ########  Internal Methods  ########
    ####################################
    @staticmethod
    def normalize(vector: np.ndarray | pd.Series, normalization: str = 'sum') -> np.ndarray | pd.Series:
        """
        Normalize a numeric vector or series using a a specified method.

        Parameters
        ----------
          - vector : np.ndarray or pd.Series
              The input array or series to normalize
          - normalization : str
              One of 'sum' (divide by sum), 'max' (divide by max),
                     'mean' (divide by mean), or 'zscore' (standard score).

        Returns
        -------
          - np.ndarray or pd.Series
              The Normalized array or series of the same type.
        """
        if normalization == 'sum':
            return vector / vector.sum(0)
        elif normalization == 'max':
            return vector / vector.max(0)
        elif normalization == 'mean':
            return vector / vector.mean(0)
        elif normalization == 'zscore':
            vec = (vector - vector.mean(0)) / vector.std(0)
            eps = np.finfo(float).eps
            vec[np.abs(vec) < eps] = 0.0
            return vec
        else:
            raise ValueError(
                f"Unknown normalization '{normalization}'. "
                "Choose from 'sum', 'max', 'mean', or 'zscore'."
            )

    def _compute_diversification_ubiquity(self, matrix) -> None:
        """
        Compute and store diversification and ubiquity vectors for the binary matrix.

        Diversification is defined as the number of products per country (row sums),
        and ubiquity as the number of countries per product (column sums).

        Results are stored in:
        diversification: numpy.ndarray of shape (n_countries,)
        ubiquity: numpy.ndarray of shape (n_products,)
        """
        diversification = np.array(matrix.sum(axis=1)).flatten()
        ubiquity = np.array(matrix.sum(axis=0)).flatten()

        return diversification, ubiquity

    def _eci_pci_indices(self, matrix, method: str = 'reflections', norm: str = 'zscore',
                         max_iterations: int = 18, eigv: bool = False,
                         verbose: bool = False) -> tuple:
        """
        Compute the Economic Complexity Index (ECI) and Product Complexity Index (PCI)
        using either the 'reflections' method or the 'spectral' method.

        Parameters
        ----------
          - method : str
              Either 'reflections' for iterative method or 'spectral' for eigenvector-based method
          - norm : str
              Normalization to apply to ECI/PCI ('sum', 'mean', etc.).
          - max_iterations : int
              Number of iterations to run for reflections method.
          - eigv : bool
              If True and using 'spectral', also return eigenvectors.
          - verbose : bool
              If True, print debugging info (only for 'spectral').

        Returns
        -------
          - tuple
              (eci, pci) or (eci, eigvecs_c, pci, eigvecs_p) if eigv=True.
        """
        if not isinstance(matrix, csr_matrix):
            raise TypeError("matrix must be CSR for ECI/PCI computation")
        if method == 'reflections':
            return self._method_of_reflections(matrix, norm, max_iterations)
        elif method == 'spectral':
            return self._eci_pci_from_eigs(matrix, norm, eigv, verbose)
        else:
            raise ValueError(f"Unsupported method '{method}' choose 'reflections' or 'spectral'")

    def _method_of_reflections(self, matrix, norm: str = 'zscore', max_iterations: int = 10) -> tuple:
        """
        Internal method that computes ECI and PCI using the Method of Reflections
        introduced by Hidalgo & Hausmann (2009).

        The algorithm iteratively updates:
        - kc (Economic Complexity Index, ECI): a complexity score for countries
        - kp (Product Complexity Index, PCI):  a complexity score for products

        Parameters
        ----------
          - norm : str
              normalization of kc and kp (default = zscore).
          - max_iterations : int
              Number of iterations to run (default = 18 as in H & H paper).

        Returns
        -------
          - tuple
              Normalized vectors: (ECI, PCI).

        Reference
        ---------
          - Hidalgo C. and Hausmann R., *The building blocks of economic complexity*, PNAS 26 (2009)
        """
        Mcp = matrix
        Mpc = Mcp.transpose()

        # Compute initial diversification and ubiquity with dedicated method
        diversification, ubiquity = self._compute_diversification_ubiquity(Mcp)

        kc0 = diversification.astype(float)  # Initial country complexity
        kp0 = ubiquity.astype(float)  # Initial product complexity

        # Set the initial condition
        total_kc0 = kc0.sum() or 1.0
        total_kp0 = kp0.sum() or 1.0
        kc = kc0 / total_kc0
        kp = kp0 / total_kp0

        # Iterative update following Method of Reflections
        for _ in range(max_iterations):
            with np.errstate(divide='ignore', invalid='ignore'):
                # Update country complexity
                kc_new = (Mcp @ kp) / np.where(kc0 != 0, kc0, 1)
                # Update product complexity
                kp_new = (Mpc @ kc) / np.where(kp0 != 0, kp0, 1)
                kc, kp = kc_new, kp_new

        return self.normalize(kc, norm), self.normalize(kp, norm)

    def _eci_pci_from_eigs(self, matrix, norm: str = 'zscore', eigv: bool = False, verbose: bool = False) -> tuple:
        """
        Compute ECI and PCI based on the spectral method introduced by Cristelli et al. (2013).

        This method uses the second-largest eigenvectors of the country–country (Mcc)
        and product–product (Mpp) projection matrices derived from the normalized
        country-product matrix.

        Parameters
        ----------
          - norm : str
              Normalization type for the output vectors.
          - eigv : bool
              If True, return eigenvectors associated with ECI and PCI too.
          - verbose : bool
              If True, print debug info.

        Returns
        -------
          - tuple
              (ECI, PCI) or (ECI, eigvec_c, PCI, eigvec_p) f eigv is True

        Reference
        ----------
          - Cristelli M. et al.,
            *Measuring the Intangibles: A Metrics for the Economic Complexity of Countries and Products*, PLoS ONE 8(8), 2013.
        """
        Mcp = matrix  # expected shape: (n_countries, n_products)

        diversification, ubiquity = self._compute_diversification_ubiquity(Mcp)

        # Normalize Mcp by diversification (rows) to get Pcp
        div = diversification.astype(float)
        with np.errstate(divide='ignore'):
            inverse_div = diags(np.where(div != 0, 1.0 / div, 0.0))
        Pcp = inverse_div.dot(Mcp)  # shape: (n_countries, n_products)

        # Normalize Mcp by ubiquity (columns) to get Ppc
        ubi = ubiquity.astype(float)
        with np.errstate(divide='ignore'):
            inverse_ubi = diags(np.where(ubi != 0, 1.0 / ubi, 0.0))
        Mpc = Mcp.transpose()  # shape: (n_products, n_countries)
        Ppc = inverse_ubi.dot(Mpc)  # normalize rows = columns of Mcp

        # Compute projection matrices
        Mcc = Pcp.dot(Ppc)  # country–country: (n_countries × n_countries)
        Mpp = Ppc.dot(Pcp)  # product–product: (n_products × n_products)

        if verbose:
            print("Ppc shape:", Ppc.shape, "\n")
            print(pd.DataFrame(Ppc.toarray()), "\n")
            print("Pcp:", Pcp.shape, "\n")
            print(pd.DataFrame(Pcp.toarray()), "\n")
            print("Mcc shape:", Mcc.shape, "\n")
            print(pd.DataFrame(Mcc.toarray()), "\n")
            print("Mpp:", Mpp.shape, "\n")
            print(pd.DataFrame(Mpp.toarray()), "\n")

        # ECI: second eigenvector of Mcc
        try:
            eigvals_c, eigvecs_c = eigs(Mcc, k=2, which='LR')  # 'LR' = Largest Real part
        except Exception as e:
            raise RuntimeError(f"Eigs Mcc failed: {e}")
        eci = np.real(eigvecs_c[:, 1])  # Second eigenvector
        if np.corrcoef(eci, div)[0, 1] < 0:
            eci *= -1
        eci = self.normalize(eci, norm)

        # PCI: second eigenvector of Mpp
        try:
            eigvals_p, eigvecs_p = eigs(Mpp, k=2, which='LR')
        except Exception as e:
            raise RuntimeError(f"Eigs Mpp failed: {e}")
        pci = np.real(eigvecs_p[:, 1])
        if np.corrcoef(pci, ubi)[0, 1] < 0:
            pci *= -1
        pci = self.normalize(pci, norm)

        if eigv:
            return eci, eigvecs_c, pci, eigvecs_p

        return eci, pci

    @staticmethod
    def _minimum_crossing_time(vector: pd.DataFrame, iterat: int, tail: int) -> float:
        """
        Estimate the minimum crossing time of the fitness or complexity ranks based on their recent growth.
        This function analyzes how quickly elements in a ranked vector are converging and predicts
        the soonest iteration in which a lower-ranked element may overtake a higher-ranked one.

        Parameters
        ----------
          - vector : pd.DataFrame
              Fitness or complexity values across recent iterations (rolling buffer).
              Shape must be (n, tail), where tail is the rolling window length.
          - iterat : int
              Current iteration index.
          - tail : int
              Number of columns in the buffer (rolling history).

        Returns
        -------
          - float
              Minimum estimated time to next rank inversion.

        Reference
        ---------
          - Pugliese E. et al.,
            *On the convergence of the Fitness-Complexity algorithm*, Eur. Phys. J. Spec. Top. 225, 1893–1911 (2016).
        """
        # Determine current and next column positions in the buffer (modulo 'tail')
        newpos = iterat + 1
        col_now = iterat % tail
        col_next = newpos % tail

        # Select current and next values
        vec_now = vector.iloc[:, col_now]
        vec_next = vector.iloc[:, col_next]

        # Estimate growth rate for each element between iterations t and t+1 (logarithmic scale)
        growth = (np.log(vec_next) - np.log(vec_now)) / np.log(newpos / iterat)

        # Create DataFrame of values and growth, sorted by current value (to analyze rank changes)
        rank_df = pd.DataFrame({'value': vec_next, 'growth': growth}, index=vector.index).sort_values('value')

        # Compute difference in growth between adjacent ranks (row i and i+1)
        rank_shift = rank_df.shift(-1)
        rank_diff = rank_shift[['growth']] - rank_df[['growth']]  # Δgrowth = g_{i+1} - g_i

        # Compute current value ratio between adjacent elements: v_i / v_{i+1}
        rank_rat = rank_df[['value']] / rank_shift[['value']]

        # Rename the column to 'value' so it matches the structure of 'rat'
        # and allows aligned filtering and indexing in the final step
        rank_diff.columns = ['value']

        # Estimate time to crossing for each pair:
        # T_cross = [(v_i / v_{i+1}) * (t+1)^Δgrowth]^(1/Δgrowth)
        rat = np.power(np.power(newpos, rank_diff) * rank_rat, 1.0 / rank_diff)

        # Return the minimum crossing time among pairs where a lower-ranked item is growing faster
        return rat[rank_diff < 0]['value'].min()

    def _fitness_complexity(self,
                            bin_rca: csr_matrix,
                            method: str = 'tacchella',
                            max_iteration: int = 1000,
                            check_stop: str = 'distance',
                            min_distance: float = 1e-14,
                            normalization: str = 'sum',
                            fit_ic: list | np.ndarray | None = None,
                            com_ic: list | np.ndarray | None = None,
                            removelowdegrees: int | float | bool = False,
                            verbose: bool = False,
                            redundant: bool = False,
                            delta: float = 1.0,  # used only in 'servedio' method
                            with_dummy: bool = False
                            ) -> tuple:
        """
        Compute country Fitness and product Complexity using the iterative algorithm proposed by
        Tacchella et al. (2012) or its Servedio et al. (2018) variant.

        The algorithm uses a bipartite country-product matrix and updates the Fitness (F)
        and Complexity (Q) values iteratively until convergence.

        Parameters
        ----------
          - bin_rca : csr_matrix
              Binary input matrix (country-product).
          - method : str
              'tacchella' or 'servedio' variant for the algorithm.
          - max_iteration : int
              Max iterations to run.
          - check_stop : str
              Stopping criterion ('distance' or 'crossing time').
          - min_distance : float
              Threshold for L1 convergence (if check_stop='distance').
          - normalization : str
              Normalization for fitness and complexity.
          - fit_ic : list or np.ndarray, optional
              Initial fitness vector.
          - com_ic : list or np.ndarray, optional
              Initial complexity vector.
          - removelowdegrees : int | float | bool
              If int or float, removes columns (products) with degree <= threshold
          - verbose : bool
              Print debug info.
          - redundant : bool
              Use current Q to immediately update F (Sinkhorn-like).
          - delta : float
              Used only in Servedio variant.
          - with_dummy : bool
              If True, rescales such that the dummy country's fitness is 1.

        Returns
        -------
          - tuple
              Final fitness and complexity vectors at convergence.

        Reference
        ---------
          - Tacchella A. et al.,
            *A New Metrics for Countries' Fitness and Products' Complexity*, SciRep vol. 2, 723 (2012)
          - Servedio V. D. P. et al.,
            *A New and Stable Estimation Method of Country Economic Fitness and Product Complexity*, Entropy 2018, 20(10), 783;
        """

        dim = bin_rca.shape
        d_row = dim[0] - 1
        index = np.arange(dim[0])
        columns = np.arange(dim[1])

        if verbose:
            print('# sparse matrix ({},{})'.format(dim[0], dim[1]))

        # Remove columns with degree lower than threshold using sparse masking
        if removelowdegrees:
            # Compute column degrees
            col_degree = np.array(bin_rca.sum(axis=0)).flatten()
            if isinstance(removelowdegrees, (int, float)):
                mask = col_degree > removelowdegrees
            else:
                mask = np.ones_like(col_degree, dtype=bool)
            D = diags(mask.astype(int))  # diagonal mask: 1=keep, 0=zero-out
            bin_rca = bin_rca.dot(D)

        bin_rca_t = bin_rca.transpose()

        # Tail determines how many past iterations are stored (rolling buffer)
        tail = 3 if check_stop == 'crossing time' else 2

        # Initialization of fitness and complexity to an array filled with 1's
        fit = pd.DataFrame(np.ones((dim[0], tail)), index=index, columns=range(tail), dtype=np.longdouble)
        com = pd.DataFrame(np.ones((dim[1], tail)), index=columns, columns=range(tail), dtype=np.longdouble)

        # Apply user-provided initial conditions
        if fit_ic is not None and len(fit_ic) == dim[0]:
            fit[0] = fit_ic
        if com_ic is not None and len(com_ic) == dim[1]:
            com[0] = com_ic

        fit = self.normalize(fit, normalization)
        com = self.normalize(com, normalization)

        ones_row = np.ones(dim[0])
        ones_col = np.ones(dim[1])

        # Main iterative loop
        for iterat in range(max_iteration):

            # colpos selects which past iteration to read from (rolling buffer)
            colpos = iterat % tail

            # Compute 1 / F_c^{(n-1)} for complexity update
            fit_here = np.zeros(dim[0])
            np.divide(ones_row, fit[colpos], out=fit_here, where=fit[colpos] != 0)
            fit_here[fit_here == np.inf] = 0.0

            # Q-update: either Tacchella or Servedio
            if method == 'servedio':
                com_here = 1.0 + bin_rca_t.dot(fit_here)  # <-- +1 shift in Servedio
            else:
                # Standard update: Q = 1 / sum_c (M_cp / F_c)
                com_here = bin_rca_t.dot(fit_here)  # Tacchella

            # Invert Q to compute F
            np.divide(ones_col, com_here, out=com_here, where=com_here != 0)

            if redundant:
                fit_here = bin_rca.dot(com_here)
            else:
                fit_here = bin_rca.dot(com[colpos])

            if method == 'servedio':
                fit_here = fit_here + delta  # <-- + delta shift in Servedio

            # enforce dummy scaling: dummy country always F=1
            if with_dummy and method != 'servedio':
                dummy_val = fit_here[d_row] if fit_here[d_row] != 0 else 1.0
                fit_here = fit_here / dummy_val

            # Store next iteration (rolling buffer)
            newpos = (iterat + 1) % tail
            fit[newpos] = self.normalize(fit_here, normalization)
            com[newpos] = self.normalize(com_here, normalization)

            # Check convergence condition
            if check_stop == 'crossing time':
                # Evaluate every 8 iterations after the first 10% of total
                if (iterat % 8 == 7) and (iterat > max_iteration // 10):
                    T = self._minimum_crossing_time(fit, iterat, tail)
                    if T + iterat + 1 > max_iteration:  # if the next swap is expected after max_iteration stop
                        break

            elif check_stop == 'distance':
                distance = np.abs(fit[newpos] - fit[colpos]).sum()
                if verbose:
                    print("iteration:", iterat, "L1 convergence:", distance)
                if iterat > max_iteration // 10:
                    if distance < min_distance:
                        break

        if normalization == 'sum' and not with_dummy:
            return fit[newpos] * dim[0], com[newpos] * dim[1]

        return fit[newpos], com[newpos]
    
    def _NODF(self) -> float:
        """
        Compute the metric Nestedness Overlap and Decreasing Fill (NODF) of a binary matrix.

        Returns
        -------
          - NODF: float
              Scalar NODF value.
        """
        if issparse(self._processed):
            A = self._processed.toarray()
        N,M = A.shape
        
        # Row degrees k_i and row‐overlaps O_{ij}
        overlap_rows = A.dot(A.T)
        k_rows  = A.sum(axis=1)
        
        # Col degrees k_α and row‐overlaps O_{αβ}
        overlap_cols = A.T.dot(A)
        k_cols  = A.sum(axis=0)  

        # N^R = sum_{i,j : k_i > k_j > 0} (O_ij / k_j)
        i_r = k_rows.reshape(-1, 1)  # shape (N,1)
        j_r = k_rows.reshape(1, -1)  # shape (1,N)

        mask_r = (i_r > j_r) & (j_r > 0)       # only pairs with strictly larger degree
        N_R = np.sum(overlap_rows[mask_r] / j_r[mask_r])

        # N^C = sum_{α,β : k_α > k_β > 0} (O_αβ / k_β)
        i_c = k_cols.reshape(-1, 1)  # shape (M,1)
        j_c = k_cols.reshape(1, -1)  # shape (1,M) 

        mask_c = (i_c > j_c) & (j_c > 0)       # only pairs with strictly larger degree
        N_C = np.sum(overlap_cols[mask_c] / j_c[mask_c])

        denom_R = N * (N - 1) / 2
        denom_C = M * (M - 1) / 2

        NODF = (N_R / denom_R + N_C / denom_C) / 2.

        return NODF
    
    def _S_NODF(self) -> float:
        """
        Compute the metric Stable NODF (S-NODF) of a binary matrix.

        Returns
        -------
          - S_NODF: float
              Scalar S-NODF value.
        """

        if issparse(self._processed):
            A = self._processed.toarray()
                
        # Row degrees k_i and row‐overlaps O_{ij}
        k_r = A.sum(axis=1)                 
        O_r = A.dot(A.T) 
        N = A.shape[0]              

        # Col degrees k_α and row‐overlaps O_{αβ}
        k_c = A.sum(axis=0)
        O_c = A.T.dot(A)     
        M = A.shape[1]

        # Mask to avoid double counting: only i<j terms 
        triu_r = np.triu_indices(N, k=1)  # k = 1 not consider diagonal
        triu_c = np.triu_indices(M, k=1)  # k = 1 not consider diagonal

        # Matrix of minima for rows
        i_r = k_r.reshape(-1, 1)
        j_r = k_r.reshape(1, -1)
        min_r = np.minimum(i_r, j_r) 
        valid_r = min_r[triu_r] > 0

        # Matrix of minima for cols
        i_c = k_c.reshape(-1, 1)
        j_c = k_c.reshape(1, -1)
        min_c = np.minimum(i_c, j_c)
        valid_c = min_c[triu_c] > 0     

        eta_R = np.sum(O_r[triu_r][valid_r] / min_r[triu_r][valid_r])        
        eta_C = np.sum(O_c[triu_c][valid_c] / min_c[triu_c][valid_c])

        denom_R = N * (N - 1) / 2   
        denom_C = M * (M - 1) / 2

        S_NODF = (eta_R + eta_C) / (denom_R + denom_C)

        return S_NODF   

    ############################
    ########  Wrappers  ########
    ############################

    def get_fitness_complexity(self,
                               force: bool = False,
                               aspandas: bool = False,
                               **kwargs) -> tuple[np.ndarray, np.ndarray] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute and cache fitness and complexity vectors.

        Parameters
        ----------
          - force : bool, default False
              If True, forces recomputation of fitness and complexity even if cached.
          - aspandas : bool, default False
              If True, returns results as pandas DataFrames with appropriate labels.
          - **kwargs : dict
              Additional keyword arguments passed to the internal _fitness_complexity() method.

        Returns
        -------
          - tuple of numpy.ndarray or tuple of pd.DataFrame
              Fitness and complexity vectors, as arrays or DataFrames depending on aspandas
        """
        # intercept force
        if self.fitness is None or self.complexity is None or force:
            # pass only the real args to the internal routine
            fit, com = self._fitness_complexity(bin_rca=self._processed, **kwargs)
            self.fitness, self.complexity = fit.to_numpy(), com.to_numpy()

        if aspandas:
            fit_df = pd.DataFrame(self.fitness, index=self.global_row_labels, columns=["fitness"])
            com_df = pd.DataFrame(self.complexity, index=self.global_col_labels, columns=["complexity"])
            return fit_df, com_df

        return self.fitness, self.complexity

    def get_exogenous_fitness(
            self,
            Q: np.ndarray,
            sp_matrix: csr_matrix | None = None,
            norm: str = 'sum',
            s_labels: list | None = None,
            aspandas: bool = False
            ) -> np.ndarray | pd.Series:
        """
        Compute exogenous fitness of subnational units using externally provided product complexities.

        Parameters
        ----------
          - Q : np.ndarray
              Complexity vector for products (1D array).
          - sp_matrix : csr_matrix, optional
              Sparse matrix mapping regions to products. If None, use self._processed.
          - norm : str, default 'sum'
              Normalization method for fitness ('sum', 'mean', etc.).
          - s_labels : list, optional
              List of region/state labels to use as index if aspandas=True.
          - aspandas : bool, default False
              If True, return result as pandas DataFrame.

        Returns
        -------
          - np.ndarray or pd.DataFrame
              Normalized exogenous fitness vector.
        """
        sp_matrix = self._processed if sp_matrix is None else sp_matrix
        Msp = sp_matrix if issparse(sp_matrix) else csr_matrix(sp_matrix)

        F_s = Msp.dot(Q)
        F_exog = self.normalize(F_s, norm)

        if aspandas:
            labels = s_labels if s_labels is not None else list(range(Msp.shape[0]))
            return pd.DataFrame(F_exog.ravel(), index=labels, columns=['exogenous fitness'])
        return F_exog

    def get_eci_pci(
        self,
        method: str = 'eigenvalue',
        norm: str = 'zscore',
        max_iterations: int = 18,
        eigv: bool = False,
        verbose: bool = False,
        force: bool = False,
        aspandas: bool = False
    ) -> Union[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Compute ECI and PCI (and optionally their eigenvectors).

        Parameters
        ----------
          - method : {'reflections', 'eigenvalue'}
              Choice of algorithm.
          - norm : str
              How to normalize the output vectors.
          - max_iterations : int
              Only for 'reflections'.
          - eigv : bool
              Only for 'spectral': if True, also return eigenvectors.
          - verbose : bool
              Only for 'spectral': print debug info.
          - force : bool
              If True, recompute even if cached.
          - aspandas : bool
              If True, return pandas DataFrames instead of numpy arrays.

        Returns
        -------
          - tuple: (eci, pci)
              Two arrays if eigv=False.
          - tuple: (eci, eci_eig, pci, pci_eig)
              Four arrays if eigv=True and method='spectral'.
        """
        # Recompute if needed
        if self.eci is None or self.pci is None or force:
            if method == 'reflections':
                # pass only the args relevant to reflections
                self.eci, self.pci = self._eci_pci_indices(
                    self._processed,
                    method=method,
                    norm=norm,
                    max_iterations=max_iterations
                )
            else:  # spectral
                if eigv:
                    self.eci, self.eci_eig, self.pci, self.pci_eig = self._eci_pci_indices(
                        self._processed,
                        method=method,
                        norm=norm,
                        eigv=eigv,
                        verbose=verbose
                    )
                else:
                    self.eci, self.pci = self._eci_pci_indices(
                        self._processed,
                        method=method,
                        norm=norm,
                        eigv=eigv,
                        verbose=verbose
                    )

        # Optionally wrap in pandas
        if aspandas:
            eci_df = pd.DataFrame(self.eci, index=self.global_row_labels, columns=["ECI"])
            pci_df = pd.DataFrame(self.pci, index=self.global_col_labels, columns=["PCI"])
            return eci_df, pci_df

        # Return numpy arrays (2 or 4)
        if method == 'spectral' and eigv:
            return self.eci, self.eci_eig, self.pci, self.pci_eig
        else:
            return self.eci, self.pci

    def get_diversification_ubiquity(self, force: bool = False, aspandas: bool = False, **kwargs) -> tuple:
        """
        Compute and optionally return diversification and ubiquity vectors.

        Parameters
        ----------
          - force : bool, default False
              If True, forces recomputation even if already cached.
          - aspandas : bool, default False
              If True, returns results as pandas DataFrames.
          - **kwargs : dict
              Additional parameters forwarded to _eci_pci_indices().

        Returns
        -------
          - tuple
              diversification and ubiquity vectors as (np.ndarray, np.ndarray) or (pd.DataFrame, pd.DataFrame).
        """
        if self.diversification is None or self.ubiquity is None or force:
            self.diversification, self.ubiquity = self._compute_diversification_ubiquity(self._processed, **kwargs)

        if aspandas:
            div = pd.DataFrame(self.diversification, index=self.global_row_labels, columns=["diversification"])
            ubi = pd.DataFrame(self.ubiquity, index=self.global_col_labels, columns=["ubiquity"])
            return div, ubi

        return self.diversification, self.ubiquity

    def get_density(self) -> float:
        """
        Computes the density of the binary matrix.

        Density is defined as the ratio of non-zero entries (typically 1s)
        to the total number of elements in the matrix..

        Parameters
        ----------
        None

        Returns
        -------
          - float
              Scalar density value between 0 and 1.
        """
        if self.density is None:
            self.density = self._processed.nnz / (self.shape[0] * self.shape[1])
        return self.density
    
    def get_nodf(self, force: bool = False) -> float:
        """
        Computes the Nestedness metric (NODF) of the binary matrix.

        NODF is a measure of how nested the matrix is, indicating how well
        the presence of certain elements in rows and columns correlates.

        Parameters
        ----------
        None

        Returns
        -------
          - float
              Scalar NODF value.
        """
        if self.nodf is None or force:
            self.nodf = _nodf(self._processed)
        return self.nodf

    def plot_matrix(self,
                    index: str = 'fitness',
                    cmap: str = 'Blues',
                    global_row_labels: str = 'Countries',
                    global_col_labels: str = 'Products',
                    fontsize: int = 20,
                    user_set0: np.ndarray | None = None,
                    user_set1: np.ndarray | None = None,
                    vmin: float | None = None,
                    vmax: float | None = None,
                    zero_nan: bool = False) -> tuple[plt.Figure, plt.Axes]:

        """
        Visualizes the binary matrix using matplotlib's `matshow`.
        Rows and columns are reordered based on a selected index (e.g., fitness, ECI, degree).
        Can optionally apply a custom row/column ordering and control colormap scaling.

        Parameters
        ----------
          - index : str, default 'fitness'
              Sorting criterion: 'fitness', 'eci', 'degree', 'custom', 'no', etc.
          - cmap : str, default 'Blues'
              Colormap to use.
          - global_row_labels : str, default 'Countries'
              Label for row axis.
          - global_col_labels : str, default 'Products'
              Label for column axis.
          - fontsize : int, default 20
              Font size for labels.
          - user_set0 : np.ndarray, optional
              Custom order for rows.
          - user_set1 : np.ndarray, optional
              Custom order for columns.
          - vmin : float, optional
              Minimum value for color scale.
          - vmax : float, optional
              Maximum value for color scale.
          - zero_nan : bool, default False
              If True, zero entries are shown as blank (NaN).

        Returns
        -------
          - tuple
              (fig, ax): Matplotlib figure and axis.
       """

        if user_set0 is not None and user_set1 is not None:
            index = 'custom'

        if index == 'eci':
            set0, set1 = self.get_eci_pci()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        elif index == 'degree':
            set0, set1 = self.get_diversification_ubiquity()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        elif index == 'no':
            set0 = np.arange(self.shape[0])
            set1 = np.arange(self.shape[1])
        elif index == 'invert_x':
            set0, set1 = self.get_fitness_complexity()
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        elif index == 'custom':
            set0 = user_set0
            set1 = user_set1
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::-1]
        elif index == 'fitness':
            set0, set1 = self.get_fitness_complexity(method='tacchella')
            set0 = np.argsort(set0)[::-1]
            set1 = np.argsort(set1)[::]

        matrix = self._processed[set0][:, set1]
        matrix = matrix.toarray() if issparse(matrix) else matrix
        matrix = matrix.astype(float)

        if zero_nan:
            matrix[matrix == 0] = np.nan

        fig, ax = plt.subplots(figsize=(18, 9))
        ax.matshow(matrix, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        if index == 'eci':
            ax.set_xlabel('{} (ordered by increasing PCI)'.format(global_col_labels), fontsize=fontsize, color='black',
                          loc='center')
            ax.set_ylabel('{} (ordered by decreasing ECI)'.format(global_row_labels), fontsize=fontsize, color='black',
                          loc='center')
        elif index == 'degree':
            ax.set_xlabel('{} (ordered by increasing degree)'.format(global_col_labels), fontsize=fontsize, color='black',
                          loc='center')
            ax.set_ylabel('{} (ordered by decreasing degree)'.format(global_row_labels), fontsize=fontsize, color='black',
                          loc='center')
        elif index == 'custom':
            ax.set_xlabel('{} (ordered by increasing rank)'.format(global_col_labels), fontsize=fontsize, color='black',
                          loc='center')
            ax.set_ylabel('{} (ordered by decreasing rank)'.format(global_row_labels), fontsize=fontsize, color='black',
                          loc='center')
        else:
            ax.set_xlabel('{} (ordered by increasing Q)'.format(global_col_labels), fontsize=fontsize, color='black',
                          loc='center')
            ax.set_ylabel('{} (ordered by decreasing F)'.format(global_row_labels), fontsize=fontsize, color='black',
                          loc='center')

        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([0, self.shape[1] - 0.5])
        if index == 'eci' or index == 'degree':
            ax.set_xticklabels(['{}°'.format(self.shape[1]), '1°'], minor=False, fontdict={'fontsize': fontsize})
        else:
            ax.set_xticklabels(['{}°'.format(self.shape[1]), '1°'], minor=False, fontdict={'fontsize': fontsize})
        ax.set_xlim(-0.5, self.shape[1] - 0.5)
        ax.set_yticks([0, self.shape[0] - 1])
        ax.set_yticklabels(['1°', '{}°'.format(self.shape[0])], minor=False, fontdict={'fontsize': fontsize})
        ax.set_ylim(self.shape[0] - 0.5, -0.5)

        return fig, ax
