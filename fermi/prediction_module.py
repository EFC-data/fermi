import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import tqdm
from typing import Union

class ECPredictor:
    """
    ECPredictor is a unified framework for predicting links in bipartite networks 
    (e.g., countries-technologies) using two distinct strategies:

    1. Network-based prediction: computes M @ B or normalized M @ B / sum(B), 
       where M is the input bipartite matrix and B is a similarity matrix among columns (e.g., technologies).

    2. Machine Learning prediction: learns link probabilities by training a classifier (e.g., Random Forest, XGBoost) 
       column-wise over temporally stacked matrices. Cross-validation is supported with row-level splits (e.g., by country).

    This class is designed for temporal economic complexity analysis and allows evaluation of predictive models 
    both in-sample and on future test matrices.
    """
    def __init__(self, M, mode='network', model=None, normalize=False):
        """
        Inizialize the ECPredictor with a binary bipartite matrix M and a prediction mode.

        Parameters
        ----------
          - M: csr_matrix 
              binary bipartite matrix (e.g. countries x technologies)
          - mode: str 
              either 'network' or 'ml'
          - model: str 
              ML model (must implement fit/predict_proba), required if mode='ml'
          - normalize: bool
              whether to normalize M @ B with B.sum(axis=0) in 'network' mode
        """
        print("Initializing ECPredictor...")
        self.M = M if isinstance(M, csr_matrix) else csr_matrix(M)
        self.mode = mode
        self.model = model
        self.normalize = normalize
        self.M_hat = None

    def predict_network(self, B):
        """
        Predict scores using M @ B or (M @ B) / B if normalize=True

        Parameters
        ----------
          - B: np.array
              similarity matrix (e.g. technologies x technologies)

        Returns
        -------
          - M_hat: np.array
              predicted scores matrix (countries x technologies)
        """
        print("Running network-based prediction...")
        MB = self.M @ B
        if self.normalize:
            print("Applying normalization (density)...")
            B_sum = B.sum(axis=0)
            B_sum[B_sum == 0] = 1  # avoid division by zero
            self.M_hat = MB / B_sum
        else:
            self.M_hat = MB

        print(f"Prediction matrix shape: {self.M_hat.shape}")
        return self.M_hat

    def predict_ml_by_rowstack(self, M_list_train, Y_list_train, M_test):
        """
        Predict using ML with row-wise stacking of M_list_train and Y_list_train.

        Parameters
        ----------
          - M_list_train: list of csr_matrix 
              (features for multiple years)
          - Y_list_train: list of csr_matrix 
              (binary targets for corresponding years)
          - M_test: csr_matrix 
              (features for the year to predict)

        Returns
        -------
          - Y_pred: np.array
              predicted scores (probabilities) for each country x technology
        """
        if self.model is None:
            raise ValueError("No ML model provided.")

        print("Stacking training matrices vertically...")
        X_train = vstack(M_list_train).toarray()
        Y_train = vstack(Y_list_train).toarray()
        X_test = M_test.toarray()

        print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
        Y_pred = np.zeros((X_test.shape[0], Y_train.shape[1]))

        print("Training ML model column by column...")
        for j in tqdm(range(Y_train.shape[1])):
            y_col = Y_train[:, j]
            if np.sum(y_col) == 0:
                continue  # skip if no positive labels
            self.model.fit(X_train, y_col)
            Y_pred[:, j] = self.model.predict_proba(X_test)[:, 1]

        self.M_hat = Y_pred
        print(f"ML prediction matrix shape: {self.M_hat.shape}")
        return self.M_hat

    def predict_ml_crossval(self, M_list_train, Y_list_train, splitter):
        """
        Perform cross-validated ML prediction using row-wise stacked matrices.
        Returns predictions with same shape as stacked training set.

        Parameters
        ----------
          - M_list_train: list of csr_matrix
              features over time
          - Y_list_train: list of csr_matrix
              targets over time (binary)
          - splitter: scikit-learn splitter instance 
              (e.g., KFold(...))

        Returns
        -------
          - Y_pred_full: np.array
              shape (total_rows, n_technologies)
        """
        if self.model is None:
            raise ValueError("No ML model provided.")

        print("Stacking training matrices for cross-validation...")
        X_full = vstack(M_list_train).toarray()
        Y_full = vstack(Y_list_train).toarray()
        n_samples, n_targets = Y_full.shape

        Y_pred_full = np.zeros_like(Y_full, dtype=float)

        print(f"Running cross-validation with {splitter.__class__.__name__}...")
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X_full)):
            print(f"Fold {fold+1}...")
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            Y_train = Y_full[train_idx]

            for j in tqdm(range(n_targets), desc=f"Fold {fold+1} - technologies"):
                y_col = Y_train[:, j]
                if np.sum(y_col) == 0:
                    continue
                self.model.fit(X_train, y_col)
                Y_pred_full[test_idx, j] = self.model.predict_proba(X_test)[:, 1]

        self.M_hat = Y_pred_full
        print(f"Cross-validated prediction shape: {Y_pred_full.shape}")
        return Y_pred_full
    

### Issue: _get_analogues() method to be fixed
# 1. The class should handle the case when the user wants to predict a point (a set of points) that is (are) not present in the state matrix.
# 2. The class should return the vector field relative to the state matrix at a fixed delta_t.
# 3. The class should take as input a vector of sigmas (one relative to the Fitness direction and one relative to the GDP direction).
# 4. The class should handle a vector of weights (one relative to the Fitness direction and one relative to the GDP direction).
# 5. The class should be able to find those points which the predicted distro is not gaussian but multimodal.

from scipy.stats import norm
from scipy.spatial import distance_matrix

class SPS_Forecaster:
    """
    Generalized SPSb Forecaster supporting N-dimensional state trajectories,
    separate Nadaraya–Watson regression, bootstrap sampling, and velocity forecast,
    with flexibility to use a distinct target variable.
    """
    def __init__(self,
                 data_dfs: dict[str, pd.DataFrame],
                 delta_t: int = 5,
                 sigma: float = 0.5,
                 n_boot: int = 1000,
                 seed: int | None = None) -> None:
        """
        Parameters
        ----------
          - data_dfs : dict[str, DataFrame]
              Mapping from dimension name to DataFrame of state variables (actors × years).
          - delta_t : int
              Forecast horizon.
          - sigma : float
              Kernel bandwidth.
          - n_boot : int
              Number of bootstrap samples.
          - seed : int or None
              Random seed.
        """
        # Validate input DataFrames share index/columns
        labels = list(data_dfs.keys())
        if not labels:
            raise ValueError("Provide at least one DataFrame in data_dfs.")
        ref = data_dfs[labels[0]]
        for df in data_dfs.values():
            if df.shape != ref.shape or not (df.index.equals(ref.index) and df.columns.equals(ref.columns)):
                raise ValueError("All DataFrames must share identical index and columns.")
        self.actors = list(ref.index)
        self.years = list(ref.columns)
        self.dim_names = labels
      
        # Build numpy trajectory: (N_actors, N_years, N_dims)
        arrays = [df.values for df in data_dfs.values()]
        self.traj = np.stack(arrays, axis=-1).astype(float)
        self.delta_t = int(delta_t)
        self.sigma = float(sigma)
        self.n_boot = int(n_boot)
        self.rng = np.random.default_rng(seed)
      
        # Build pandas state matrix for lookup
        stacked = [pd.DataFrame(df).stack().rename(name) for name, df in data_dfs.items()]
        state = pd.concat(stacked, axis=1)
        state.index.names = ['actor','year']
        self.state_matrix = state.dropna()
        
        ### Placeholders for chainable results ###
    
        # bootstrap
        self._mu_boot = None
        self._sigma_boot = None
        self._samples_boot = None

        # Nadaraya–Watson 
        self._avg_nw = None
        self._var_nw = None
        self._pred_nw = None
        self._wgt_nw = None
        self._deltaX_nw = None
        
        # Velocity prediction 
        self._mu_vel = None
        self._sigma_vel = None
        self._pred_vel = None

    # ------------------------------------------------------------------
    # Core kernel regression: Nadaraya–Watson
    # ------------------------------------------------------------------
    @staticmethod
    def _nad_wat_regression(traj: np.ndarray,
                             d: int,
                             sigma: float = 0.3,
                             target: np.ndarray | None = None,
                             predict_points: np.ndarray | None = None,
                             avoid_self: bool = True,
                             extended: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Nadaraya–Watson kernel regression to forecast state evolution.

        Parameters
        ----------
          - traj : np.ndarray
              Array of trajectories for each actor over time.
              Shape: (N_actors, N_years, N_dims) or (N_actors, N_years) for 1D.
          - d : int
              Forecast horizon (number of time steps to project forward).
          - sigma : float, default 0.3
              Bandwidth of the Gaussian kernel used to compute weights.
          - target : np.ndarray, optional
              Optional array of target variables to predict. Must have same shape as traj
              on the first two axes. If None, the target is assumed to be equal to traj.
          - predict_points : np.ndarray, optional
              Optional array of shape (N_predict, N_dims) specifying points from which to start
              predictions. If None, use the last point in each trajectory.
          - avoid_self : bool, default True
              If True, prevents an actor from using its own past as an analogue when predicting itself.
          - extended : bool, default False
              If True, returns extra outputs (weights and deltas).

        Returns
        -------
          - avg_delta : np.ndarray, shape (N_predict, N_dims)
              Expected displacement vectors computed as a weighted average.
          - var_delta : np.ndarray, shape (N_predict, N_dims)
              Variance of the displacements (measure of uncertainty).
          - prediction : np.ndarray, shape (N_predict, N_dims)
              Forecasted positions = predict_points + avg_delta.
        """
      
        # if no specific starting points of the predictions are given,
        # then predict from the end of trajectories
        if predict_points is None:
            pp = None
            predict_points = traj[:, -1]
        else:
            pp = True
          
        # Ensure 3D: if in the trajectories only a single quantity
        # over years per actor is considered, i.e
        #
        # traj = np.array([
        #     [1.0, 1.2, 1.5, 1.6],  # actor 1 
        #     [2.0, 2.1, 2.3, 2.4],  # actor 2
        #     [0.9, 1.0, 1.1, 1.3]   # actor 3
        #     ])
        #
        # shape: (3 actors, 4 years, 1 quanty ~ GDP)
        #
        # then add a newaxis to have tensor form as in higher dimensional cases
        if traj.ndim == 2:
            traj = traj[:, :, np.newaxis]
        ndims = traj.shape[-1]
        
        # example for higher dimensional case: two quantities over years per actor
        # traj = np.array([
        #     [[1.0, 10.0], [1.2, 10.5], [1.5, 10.7], [1.6, 10.8]],  # actor 1
        #     [[2.0, 20.0], [2.1, 20.1], [2.3, 20.3], [2.4, 20.4]],  # actor 2
        #     [[0.9, 9.0],  [1.0, 9.5],  [1.1, 9.8],  [1.3, 9.9]]    # actor 3
        #     ])
        # shape: (3 actors, 4 years, 2 quantities ~ (Fitness, GDP))
    
        # Target default = traj
        if target is None:
            target = traj
            predict_target = predict_points
        else:
            if target.ndim == 2:
                target = target[:, :, np.newaxis]
            assert target.shape[:2] == traj.shape[:2]
            predict_target = target[:, -1]
          
        # Replace inf with nan
        target[np.isinf(target)] = np.nan
        traj[np.isinf(traj)] = np.nan
      
        # compute delta X for each trajectory
        delta_X = target[:, d:] - target[:, :-d]
        # starting positions of all delta X
        starting_pos = traj[:, :-d]
      
        # Compute distance tensor
        # dm[i,j,k]: distance between last point of country i from point of country j in year k
        dm = distance_matrix(predict_points, starting_pos.reshape(-1, ndims))
        dm = dm.reshape(predict_points.shape[0], traj.shape[0], -1)
      
        # Avoid self
        if avoid_self and (pp is None):
            for i in range(traj.shape[0]):
                dm[i, i] = np.inf   # In this way dm[i,i,:] is always np.inf, so kernel weights [i,i]
                                    # will always be 0.
              
        # compute regression weights from a gaussian kernel
        wgt = norm.pdf(dm, 0, sigma)
      
        # Mask invalid in traj
        # if any point is nan in traj, set their weight and delta to 0, so they don't affect predictions
        cou, years, _ = np.where(np.isnan(traj) | np.isinf(traj))
        for c, y in zip(cou, years):
            if y < wgt.shape[2]:
                wgt[:, c, y] = 0
                delta_X[c, y] = 0
            if y - d >= 0:
                delta_X[c, y - d] = 0
                wgt[:, c, y - d] = 0
              
        # Mask invalid in target; same logic as in traj
        if not np.allclose(np.nansum(target), np.nansum(traj)):
            cou, years, _ = np.where(np.isnan(target) | np.isinf(target))
            for c, y in zip(cou, years):
                if y < wgt.shape[2]:
                    wgt[:, c, y] = 0
                    delta_X[c, y] = 0
                if y - d >= 0:
                    delta_X[c, y - d] = 0
                    wgt[:, c, y - d] = 0
                  
        # compute the Nadaraya-Watson denominator
        wgt_den = wgt.sum(-1).sum(-1)[:, np.newaxis]
        # if some kernel regression has all-0 weights
        # set denominator to NaN, to have NaN as an output instead of inf
        wgt_den[wgt_den == 0] = np.nan
      
        # Average displacement for each actor in each target's dimension
        avg_delta_X = np.einsum('ijy,jyd->id', wgt, delta_X) / wgt_den
      
        # average quadratic deviations from the average displacement:
        # tovar[ijyd]: quadratic deviation of delta of country j in year y from avg_delta[i,d]
        tovar = (delta_X[np.newaxis] - avg_delta_X[:, np.newaxis, np.newaxis])**2
        tovar[np.isnan(tovar)] = 0
        var_delta_X = np.einsum('ijy,ijyd->id', wgt, tovar) / wgt_den
      
        if extended:
            return avg_delta_X, var_delta_X, predict_target + avg_delta_X, wgt, delta_X
        return avg_delta_X, var_delta_X, predict_target + avg_delta_X

    # ------------------------------------------------------------------
    # Kernel regression wrapper (vectorized for all actors)
    # ------------------------------------------------------------------
    def get_Nad_Wat_regression(self,
                            predict_points: np.ndarray | None = None,
                            target: np.ndarray | None = None,
                            avoid_self: bool = True,
                            extended: bool = False) -> "SPS_Forecaster":
        """
        Chainable wrapper for Nadaraya–Watson regression on the stored traj.
        Parameters
        ----------
            - predict_points : np.ndarray, optional
                Optional array of shape (N_predict, N_dims) specifying points from which to start
                predictions. If None, use the last point in each trajectory.
            - target : np.ndarray, optional
                Optional array of target variables to predict. Must have same shape as traj
                on the first two axes. If None, the target is assumed to be equal to traj.
            - avoid_self : bool, default True
                If True, prevents an actor from using its own past as an analogue when predicting itself.
            - extended : bool, default False
                If True, returns extra outputs (weights and deltas).

        Returns
        -------
          - self : SPS_Forecaster
            Chainable return containing:
               - self._avg_nw : np.ndarray, shape (N_predict, N_dims)
               - self._var_nw : np.ndarray, shape (N_predict, N_dims)
               - self._pred_nw : np.ndarray, shape (N_predict, N_dims)
               - self._wgt_nw, self._deltaX_nw if extended=True
        """
        avg, var, pred = self._nad_wat_regression(
            traj=self.traj,
            d=self.delta_t,
            sigma=self.sigma,
            target=target,
            predict_points=predict_points,
            avoid_self=avoid_self,
            extended=False
        )
        self._avg_nw = avg
        self._var_nw = var
        self._pred_nw = pred
        if extended:
            wgt, dX = self._nad_wat_regression(
                traj=self.traj,
                d=self.delta_t,
                sigma=self.sigma,
                target=target,
                predict_points=predict_points,
                avoid_self=avoid_self,
                extended=True
            )[3:]
            self._wgt_nw = wgt
            self._deltaX_nw = dX
        return self

    # ------------------------------------------------------------------
    # Bootstrap sampling of predictions
    # ------------------------------------------------------------------   
    def _bootstrap(self,
                  predict_points: np.ndarray | None = None,
                  target: np.ndarray | None = None,
                  avoid_self: bool = True,
                  return_samples: bool = False) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform bootstrap on historical deltas weighted by kernel to obtain
        a distribution of future predictions.

        Returns
        -------
          - mu : np.ndarray, shape (N_dims,)
              Mean forecasted displacement added to predict_point.
          - sigma : np.ndarray, shape (N_dims,)
              Standard deviation of forecast.
          - samples : np.ndarray, shape (n_boot, N_dims), optional
              If return_samples=True, full bootstrap samples.
        """

        # Get weights and delta_x
        _, _, _, wgt, deltas = self._nad_wat_regression(
            self.traj, self.delta_t, self.sigma,
            target, predict_points, avoid_self, debug=True
        )
      
        # Flatten for sampling
        w_flat = wgt.reshape(-1)
        d_flat = deltas.reshape(-1, self.traj.shape[-1])
        p_sum = w_flat.sum()
      
        if p_sum == 0:
            raise ValueError("All weights zero: no valid analogues.")
          
        probs = w_flat / p_sum
        n = len(probs)
        samples = np.empty((self.n_boot, self.traj.shape[-1]))

        # Bootstrap sampling of displacement vectors
        for b in range(self.n_boot):
            # Create a sample of n different vectors among the displacement vectors 
            idx = self.rng.choice(n, size=n, replace=True, p=probs)
            # Compute the (mean) forecast displacement in the phase space  
            mean_d = d_flat[idx].mean(axis=0)
            samples[b] = (predict_points if predict_points is not None else self.traj[:, -1])[0] + mean_d
          
        mu_boot = samples.mean(axis=0)
        sigma_boot = samples.std(axis=0, ddof=1)
      
        if return_samples:
            return mu_boot, sigma_boot, samples
        return mu_boot, sigma_boot
    
    # ------------------------------------------------------------------
    # Bootstrap wrapper (vectorized for all actors)
    # ------------------------------------------------------------------
    def get_bootstrap(self,
                     target: np.ndarray | None = None,
                     avoid_self: bool = True,
                     return_samples: bool = False
                     ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized bootstrap forecast for all actors.

        Parameters
        ----------
          - target : np.ndarray, optional
              Optional external target array.
          - avoid_self : bool, default True
              Whether to exclude self-past in regression.
          - return_samples : bool, default False
              If True, return full bootstrap sample matrix.

        Returns
        -------
          - self : SPS_Forecaster
                Chainable return containing:
                    - self._mu_boot : np.ndarray, mean displacement (N_actors × N_dims)
                    - self._sigma_boot : np.ndarray, std dev of displacement
                    - self._samples_boot : np.ndarray, bootstrap samples (n_boot, N_actors, N_dims)
                      if return_samples=True
        """
        pts = self.traj[:, -1, :]
        mus = []
        sigs = []
        all_samples = []
        for i in range(pts.shape[0]):
            out = self._bootstrap(predict_points=pts[i:i+1], target=target,
                                 avoid_self=avoid_self, return_samples=return_samples)
            if return_samples:
                m, s, samp = out
                all_samples.append(samp)
            else:
                m, s = out
            mus.append(m)
            sigs.append(s)
        # Stack results per actor
        self._mu_boot = np.vstack(mus)
        self._sigma_boot = np.vstack(sigs)
        if return_samples:
            # assemble samples into shape (n_boot, N_actors, N_dims)
            self._samples_boot = np.stack(all_samples, axis=1)
        return self

    # ------------------------------------------------------------------
    # Velocity-based forecast (global average velocity bootstrap)
    # ------------------------------------------------------------------
    @staticmethod
    def _velocity_predict(traj: np.ndarray,
                          d: int,
                          target: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute average velocity and variance over entire history, scaled by d.
        traj: np.array, shape (N_actors, N_years, N_dims) or (N_actors, N_years)
        d: int, time-delay (horizon)
        target: int dimension index for which to return specific component (optional)
        Returns (mean, var, pred) arrays of shape (N_actors, N_dims) or (N_actors, 1).
        """
        import warnings
        # Ensure 3D
        if traj.ndim == 2:
            traj = traj[:, :, np.newaxis]
        # Compute diffs year-to-year
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Degrees of freedom")
            # average one-step diff then scale
            mean = np.nanmean(np.diff(traj, axis=1), axis=1) * d
            var = np.nanvar (np.diff(traj, axis=1), axis=1) * d
        # If user requested a specific target dimension
        if target is not None:
            if target < 0 or target >= traj.shape[-1]:
                raise ValueError('target must be a valid dimension index', target, traj.shape[-1])
            # select that dimension
            m = mean[..., target][:, np.newaxis]     # '...' numpy ellipsis operator
            v = var[..., target][:, np.newaxis]
            p = traj[:, -1, target][:, np.newaxis] + m
            return m, v, p
        # else return full
        p = traj[:, -1, :] + mean
        return mean, var, p
    
    # ------------------------------------------------------------------
    # velocity predictor wrapper (vectorized for all actors)
    # ------------------------------------------------------------------
    def get_velocity_predict(self,
                             target: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        CHainable velocity forecast for all actors based on historical average velocity.

        Parameters
        ----------
          - target : int or None
              If provided, selects a single target dimension index; otherwise returns all dims.

        Returns
        -------
        self : SPS_Forecaster
            Chainable return containing:
              - self._mu_vel    : np.ndarray, mean displacement (N_actors × N_dims or ×1)
              - self._sigma_vel : np.ndarray, standard deviation of displacement
              - self._pred_vel  : np.ndarray, forecasted positions = last value + mean
        """
        # Compute vectorized velocity predictions
        mean, var, pred = self._velocity_predict(self.traj, self.delta_t, target)
        # Convert var to standard deviation
        sigma = np.sqrt(var)
        
        # Store results for chaining
        self._mu_vel = mean
        self._sigma_vel = sigma
        self._pred_vel = pred

        return self


    # ------------------------------------------------------------------
    # Accessory: Combine last forecast with velocity forecast
    # ------------------------------------------------------------------
    def with_velocity_correction(self,
                                 method: str = 'bootstrap',
                                 target: int | None = None
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply velocity-based correction to the most recent forecast.

        This method uses the historical velocity forecast (via _velocity_predict)
        to adjust the latest Nadaraya-Watson or bootstrap forecast in a maximum-
        likelihood fusion, yielding a new mean and uncertainty.

        Parameters
        ----------
          - method : str
              Which forecast to correct: 'bootstrap' or 'nw' (Nadaraya-Watson).
          - target : int or None
              If provided, selects a single dimension index for the velocity forecast.

        Returns
        -------
          - mu_corr : np.ndarray
              corrected forecast mean (N_actors * N_dims)
          - sigma_corr : np.ndarray
              corrected forecast std dev
        """
        # Retrieve the last forecast
        if method.lower().startswith('nw'):
            mu_hist   = self._pred_nw
            sigma_hist= np.sqrt(self._var_nw)
        else:
            mu_hist   = self._mu_boot
            sigma_hist= self._sigma_boot

        # Compute velocity forecast
        mean_vel, var_vel, pred_vel = self._velocity_predict(self.traj, self.delta_t, target)
        sigma_vel = np.sqrt(var_vel)

        # Fuse via Maximum Likelihood: μ* = (μ_hist·σ_vel² + μ_vel·σ_hist²) / (σ_hist² + σ_vel²)
        var1 = sigma_hist**2
        var2 = sigma_vel**2
        mu_corr = (mu_hist * var2 + pred_vel * var1) / (var1 + var2)
        sigma_corr = np.sqrt((var1 * var2) / (var1 + var2))

        return mu_corr, sigma_corr
