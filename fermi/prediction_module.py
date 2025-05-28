import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import tqdm
from typing import Union, List, Tuple
from scipy.stats import norm

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
    

# Ok
# 1. The class handles the case when the user wants to predict a point (a set of points) with its (their) trajectory (ies)
# that is (are) not present in the state matrix.

# To do list
# 2. The class should return the vector field relative to the state matrix at a fixed delta_t.
# 3. The class should take as input a vector of sigmas (one relative to the Fitness direction and one relative to the GDP direction).
# 4. The class should handle a vector of weights (one relative to the Fitness direction and one relative to the GDP direction).
# 5. The class should be able to find those points which the predicted distro is not gaussian but multimodal.

class SPS:
    """
    SPS class based entirely on pandas DataFrames for trajectory-based forecasting
    using bootstrap and Nadaraya–Watson regression without peeking ahead in time.
    Chainable wrappers `predict_actor` and `predict_all` allow fluent API.
    """
    def __init__(self,
                 data_dfs: dict[str, pd.DataFrame],
                 delta_t: int = 5,
                 sigma: float = 0.5,
                 n_boot: int = 100,
                 seed: int | None = 42
                ) -> None:
        """
        Initialize an SPS forecaster on a set of trajectory DataFrames.

        Parameters
        ----------
        data_dfs : dict[str, pd.DataFrame]
            A mapping from dimension names (e.g. 'x', 'y', …) to pandas DataFrames.
            Each DataFrame must be indexed by actor and have columns representing years.
        delta_t : int, default=5
            Forecast horizon (number of years ahead to predict).
        sigma : float, default=0.5
            Bandwidth for the Gaussian kernel in Nadaraya–Watson weighting.
        n_boot : int, default=100
            Number of bootstrap samples to draw when using the bootstrap method.
        seed : int or None, default=42
            Seed for the internal random number generator (for reproducible bootstrap draws).

        Raises
        ------
        ValueError
            If `data_dfs` is empty.

        Notes
        -----
        - Actors present in one DataFrame but missing in another will have rows of NaN for missing dimensions.
        - All DataFrames are reindexed to the full range of years, with missing entries as NaN.
        - Builds a long‑form `state_matrix` (MultiIndex: actor * year) holding all dimensions.
        - Placeholders for later results (`nw_actor`, `nw_full`, `boot_actor`, `boot_full`) are created.
        """
        # Input validation: at least one DataFrame
        if not data_dfs:
            raise ValueError("Provide at least one DataFrame in data_dfs.")

        # Compute union of all actors across dimensions
        all_actors = set().union(*(df.index for df in data_dfs.values()))
        actors_keep = sorted(all_actors)

        # Compute continuous range of years across all DataFrames
        min_year = int(min(df.columns.min() for df in data_dfs.values()))
        max_year = int(max(df.columns.max() for df in data_dfs.values()))
        all_years = list(range(min_year, max_year + 1))

        # Reindex each df: align actors and full year range, introduce NaNs
        aligned = {}
        for dim, df in data_dfs.items():
            aligned[dim] = df.reindex(index=actors_keep, columns=all_years)
        self.data_dfs = aligned

        # Update actors and years attributes
        self.actors = pd.Index(actors_keep)
        self.years = pd.Index(all_years)
        self.dimensions = list(self.data_dfs.keys())
        self.delta_t = int(delta_t)
        self.sigma = float(sigma)
        self.n_boot = int(n_boot)
        self.rng = np.random.default_rng(seed)

        # Build long-form state matrix: MultiIndex (actor, year) * dimensions
        stacked = [df.stack().rename(dim) for dim, df in self.data_dfs.items()]
        state = pd.concat(stacked, axis=1)
        full_index = pd.MultiIndex.from_product([self.actors, self.years], names=['actor', 'year'])
        state = state.reindex(full_index)
        
        # Replace infinities with NaN; leave NaNs for methods to handle
        state = state.replace([np.inf, -np.inf], np.nan)
        state = state.sort_index()
        self.state_matrix = state

        ### Placeholders for chainable results ###
        # Nadaraya-Watson
        self.nw_actor: pd.DataFrame | None = None       # df for a single actor
        self.nw_full: pd.DataFrame | None = None        # df for all actors

        # Bootstrap
        self.boot_actor: pd.DataFrame | None = None     # df for a single actor
        self.boot_full: pd.DataFrame | None = None      # df for all actors

    def _compute_analogues(self, actor: str, year: int, delta: int) -> pd.DataFrame:
        """
        Retrieve all "analogue" observations: past states of other actors at least
        `delta` time steps before the target year.

        Parameters
        ----------
        actor : str
            The focal actor whose future we want to predict.
        year : int
            The target year for which we’ll forecast.
        delta : int
            The forecast horizon (number of years ahead).

        Returns
        -------
        pd.DataFrame
            DataFrame of analogue records with MultiIndex (actor, year).
        """
        df = self.state_matrix.reset_index()

        # Build a boolean mask:
        #  - exclude the target actor itself,
        #  - only keep rows at least `delta` years before `year`
        mask = (
            (df['actor'] != actor) &
            (df['year']  <= year - delta)
        )

        # Filter and restore the (actor, year) MultiIndex for the analogue set
        return df.loc[mask].set_index(['actor', 'year'])


    def get_analogues(self,
                      actor: str,
                      year: int,
                      delta: int | None = None) -> pd.DataFrame:
        """
        Public wrapper for _compute_analogues.

        Parameters
        ----------
        actor : str
            Identifier of the actor whose analogues we want to retrieve.
        year : int
            The target year for forecasting.
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.

        Returns
        -------
        pd.DataFrame
            Analogue records indexed by (actor, year).
        """
        if delta is None:
            delta = self.delta_t
        return self._compute_analogues(actor, year, delta)
        
        
    def _regression_core(self,
                         actor: str,
                         year: int,
                         delta: int,
                         dims: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Core routine extracting predict_point, weights, and displacements.

        Parameters
        ----------
        actor : str
            The focal actor identifier.
        year : int
            The base year for prediction.
        delta : int
            Forecast horizon for displacement.
        dims : list[str]
            Dimensions to include in the regression.

        Returns
        -------
        predict_point : np.ndarray
            State vector at (actor, year).
        weights : np.ndarray
            Kernel or sampling weights for analogues.
        delta_X : np.ndarray
            Displacements of analogues over `delta` years.

        Raises
        ------
        ValueError
            If no data for the specified actor/year or no analogues found.
        """
        # predict point
        try:
            point_to_predict = self.state_matrix.loc[(actor, year), dims].astype(float).values
        except KeyError:
            raise ValueError(f"No data for actor '{actor}' at year {year} with delta = {delta}.")

        # Fetch analogues
        analogues = self._compute_analogues(actor, year, delta)
        if analogues.empty:
            raise ValueError(f"No analogues found for actor '{actor}' at year {year} with delta={delta}.")
            
        # Starting and future positions
        start = analogues[dims].astype(float)
        future_idx = [(act, yr + delta) for act, yr in start.index]
        future = self.state_matrix.reindex(future_idx)[dims]
        future.index = start.index
        start_vals = start.values
        
        # Compute displacements
        delta_X = future.values - start_vals
        
        # Clean invalid entries
        delta_X[np.isinf(start_vals)] = np.nan
        delta_X[np.isinf(delta_X)] = np.nan
        valid = ~np.isnan(start_vals).any(axis=1) & ~np.isnan(delta_X).any(axis=1)
        start_vals = start_vals[valid]
        delta_X = delta_X[valid]

        # Compute weights for kernel and bootstrap regressions
        dists = np.linalg.norm(start_vals - point_to_predict, axis=1)
        weights = norm.pdf(dists, 0, self.sigma)
        weights = np.where(np.isfinite(dists), weights, 0.0)

        return point_to_predict, weights, delta_X
        

    def _nad_wat_regression(self,
                             actor: str,
                             year: int,
                             delta: int | None = None,
                             dims: list[str] | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Nadaraya–Watson regression for a single actor-year.

        Parameters
        ----------
        actor : str
            Actor identifier.
        year : int
            Base year for prediction.
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.

        Returns
        -------
        avg_delta : np.ndarray
            Weighted average displacement.
        var_delta : np.ndarray
            Weighted variance of displacements.
        prediction : np.ndarray
            Forecasted state at (actor, year + delta).
        weights : np.ndarray
            Kernel weights for each analogue.
        delta_X : np.ndarray
            Cleaned displacements of analogues.

        Raises
        ------
        ValueError
            If all regression weights are zero.
        """                           
        if delta is None:
            delta = self.delta_t
            
        dims = dims or self.dimensions
        x0, weights, delta_X = self._regression_core(actor, year, delta, dims)
        
        # compute Nadaraya - Watson denominator
        denom = weights.sum()
        if denom == 0:
            raise ValueError("All regression weights are zero; increase sigma or relax filters.")
        
        # Compute avrage displacements (weighted average of all delta X)    
        x_nw = (weights[:, None] * delta_X).sum(axis=0) / denom
        
        # Compute avrage square displacements    
        var_nw = (weights[:, None] * (delta_X - x_nw)**2).sum(axis=0) / denom
        
        # Compute the predicted position of x0
        prediction = x0 + x_nw
                    
        return x_nw, var_nw, prediction, weights, delta_X    
        
    def _bootstrap_regression(self,
                             actor: str,
                             year: int,
                             delta: int | None = None,
                             dims: list[str] | None = None,
                             return_samples: bool = False
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform bootstrap resampling for a single actor-year prediction.

        Parameters
        ----------
        actor : str
            Actor identifier to predict.
        year : int
            Year at which to forecast.
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.
        return_samples : bool, default=False
            Whether to return full bootstrap samples.

        Returns
        -------
        x_boot : np.ndarray
            Mean forecasted position from bootstrap samples.
        sigma_boot : np.ndarray
            Standard deviation of bootstrap samples.
        weights : np.ndarray
            Sampling probabilities for analogues.
        delta_X : np.ndarray
            Original analogue displacements.
        samples : np.ndarray, optional
            Full bootstrap sample trajectories (if `return_samples=True`).
        """
        if delta is None:
            delta = self.delta_t

        dims = dims or self.dimensions
        
        # Get the ingredients for bootstrap
        x0, weights, delta_X = self._regression_core(actor, year, delta, dims)
        
        # normalize weights to probabilities
        probs = weights / weights.sum()
        
        # Number of samples to bootstrap
        n = len(delta_X)
        samples = np.empty((self.n_boot, delta_X.shape[1]))
        
        # Bootstrap sampling: creation of *n_boot* batches
        for b in range(self.n_boot):
            # Choose n different displacements among the delta_X set
            idx = self.rng.choice(n, size=n, replace=True, p=probs)
            
            # Compute the average displacements
            dX_sample = delta_X[idx].mean(axis=0)
            
            # Create the sampled probability distribution
            samples[b] = x0 + dX_sample
        
        # mean of the sampled prob distro    
        x_boot = samples.mean(axis=0)
        
        # std dev of the sampled prob distro    
        sigma_boot = samples.std(axis=0, ddof=1)
        
        # Bootstrap prediction
        prediction = x0 + x_boot
        
        if return_samples:
            return x_boot, sigma_boot, prediction, weights, delta_X, samples
        return x_boot, sigma_boot, prediction, weights, delta_X
    
    
    def predict_actor(self,
                      actor: str,
                      year: int,
                      method: str = 'nw',
                      delta: int | None = None,
                      dims: list[str] | None = None,
                      return_samples: bool = False
                     )-> Union['SPS', Tuple['SPS', np.ndarray]]:
        """
        Chainable wrapper for predicting a single actor-year.

        Parameters
        ----------
        actor : str
            Actor to predict.
        year : int
            Year for prediction.
        method : {'nw', 'boot'}, default='nw'
            Prediction method: Nadaraya–Watson or bootstrap.
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.
        return_samples : bool, default=False
            For 'boot', whether to return bootstrap samples.

        Returns
        -------
        self : SPS
            The instance with `nw_actor` or `boot_actor` set.
        samples : np.ndarray, optional
            Bootstrap samples (if `method='boot'` and `return_samples=True`).
        """
        if delta is None:
            delta = self.delta_t
        
        if dims is None:
            dimes = self.dimensions

        idx = pd.MultiIndex.from_tuples(
            [(actor, year)],
            names=['actor','year']
        )

        if method == 'nw':
            nw_avg, nw_var, nw_pred, weights, dX = \
                self._nad_wat_regression(actor, year, delta, dims)

            df = pd.DataFrame([{
                'nw_avg':  nw_avg,
                'nw_var':  nw_var,
                'nw_pred': nw_pred,
                'weights': weights,
                'dX':      dX
                }], index=idx)

            self.nw_actor = df
            return self

        elif method == 'boot':
            if return_samples:
                boot_avg, boot_var, boot_pred, weights, dX, samples = \
                    self._bootstrap_regression(actor, year, delta=delta, dims=dims, return_samples=True)
            else:
                boot_avg, boot_var, boot_pred, weights, dX = \
                    self._bootstrap_regression(actor, year, delta=delta, dims=dims, return_samples=False)

            df = pd.DataFrame([{
                'boot_avg': boot_avg,
                'boot_var': boot_var,
                'boot_pred':boot_pred,
                'weights':  weights,
                'dX':       dX
                }], index=idx)
            
            self.boot_actor = df

            return (self, samples) if return_samples else self

        else:
            raise ValueError("Method must be 'nw' or 'boot'.")


    def predict_all(self,
                    method: str = 'nw',
                    delta: int | None = None,
                    dims: list[str] | None = None,
                    extra: pd.DataFrame | None = None,
                    return_samples: bool = False) -> Union['SPS', Tuple['SPS', List[np.ndarray]]]:
        """
        Chainable wrapper for batch predictions of all valid actor-years.

        Parameters
        ----------
        method : {'nw', 'boot'}, default='nw'
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.
        extra : pd.DataFrame, optional
            Additional state entries to append before prediction.
        return_samples : bool, default=False
            For 'boot', whether to return list of sample matrices.

        Returns
        -------
        self : SPS
            The instance with `nw_full` or `boot_full` set.
        samples_list : list[np.ndarray], optional
            List of bootstrap sample matrices (if `return_samples=True`).
        """
        if delta is None:
            delta = self.delta_t
        
        if dims is None:
            dimes = self.dimensions

        # Insert eventual `extra` trajectories in self.state_matrix
        original = self.state_matrix
        if extra is not None:
            # Check extra is a MultiIndex (actor, year)
            if not isinstance (extra.index, pd.MultiIndex) or extra.index.names != ['actor', 'year']:
                raise ValueError("`extra` must be a MultiIndex with names=['actor','year']")
            
            # Check that extra has the same keys of data_dfs
            missing = set(self.dimensions) - set(extra.columns)
            if missing:
                raise ValueError(f"'extra' misses dimension {missing}")
            
            # Align extra on every actor, year and dimension
            full_idx = pd.MultiIndex.from_product(
                [self.actors, self.years],
                names=['actor','year']
            )
            e = extra.copy().reindex(index=full_idx, columns=self.dimensions)
            
            # Replace inf with NaNs
            e = e.replace([np.inf, -np.inf], np.nan)
            # Concatenate the extra trajectory to the state_matrix
            self.state_matrix = pd.concat([original, e]).sort_index()    

        # Generate the list of valid actors and years (using updated state_matrix)
        indices = self.state_matrix.index
        valid_keys = [
            (actor, year)
            for actor, year in indices
            if (actor, year + delta) in self.state_matrix.index
        ]

        # each "row" stores the result for _nad_wat_regression or _bootstrap_regression 
        # for that particular pair (actor, year)
        rows = []
        if return_samples: samples_list = []

        for actor, year in valid_keys:
            if method == 'nw':
                nw_avg, nw_var, nw_pred, wgt, dX = self._nad_wat_regression(actor, year, delta, dims)
                row = {
                    'actor':   actor,
                    'year':    year,
                    'nw_avg':  nw_avg,
                    'nw_var':  nw_var,
                    'nw_pred': nw_pred,
                    'weights': wgt,
                    'dX':      dX
                }
            elif method == 'boot':
                if return_samples:
                    boot_avg, boot_var, boot_pred, wgt, dX, samples = \
                        self._bootstrap_regression(actor, year, delta, dims, return_samples=True)
                    samples_list.append(samples)
                else:
                    boot_avg, boot_var, boot_pred, wgt, dX = \
                        self._bootstrap_regression(actor, year, delta, dims, return_samples=False)

                row = {
                    'actor':    actor,
                    'year':     year,
                    'boot_avg': boot_avg,
                    'boot_var': boot_var,
                    'boot_pred':boot_pred,
                    'weights':  wgt,
                    'dX':       dX
                }
            else:
                raise ValueError("Method must be 'nw' or 'boot'.")

            rows.append(row)

        # If extra were add now remove it from the state matrix
        if extra is not None:
            self.state_matrix = original

        # Create the final df containg also features from extra trajectories if present
        df = pd.DataFrame(rows).set_index(['actor', 'year'])

        # Assign to the crrect field and returns self (and samples)
        if method == 'nw':
            self.nw_full = df
            return self
        else:
            self.boot_full = df
            return (self, samples_list) if return_samples else self

    def _velocity_predict(self,
                           actor: str,
                           year: int,
                           delta: int | None = None,
                           dims: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Internal method: forecasts displacement and variance based solely on
        past velocities (first differences).

        Parameters
        ----------
        actor : str
        year : int
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.

        Returns
        -------
        vel_avg : np.ndarray
            Mean velocity-based displacement over `delta` horizon.
        vel_var : np.ndarray
            Variance of velocity-based estimates.

        Raises
        ------
        ValueError
            If insufficient history (<2 points) to compute velocity.
        """
        delta = delta or self.delta_t
        dims = dims or self.dimensions

        # extract historical values for 'actor' up to `year`
        hist = self.state_matrix.loc[actor].sort_index(level='year')  # DataFrame years x dims
        years = hist.index.get_level_values('year').unique()
        valid_years = years[years <= year]
        if len(valid_years) < 2:
            raise ValueError(f"Not enough history to compute velocity for {actor} at {year}.")

        # compute velocities (first differences)
        vals = hist.loc[:, dims].astype(float).values
        vel = np.diff(vals, axis=0)

        # estimate mean and var of velocity over delta steps
        # use velocities from last `delta` differences if available, else all
        if len(vel) >= delta:
            window = vel[-delta:]
        else:
            window = vel
        vel_avg = window.mean(axis=0) * delta
        vel_var = window.var(axis=0, ddof=1) * delta

        return vel_avg, vel_var
    
    def with_velocity_correction(self,
                                  method: str = 'nw',
                                  delta: int | None = None,
                                  dims: list[str] | None = None)-> 'SPS':
        """
        Chainable correction based on velocity: combines base prediction with
        velocity-based estimate using MLE.

        Parameters
        ----------
        method : {'nw', 'boot'}, default='nw'
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.

        Returns
        -------
        self : SPS
            The instance with added 'corr_pred' and 'corr_var' columns in
            the relevant prediction attribute.

        Raises
        ------
        ValueError
            If no base prediction found before correction.
        """
        delta = delta or self.delta_t
        dims  = dims  or self.dimensions

        # determina contesto: full vs single actor
        if method == 'nw':
            target_full = self.nw_full is not None
            df = (self.nw_full.copy()  if target_full else
                  self.nw_actor.copy() if self.nw_actor is not None else None)
            mean_col, var_col = 'nw_pred', 'nw_var'
            attr_full, attr_actor = 'nw_full', 'nw_actor'
        elif method == 'boot':
            target_full = self.boot_full is not None
            df = (self.boot_full.copy()  if target_full else
                  self.boot_actor.copy() if self.boot_actor is not None else None)
            mean_col, var_col = 'boot_avg', 'boot_var'
            attr_full, attr_actor = 'boot_full', 'boot_actor'
        else:
            raise ValueError("Method must be 'nw' or 'boot'.")
        if df is None:
            raise ValueError("No base prediction found. Run predict_all or predict_actor first.")

        corr_means, corr_vars = [], []
        # itera su righe
        for (actor, year), row in df.iterrows():
            m1, v1 = row[mean_col], row[var_col]
            m2, v2 = self._velocity_predict(actor, year, delta, dims)
            inv1, inv2 = np.reciprocal(v1), np.reciprocal(v2)
            cvar = np.reciprocal(inv1 + inv2)
            cmean = cvar * (m1 * inv1 + m2 * inv2)
            corr_means.append(cmean)
            corr_vars.append(cvar)

        df['corr_pred'] = corr_means
        df['corr_var']  = corr_vars

        # aggiorna attributo corretto
        setattr(self, attr_full if target_full else attr_actor, df)
        return self

    def predict_velocity(self,
                              actor: str,
                              year: int,
                              delta: int | None = None,
                              dims: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Chainable wrapper for velocity-based prediction for a single actor-year.

        Parameters
        ----------
        actor : str
            Actor to predict.
        year : int
            Year for prediction.
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.

        Returns
        -------
        vel_avg : np.ndarray
            Mean velocity-based displacement.
        vel_var : np.ndarray
            Variance of velocity-based estimates.
        """
        delta = delta or self.delta_t
        dims  = dims  or self.dimensions

        vel_avg, vel_var = self._velocity_predict(actor, year, delta, dims)
        return vel_avg, vel_var
