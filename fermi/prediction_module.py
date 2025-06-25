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
    Chainable wrappers `predict_actor` allow fluent API.
    """
    def __init__(self,
                 data_dfs: dict[str, pd.DataFrame],
                 delta_t: int = 5,
                 sigma: float = 0.5,
                 n_boot: int = 100,
                 seed: int | None = 42   # The Answer to the Ultimate Question of Life, the Universe, and Everything 
                ) -> None:
        """
        Initialize an SPS forecaster on a set of trajectory DataFrames.

        Parameters
        ----------
        data_dfs : dict[str, pd.DataFrame]
            dictionary of DataFrames that must be indexed by actor and have columns representing years.
            Each DataFrame represents a different dimension (e.g., GDP, fitness) of the actors' trajectories.
        delta_t : int, default=5
            Forecast horizon (number of years ahead to predict).
        sigma : float, default=0.5
            Bandwidth for the Gaussian kernel in Nadaraya-Watson weighting.
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
        - Builds a long-form `state_matrix` (MultiIndex: actor * year) holding all dimensions.
        - Placeholders for later results (`nw_actor`,`boot_actor`) are created.

        References
        ----------
        - A. Tacchella, D. Mazzilli, L. Pietronero, A dynamical systems approach to gross domestic product forecasting, Nature Physics 14 (8), 861-865
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
        self.nw_actor: pd.DataFrame | None = None       # refers to a single actor

        # Bootstrap
        self.boot_actor: pd.DataFrame | None = None  

        # Placeholder for velocity-based predictions
        self.vel_actor = None
        self.var_vel_actor = None  

        # Placeholder for combined SPS + velocity predictions
        self.sps_vel_actor: pd.DataFrame | None = None


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
        point_to_predict : np.ndarray
            State vector at (actor, year).
        weights : np.ndarray
            Kernel or sampling weights for analogues.
        delta_X : np.ndarray
            Displacements of `point_to_predict` over `delta` years.

        Raises
        ------
        ValueError
            If no data for the specified actor/year or no analogues found.
        """
        # starting point
        try:
            point_to_predict = self.state_matrix.loc[(actor, year), dims].astype(float).values
        except KeyError:
            raise ValueError(f"No data for actor '{actor}' at year {year} with delta = {delta}.")

        # Fetch analogues of the actor at the specified year
        analogues = self._compute_analogues(actor, year, delta)
        if analogues.empty:
            raise ValueError(f"No analogues found for actor '{actor}' at year {year} with delta={delta}.")
            
        # Starting and future positions
        start = analogues[dims].astype(float)
        future_idx = [(act, yr + delta) for act, yr in start.index]
        future = self.state_matrix.reindex(future_idx)[dims]
        future.index = start.index
        start_vals = start.values
        
        # Compute displacements of the analogues
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
        Perform Nadaraya-Watson regression for a single actor-year.

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
            raise ValueError("All regression weights are zero for actor '{actor}' at year {year} (delta={delta});" \
            "cannot perform Nadaraya-Watson regression.")
        
        # Compute average displacements (weighted average of all delta X)    
        x_nw = (weights[:, None] * delta_X).sum(axis=0) / denom
        
        # Compute average square displacements    
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
        Perform vectorized bootstrap resampling for a single actor-year prediction.

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
        pred_samples : np.ndarray, optional
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

        # Draw all bootstrap indices at once: shape (n_boot, n)
        all_idx = self.rng.choice(n, size=(self.n_boot, n), replace=True, p=probs)
        
        # gather all displacements at once
        disp_samples = delta_X[all_idx].mean(axis=1)

        # Compute mean and std of the displacement samples
        x_boot = disp_samples.mean(axis=0)
        sigma_boot = disp_samples.std(axis=0, ddof=1)

        # Bootstrap prediction
        prediction = x0 + x_boot

        # If return_samples=True, add x0 back to each sample:
        if return_samples:
            pred_samples = disp_samples + x0
            return x_boot, sigma_boot, prediction, weights, delta_X, pred_samples

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
            Prediction method: Nadaraya-Watson or bootstrap.
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
            dims = self.dimensions

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


    def predict_actor_velocity(self,
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
            Actor to predict.
        year : int
            Year for prediction.
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.

        Returns
        -------
        delta_vel : np.ndarray
            velocity computed as most (recent value - value delta steps ago) / delta
        var_vel : np.ndarray
            Variance of the estimated velocity, computed as the sample variance of 
            the one-year displacements and scaled by the horizon delta.

        Raises
        ------
        ValueError
            If the actor has no history.
            If the observations at year-delta or year are NaN.
            If the actor has insufficient history for velocity computation.
        """
        delta = delta or self.delta_t
        dims = dims or self.dimensions

        # Get actor's trajectory 
        history = (
            self.state_matrix
                .loc[actor]
                .sort_index(level='year')
        )

        if history.empty:
            raise ValueError(f"No history for actor '{actor}'.")
        
        years = history.index.get_level_values('year').astype(int).values
        
        # Restrict to years ≤ target
        mask = years <= year
        yrs = years[mask]

        # Extract values for the relevant years
        vals = history.values[mask]

        # Check delta-year observation
        if np.isnan(vals[-delta]).any():
            raise ValueError(f"Cannot compute velocity: observation at {year-delta} for actor='{actor}'")

        # Check target-year observation
        if np.isnan(vals[-1]).any():
            raise ValueError(f"Cannot compute velocity: observation at {year} for actor='{actor}'")

        # Compute velocity over the horizon directly as (most recent value − value delta steps ago) / delta
        delta_vel = (vals[-1] - vals[-delta]) / delta

        # Identify which adjacent entries in `yrs` are exactly one year apart
        consecutive = np.diff(yrs) == 1      
        if not np.any(consecutive):
            raise ValueError(f"Insufficient history for velocity computation: actor='{actor}' at year={year}.")  

        # Compute the valid one-year displacements
        one_year_disp = vals[1:][consecutive]- vals[:-1][consecutive]

        # Sample variance of those displacements, scaled by the horizon delta
        var_vel = one_year_disp.var(axis=0, ddof=1) * delta

        self.vel_actor = delta_vel
        self.var_vel_actor = var_vel

        return self

    def sps_plus_velocity(self,
                        actor: str,
                        year: int,
                        method: str = 'nw',
                        delta: int | None = None,
                        dims: list[str] | None = None) -> 'SPS':
        """
        Combine SPS forecast and velocity forecast via precision-weighting:
            mu    = (mu_sps/sigma_sps + mu_vel/sigma_vel) / (1/sigma_sps + 1/sigma_vel)
            sigma = 1 / (1/sigma_sps + 1/sigma_vel)

        Parameters
        ----------
        actor : str
            Actor to predict.
        year : int
            Year for prediction.
        method : {'nw', 'boot'}, default='nw'
            Prediction method: Nadaraya-Watson or bootstrap.
        delta : int, optional
            Forecast horizon; defaults to self.delta_t.
        dims : list[str], optional
            Dimensions to include; defaults to self.dimensions.

        Returns
        -------
        self : SPS
            The instance with `sps_vel_actor` set, containing combined predictions.

        References
        """

        delta = delta or self.delta_t
        dims = dims or self.dimensions

        # Get SPS-based forecast
        self.predict_actor(actor=actor, year=year, method=method, delta=delta, dims=dims)

        # Get velocity-based forecast
        self.predict_actor_velocity(actor=actor, year=year, delta=delta, dims=dims)

        # Extract SPS mean & variance
        idx = (actor, year)
        if method == 'nw':
            row = self.nw_actor.loc[idx]         # fields: 'nw_avg', 'nw_var', etc.
            mu_sps, var_sps = row['nw_avg'], row['nw_var']
        else:
            row = self.boot_actor.loc[idx]       # fields: 'boot_avg', 'boot_var', ...
            mu_sps, var_sps = row['boot_avg'], row['boot_var']
    
        # Extract velocity mean & variance
        mu_vel  = self.vel_actor
        var_vel = self.var_vel_actor

        # Precision-weighted combination
        prec_sps = 1.0 / var_sps
        prec_vel = 1.0 / var_vel
        combined_var = 1.0 / (prec_sps + prec_vel)
        combined_mu  = combined_var * (mu_sps * prec_sps + mu_vel * prec_vel)

        # Compute combined prediction
        x0 = self.state_matrix.loc[idx, dims].astype(float).values
        combined_pred = x0 + combined_mu

        # Store the result in a new DataFrame
        df = pd.DataFrame(
        [{
            'combined_var':  combined_var,
            'combined_mu':   combined_mu,
            'combined_pred': combined_pred
        }],
        index=pd.MultiIndex.from_tuples([idx], names=['actor', 'year'])
    )
        self.sps_vel_actor = df
        return self
    
