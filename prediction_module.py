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
class SPSbForecaster:    
    def __init__(self,
                 fitness_df: pd.DataFrame,
                 GDP_pc_df: pd.DataFrame,
                 delta_t: int = 5,
                 sigma: float = 0.5,
                 n_boot: int = 1000,
                 seed: int | None = None) -> None:
    
        """
        Inizialize the Bootstrap Selective Predictability Scheme (SPSb) forecaster.

        Parameters
        ----------
        - fitness_df : pd.DataFrame
            Rows represent actors, columns represent years (int).
            Values are positive fitness scores.
        - gdp_df : pd.DataFrame
            Same shape,index, and columns as ``fitness_df`` with PPP‑adjusted
            GDP per capita (constant prices), rows = actors, columns = years.
        - delta_t : int, default 5
            Forecast horizon in years.
        - sigma : float, default 0.5
            Gaussian kernel bandwidth in (log F, log GDP) space.
        - n_boot : int, default 1000
            Number of bootstrap batches.
        - seed : int or None, default None
            Seed for the internal random number generator.

        Attributes
        ----------
        state_matrix : pd.DataFrame
            MultiIndex DataFrame indexed by (year, actor), with columns ["logF", "logGDP"].
        """
        self._validate_inputs(fitness_df, GDP_pc_df)
        self.fitness = fitness_df.astype(float)
        self.gdp = GDP_pc_df.astype(float)
        self.delta_t = int(delta_t)
        
        self.sigma = sigma
        self.n_boot = int(n_boot)
        self.rng = np.random.default_rng(seed)
        
        # Pre‑compute log‑transformed state matrix once for all.
        self.state_matrix = self._build_state_matrix()
        
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------    
    def predict_SPSb(self, actor: str, year: int, use_velocity: bool = False,
                     return_distro: bool = False) -> Union[tuple[np.ndarray, np.ndarray],
                      tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict (log F, log GDP) at ``year + delta_t`` for a specific *actor*.
        
        Parameters
        ----------
          - actor : str
              Code of the target actor.
          - year : int
              Base year for the forecast; must satisfy year + delta_t in data.
          - use_velocity : bool, default False
              If True, combine SPSb forecast with naive velocity prediction.
          - return_distro : bool, default False
              If True, also return the full bootstrap sample distribution.

        Returns
        -------
        x : np.ndarray, shape (2,)
            Forecasted point [logF, logGDP]. 
            If `use_velocity=True`, this is the mean of the combined distribution.
        sigma : np.ndarray, shape (2,)
            Component-wise standard deviation (1σ uncertainty). 
            If `use_velocity=True`, this is the std dev of the combined distribution.
        samples : np.ndarray, shape (n_boot, 2), optional
            Bootstrap distribution of SPSb forecasted points (only if return_distribution=True).
        """
        
        # Choose a target *actor* in the phase space at a certain *year*
        x_target = self._get_state(actor, year)
        # get all the points of the phase space up to *year* and their displacement
        # vectors consitent with a delta_t time window
        X, dX = self._get_analogues(year)
        # generate the gaussian weights
        weights = self._kernel_weights(X, x_target)
        # normalize the wieghts
        probs = weights / weights.sum()
        n = len(dX)
        
        # Bootstrap sampling of displacement vectors
        samples = np.empty((self.n_boot, 2))
        for b in range(self.n_boot):
            # Create a sample of n different vectors among the displacement vectors 
            idx = self.rng.choice(n, size=n, replace=True, p=probs)
            # Compute the (mean) forecast displacement in the phase space  
            dX_sample = dX[idx].mean(axis=0)
            samples[b] = x_target + dX_sample
            
        x_spsb = samples.mean(axis=0)
        sigma_spsb = samples.std(axis=0, ddof=1)
        
        if not use_velocity:
            if return_distro:
                return x_spsb, sigma_spsb, samples
            return x_spsb, sigma_spsb
        
        # Velocity-based prediction
        x_vel, sigma_vel = self.predict_velocity(actor, year)

        # Combine two Gaussian forecasts
        # SPSb ~ N(x_spsb, sigma_spsb^2)
        # Velocity ~ N(x_vel, sigma_vel^2)
        #
        # Derivation of the fused mean via Maximum Likelihood:
        #
        # Suppose we have two independent observations x1, x2 of the same unknown μ,
        # where:
        #   x1 ~ N(μ, σ1^2)
        #   x2 ~ N(μ, σ2^2)
        #
        # The joint likelihood is:
        #   L(μ) ∝ exp[-(x1 - μ)^2 / (2σ1^2)] * exp[-(x2 - μ)^2 / (2σ2^2)]
        #        = exp[- ( (x1 - μ)^2 / (2σ1^2) + (x2 - μ)^2 / (2σ2^2) ) ]
        #
        # Maximizing log L(μ) gives:
        #   d/dμ log L = (x1 - μ)/σ1^2 + (x2 - μ)/σ2^2 = 0
        #   μ * (1/σ1^2 + 1/σ2^2) = x1/σ1^2 + x2/σ2^2
        #   => μ = (x1 * σ2^2 + x2 * σ1^2) / (σ1^2 + σ2^2)
        #
        # Applied here with:
        #   x1 = x_spsb, σ1 = sigma_spsb
        #   x2 = x_vel,  σ2 = sigma_vel
        #
        # Final distribution:
        #   x_spsb_vel = (x_spsb * sigma_vel^2 + x_vel * sigma_spsb^2) / (sigma_spsb^2 + sigma_vel^2)
        #   sigma^2_spsb_vel = (sigma_spsb^2 * sigma_vel^2) / (sigma_spsb^2 + sigma_vel^2)

        var_spsb = sigma_spsb ** 2
        var_vel = sigma_vel ** 2
        denom = var_spsb + var_vel

        x_spsb_vel = (x_spsb * var_vel + x_vel * var_spsb) / denom
        sigma_spsb_vel = np.sqrt((var_spsb * var_vel) / denom)

        if return_distro:
            return x_spsb_vel, sigma_spsb_vel, samples
        return x_spsb_vel, sigma_spsb_vel
             
    
    def predict_velocity(self, actor: str, year: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast (log F, log GDP) at year + delta_t using only the country's past velocity.
        x_c,t+Δt = x_c,t + δx_c_past where δx_c_past = x_c,t* - x_c,t*−Δt.
        This forecast does not look at other actors (no analogues), 
        but assumes that the target actor continues to move exactly as it did in the 
        previous time step.

        Parameters
        ----------
          - actor : str
              Code of the target actor.
          - year : int
              Base year for the forecast; must satisfy year - delta_t in data.

        Returns
        -------
          - x_naive : np.ndarray, shape (2,)
              Forecasted point [logF, logGDP] based on past delta_t displacement.
          - sigma_naive : np.ndarray, shape (2,)
              Standard deviation of past one-year-displacements, used as uncertainty estimate.

        Raises
        ------
          - ValueError
              If required historical data is missing for computing the past velocity.
        """
        try:
            x_t = self._get_state(actor, year)
            x_t_minus_dt = self._get_state(actor, year - self.delta_t)
        except ValueError as e:
            raise ValueError("Cannot compute past velocity: missing historical data.") from e

        delta_x_past = x_t - x_t_minus_dt

        # Estimate variance from annual displacements over the past one-year-displacements
        try:
            x_series = np.array([
                self._get_state(actor, y)
                for y in range(year - self.delta_t + 1, year + 1)
            ])
        except ValueError:
            raise ValueError("Cannot compute velocity uncertainty: incomplete annual history.")

        diffs = x_series[1:] - x_series[:-1]  # shape (delta_t - 1, 2)
        sigma_naive = diffs.std(axis=0, ddof=1)
        x_naive = x_t + delta_x_past
        return x_naive, sigma_naive
    
    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_inputs(fit: pd.DataFrame, gdp: pd.DataFrame):
        if fit.shape != gdp.shape or not (
           fit.index.equals(gdp.index) and fit.columns.equals(gdp.columns)):
            raise ValueError(
                "fitness_df and gdp_df must have identical shape, index, and columns"
            )
        if (fit <= 0).any().any() or (gdp <= 0).any().any():
            raise ValueError(
                "All fitness and GDP values must be strictly positive to compute logarithms"
            )
            
    def _build_state_matrix(self) -> pd.DataFrame:
        """
        Construct the full state matrix as a MultiIndex DataFrame.

        Transforms the input fitness and GDP DataFrames (indexed by actor, columns = years)
        into a long-form DataFrame indexed by (year, country), with columns [logF, logGDP].

        Returns
        -------actor
          - pd.DataFrame
              MultiIndex DataFrame indexed by (year, actor), containing
              columns 'logF' and 'logGDP' (natural logarithms of input data),
              with any rows containing NaN values dropped.
        """
            
        # Create MultiIndex (year, country) with columns [logF, logGDP]
        log_f = np.log(self.fitness.T).stack().rename("logF")
        log_g = np.log(self.gdp.T).stack().rename("logGDP")
        state = pd.concat([log_f, log_g], axis=1)
        state.index.names = ["year","actor"]
        return state.dropna()
    
    def _get_state(self, actor: str, year: int) -> np.ndarray:
        """
        Retrieve the 2D state vector (logF, logGDP) for a given actor and year.

        Parameters
        ----------
          - actor : str
              Code of the actor to retrieve.
          - year : int
              Year to retrieve.

        Returns
        -------
          - np.ndarray, shape (2,)
              Array containing [logF, logGDP] for the given actor and year.

        Raises
        ------
          - ValueError
              If the (actor, year) pair is not found in the state matrix.
        """
        try:
            return self.state_matrix.loc[(year,actor)].values
        except KeyError as exc:
            raise ValueError(f"State not available for the specified pair ({year},{actor})") from exc
            
    def _get_analogues(self, target_year: int):
        """
        Return arrays (X, dX) of analogues prior to target_year:
        positions x_{i,τ} and displacements x_{i,τ+Δt} - x_{i,τ},
        where τ + Δt <= *target_year*.
        """
        idx = self.state_matrix.index
        tau = idx.get_level_values("year")
        actor = idx.get_level_values("actor")

        # Select only analogues x_{i,τ} with τ + Δt ≤ *target_year*
        valid = tau + self.delta_t <= target_year
        state_tau = self.state_matrix[valid]

        tau_valid = tau[valid]
        actor_valid = actor[valid]
        tau_plus_dt = tau_valid + self.delta_t

        # Build MultiIndex at τ+Δt
        idx_next = pd.MultiIndex.from_arrays([tau_plus_dt, actor_valid], names=["year", "actor"])
        state_tau_plus_dt = self.state_matrix.reindex(idx_next)

        # drop NaN rows
        mask = ~state_tau_plus_dt.isna().any(axis=1)
        print("\n",mask)
        # keep only valid rows
        state_tau_plus_dt = state_tau_plus_dt[mask]
        
        # drop duplicates
        dup_idx = state_tau.index.intersection(state_tau_plus_dt.index)   
        state_tau = state_tau.drop(dup_idx)

        print("\n Valid analogues (tau):")
        print(state_tau)
        print("\n Valid analogues (tau+dt):")
        print(state_tau_plus_dt)
        
        X = state_tau.values
        dX = ( - ).values
        
        print("\n",dX)
        return X, dX
            
    
    def _kernel_weights(self, X: np.ndarray, x_target: np.ndarray, verbose: bool = False) -> np.ndarray:
        """
        Compute Gaussian kernel weights for a target point relative to a set of analogue points.

        Parameters
        ----------
          - X : np.ndarray, shape (n, 2)
              Array of analogue positions in log(F), log(GDP) space.
          - x_target : np.ndarray, shape (2,)
              Target position in log(F), log(GDP) space.

        Returns
        -------
          - np.ndarray, shape (n,)
              Array of kernel weights, one for each analogue point.

        Raises
        ------
          - ValueError
              If all computed weights are zero (e.g., sigma too small).
        """
        diff = X - x_target
        dist2 = (diff ** 2).sum(axis=1)
        w = np.exp(-dist2 / (2 * self.sigma ** 2))

        if verbose:
            print("square dist_i between analogues:", dist2)
            print("weights w_i:", w)

        if not np.any(w):
            raise ValueError("All kernel weights are zero; try larger sigma.")
        return w