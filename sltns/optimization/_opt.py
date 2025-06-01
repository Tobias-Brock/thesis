from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sltns.discrepency._disc import DiscrepancyEstimator


def project_simplex(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project a vector v onto the probability simplex (sum to 1, non-negative).

    Args:
        v (NDArray[np.float64]): Input vector.

    Returns:
        NDArray[np.float64]: Projected vector.
    """
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    q = np.maximum(v - theta, 0)
    return q


class MDRIFT:
    """MDRIFT: Alternating minimization algorithm for Φ(h, q).

    Implements alternating updates between a ridge model h and a distribution q,
    minimizing Φ(h,q) = ∑ᵢ qᵢ(h(Xᵢ)-yᵢ)² + ∑ᵢ qᵢ dᵢ
                       + λ₂||q_src||₂² + λ₃||q_src - p_src||_{norm}

    Args:
        lambda_1 (float):
            Ridge regularization weight for fitting h.
        lambda_2 (float):
            L2 penalty coefficient on q.
        lambda_3 (float):
            Penalty coefficient on ||q - p|| term.
        eta (float):
            Base learning rate for q updates.
        delta (float):
            Relative tolerance for stopping outer iterations.
        max_outer (int):
            Maximum number of (h,q) outer loops.
        max_inner (int):
            Maximum q-update steps per outer iteration.
        pq_norm ({'l1','l2'}):
            Norm type for the q-p penalty.
        verbose (bool):
            If True, print iteration diagnostics.
        patience: int = 5,
    """

    def __init__(
        self,
        lambda_1: float = 1e-3,
        lambda_2: float = 1e-2,
        lambda_3: float = 1e-2,
        eta: float = 1.0,
        delta: float = 1e-4,
        max_outer: int = 100,
        max_inner: int = 100,
        pq_norm: Literal["l1", "l2"] = "l1",
        verbose: bool = True,
        tol_inner: float = 1e-4,
        patience: int = 5,
    ):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.eta = eta
        self.delta = delta
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.pq_norm = pq_norm
        self.verbose = verbose
        self.tol_inner = tol_inner
        self.patience = patience

    def Phi(
        self,
        model: Ridge,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        d: NDArray[np.float64],
        p: NDArray[np.float64],
        q_vec: NDArray[np.float64],
    ) -> float:
        """Compute the objective Φ(h, q) at given parameters.

        Args:
            model (Ridge):
                Trained Ridge model h.
            X (ndarray):
                Feature matrix of shape (n, d).
            y (ndarray):
                Target vector of length n.
            d (ndarray):
                Discrepancy per point.
            p (ndarray):
                Prior distribution over points.
            q_vec (ndarray):
                Current sampling distribution q.

        Returns:
            float: Current objective value Φ(h,q).
        """
        pq_fn = np.linalg.norm if self.pq_norm == "l2" else lambda v: np.abs(v).sum()

        preds = model.predict(X)
        loss = np.sum(q_vec * (preds - y) ** 2)
        disc = np.dot(q_vec, d)
        reg_q = self.lambda_2 * np.dot(q_vec, q_vec)
        reg_pq = self.lambda_3 * pq_fn(q_vec - p)
        return float(loss + disc + reg_q + reg_pq)

    def compute_q_grad(
        self,
        preds: np.ndarray,
        y: np.ndarray,
        d: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute gradient and step size for updating q.

        Args:
            preds (ndarray): Predictions h.predict(X), shape (n,).
            y (ndarray):    True targets, shape (n,).
            d (ndarray):    Discrepancy per point, shape (n,).
            p (ndarray):    Prior distribution, shape (n,).
            q (ndarray):    Current q, shape (n,).

        Returns:
            grad (ndarray): Full gradient vector, shape (n,).
            step (float):   Adaptive step size = eta / ||grad_src||₂.
        """
        grad = (preds - y) ** 2 + d + 2 * self.lambda_2 * q
        diff = q - p
        if self.pq_norm == "l2":
            norm = np.linalg.norm(diff)
            if norm > 0:
                grad += self.lambda_3 * diff / norm
        else:
            norm = np.abs(diff).sum()
            if norm > 0:
                grad += self.lambda_3 * np.sign(diff)
        # step size
        gnorm = np.linalg.norm(grad) + 1e-12
        step = self.eta / gnorm
        return grad, step

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        d: NDArray[np.float64],
        p: NDArray[np.float64],
        X_val: NDArray | None = None,
        y_val: NDArray | None = None,
    ) -> tuple[Ridge, NDArray[np.float64], List[float]]:
        """Run alternating minimization and return final model, q, and Φ trace.

        Steps:
        1) Initialize q uniform on source.
        2) Fit h via Ridge with weight q.
        3) Update q by gradient + projection onto simplex.
        4) Repeat until Φ converges or max_outer reached.
        5) If X_val and y_val are provided, use them for early stopping.

        Args:
            X (ndarray):
                Feature matrix of shape (n, d).
            y (ndarray):
                Target vector of length n.
            d (ndarray):
                Discrepancy values per point (length n).
            p (ndarray):
                Prior distribution of length n.
            X_val (ndarray, optional):
                Validation feature matrix of shape (m, d) used for early stopping.
            y_val (ndarray, optional):
                Validation target vector of length m used for early stopping.

        Returns:
            h_final (Ridge):
                Ridge model refit on the learned q.
            q (ndarray):
                Learned sampling distribution of length n.
            phi_trace (list of float):
                Objective values Φ after each outer iteration.
        """
        n, _ = X.shape
        q = np.full(n, 1.0 / n)
        h = Ridge(alpha=self.lambda_1, solver="cholesky").fit(X, y)
        phi_prev = self.Phi(h, X, y, d, p, q)
        trace = [phi_prev]
        if self.verbose:
            print(f"[init] Φ = {phi_prev:.6f}")

        best_val = float("inf")
        best_state = None
        epochs_since = 0
        for outer in range(1, self.max_outer + 1):
            h = Ridge(alpha=self.lambda_1, solver="cholesky")
            h.fit(X, y, sample_weight=q)
            phi_old = self.Phi(h, X, y, d, p, q)

            for inner in range(1, self.max_inner + 1):
                preds = h.predict(X)
                grad, step = self.compute_q_grad(preds, y, d, p, q)
                q_c = q - step * grad
                q_c = project_simplex(q_c)
                phi_new = self.Phi(h, X, y, d, p, q_c)
                # accept if Φ has *any* improvement larger than tol_inner
                if phi_old - phi_new >= self.tol_inner:
                    q, phi_old = q_c, phi_new
                else:
                    break
                if self.verbose and inner % 50 == 0:
                    print(f"    inner {inner}: step={step:.2e}")

            # Evaluate
            phi_curr = self.Phi(h, X, y, d, p, q)
            if X_val is not None and y_val is not None:
                val_rmse = float(np.sqrt(mean_squared_error(y_val, h.predict(X_val))))
                if val_rmse < best_val:
                    best_val = val_rmse
                    best_state = (h, q.copy())
                    epochs_since = 0
                else:
                    epochs_since += 1
                    if epochs_since >= self.patience:
                        if self.verbose:
                            print("  ↳ early-stop (val no improvement)")
                        break
            rel = abs(phi_curr - phi_prev) / (phi_prev + 1e-12)
            if self.verbose:
                msg = f"[outer {outer}] Φ={phi_curr:.6f} Δrel={rel:.2e}"
                print(msg)
            trace.append(phi_curr)
            if rel <= self.delta:
                if self.verbose:
                    print("  ↳ stopping tol reached")
                break
            phi_prev = phi_curr

        if best_state is not None:
            h_final, q = best_state  # roll back to best validation state
        else:
            h_final = h
        return h_final, q, trace


class MDRIFTTuner:
    """Hyperparameter tuner for the MDRIFT algorithm.

    Supports both exhaustive grid search and randomized search over:
      - lambda_2 (L2 q penalty)
      - lambda_3 (q-p penalty)
      - eta (q update step size)

    Args:
        param_grid (dict):
            Dictionary specifying lists of candidate values:
              - 'prior': list of (begin,end) tuples (1-based, inclusive)
              - 'lambda_2': list of floats
              - 'lambda_3': list of floats
              - 'eta': list of floats
        fixed_kwargs (dict):
            Additional fixed arguments passed to MDRIFT (e.g., lambda_1,
            delta, max_outer, max_inner, pq_norm, verbose).
        random_state (int or None): Seed for reproducible random search sampling.
        slice_prior (bool, optional): Slice prior from validation set if True.
            Defaults to False.
    """

    def __init__(
        self,
        param_grid: Dict[str, Sequence[Any]],
        fixed_kwargs: Dict[str, Any],
        full_n: int,
        random_state: int | None = None,
        trs_plus_only: bool = False,
        full_refit: bool = True,
        slice_prior: bool = False,
    ) -> None:
        self.param_grid = param_grid
        tune_keys = {"lambda_1", "lambda_2", "lambda_3", "eta", "pq_norm", "max_inner"}
        self.base_kwargs = {k: v for k, v in fixed_kwargs.items() if k not in tune_keys}
        self.full_n = full_n
        self.rng = np.random.default_rng(random_state)
        self.trs_plus_only = trs_plus_only
        self.full_refit = full_refit
        self.slice_prior = slice_prior

    def _build_prior(self, n: int, k: int) -> np.ndarray:
        """Uniform on the last k of the first (n) rows."""
        k = min(k, n)  # clip if k>n_source
        p = np.zeros(n)
        start = n - k
        p[start:n] = 1.0 / k
        return p

    def _fit_and_score(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prior_len: int,
        l1: float,
        l2: float,
        l3: float,
        eta: float,
        pq_norm: str,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[float, Any]:
        """Fit MDRIFT with specified hyperparameters and evaluate on validation set.

        Args:
            X (ndarray): Training feature matrix of shape (n, d).
            y (ndarray): Training target vector of length n.
            prior_len (int): Number of most recent source points in the prior support.
            l1 (float): Regularization weight λ₁.
            l2 (float): Regularization weight λ₂.
            l3 (float): Regularization weight λ₃.
            eta (float): Step size for the q‐update.
            pq_norm (str): Norm type for (p, q) regularization, e.g., "l1" or "l2".
            x_val (ndarray): Validation feature matrix of shape (m, d).
            y_val (ndarray): Validation target vector of length m.

        Returns:
            validation_rmse (float): RMSE computed on the validation set.
            model (Any): Fitted MDRIFT model instance.
        """
        n = X.shape[0]

        if self.slice_prior:
            p_full = self._build_prior(n=self.full_n, k=prior_len)
            p0 = p_full[:n]
            total = p0.sum()
            if total <= 0:
                msg = (
                    f"After slicing, prior is zero "
                    f"(n={n}, full_n={self.full_n}, prior_len={prior_len})."
                )
                raise ValueError(msg)
            p0 /= total
        else:
            p0 = self._build_prior(n=n, k=prior_len)

        d_full = DiscrepancyEstimator(
            X=X, y=y, p=p0, trs_plus_only=self.trs_plus_only
        ).all_di_trs()

        run_kwargs = self.base_kwargs.copy()
        run_kwargs.update(
            {
                "lambda_1": l1,
                "lambda_2": l2,
                "lambda_3": l3,
                "eta": eta,
                "pq_norm": pq_norm,
            }
        )
        md = MDRIFT(**run_kwargs)
        h, _, _ = md.fit(X=X, y=y, d=d_full, p=p0, X_val=x_val, y_val=y_val)

        y_pred = h.predict(x_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        return rmse, h

    def random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        n_iter: int = 20,
    ) -> Tuple[Any, Dict[str, Any], float]:
        """Sample n_iter random hyperparameter combos and evaluate.

        Args:
            X (ndarray): Full feature data.
            y (ndarray): Full target data.
            x_val (ndarray): Validation features.
            y_val (ndarray): Validation targets.
            n_iter (int): Number of random combinations to try.

        Returns:
            best_model: MDRIFT model with lowest validation RMSE.
            best_params (dict): Hyperparameters for best_model.
            best_rmse (float): Validation RMSE of best_model.
        """
        best_rmse = float("inf")
        best_model = None
        best_params: Dict[str, Any] = {}
        all_k = self.param_grid["prior_len"]
        all_l1 = self.param_grid["lambda_1"]
        all_l2 = self.param_grid["lambda_2"]
        all_l3 = self.param_grid["lambda_3"]
        all_eta = self.param_grid["eta"]
        all_pq = self.param_grid["pq_norm"]

        for _ in range(n_iter):
            k = int(self.rng.choice(all_k))
            l1 = float(self.rng.choice(all_l1))
            l2 = float(self.rng.choice(all_l2))
            l3 = float(self.rng.choice(all_l3))
            eta = float(self.rng.choice(all_eta))
            pq_norm = self.rng.choice(all_pq)

            rmse, model = self._fit_and_score(
                X=X,
                y=y,
                prior_len=k,
                l1=l1,
                l2=l2,
                l3=l3,
                eta=eta,
                pq_norm=pq_norm,
                x_val=x_val,
                y_val=y_val,
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_params = {
                    "prior_len": k,
                    "lambda_1": l1,
                    "lambda_2": l2,
                    "lambda_3": l3,
                    "eta": eta,
                    "pq_norm": pq_norm,
                }

        return best_model, best_params, best_rmse

    def refit_on_full(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        best_params: Dict[str, Any],
    ) -> Tuple[Any, np.ndarray, list]:
        """Refit MDRIFT on combined train+val set or with validation early stopping.

        Args:
            X_train (ndarray): Training features (n_train, d).
            y_train (ndarray): Training targets (n_train,).
            X_val   (ndarray): Validation features (n_val, d).
            y_val   (ndarray): Validation targets (n_val,).
            best_params (dict):
                Should include keys:
                - 'prior_len' : int  ← how many of the last source points to weigh
                - 'lambda_1', 'lambda_2', 'lambda_3', 'eta', 'pq_norm'

        Returns:
            h_final: MDRIFT Ridge model fitted according to `use_full_data`.
            q_final: Learned sampling distribution
                    (length = n_train + n_val + w).
            trace  : List of Φ values over outer iterations.
        """
        run_kwargs = self.base_kwargs.copy()
        run_kwargs.update(
            {
                "lambda_1": best_params["lambda_1"],
                "lambda_2": best_params["lambda_2"],
                "lambda_3": best_params["lambda_3"],
                "eta": best_params["eta"],
                "pq_norm": best_params["pq_norm"],
            }
        )
        md = MDRIFT(**run_kwargs)
        k = int(best_params["prior_len"])

        if self.full_refit:
            # merge train + val (if provided) and fit on full
            if X_val is not None and y_val is not None:
                X_full = np.concatenate((X_train, X_val), axis=0)
                y_full = np.concatenate((y_train, y_val), axis=0)
            else:
                X_full, y_full = X_train, y_train

            n_full = X_full.shape[0]
            p0 = self._build_prior(n_full, k)
            d_full = DiscrepancyEstimator(
                X=X_full, y=y_full, p=p0, trs_plus_only=self.trs_plus_only
            ).all_di_trs()

            h_final, q_final, trace = md.fit(
                X=X_full,
                y=y_full,
                d=d_full,
                p=p0,
            )
        else:
            if X_val is None or y_val is None:
                raise ValueError(
                    "X_val and y_val must be provided when use_full_data=False"
                )

            n_train = X_train.shape[0]
            p0 = self._build_prior(n_train, k)
            d_train = DiscrepancyEstimator(
                X=X_train, y=y_train, p=p0, trs_plus_only=self.trs_plus_only
            ).all_di_trs()

            h_final, q_final, trace = md.fit(
                X=X_train,
                y=y_train,
                d=d_train,
                p=p0,
                X_val=X_val,
                y_val=y_val,
            )

        return h_final, q_final, trace
