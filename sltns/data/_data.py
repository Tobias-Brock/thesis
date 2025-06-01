from typing import Any, Callable, Dict, List, Tuple

import numpy as np


class TimeSeriesSimulator:
    """Time series simulator for multiple processes.

    Generate, step through, and record values of various time series:
      - white_noise, random_walk, AR(1), unit_root, periodic, heteroskedastic.
    """

    configs: List[Dict[str, Any]]
    n: int
    rng: np.random.Generator
    t: int
    states: List[float]
    phases: List[float]
    variances: Dict[int, float]
    history: List[List[float]]

    def __init__(
        self,
        configs: list,
        n: int = 200,
        seed: int = 42,
    ):
        """Initialize the simulator and its state.

        Args:
            configs (list): Each dict may include:
                - kind (str): "white_noise", "random_walk", "ar1",
                  "unit_root", "periodic", "heteroskedastic".
                - sigma (float): Std of noise (default 1.0).
                - phi (float): AR(1) coefficient (default 0.7).
                - omega (float): Frequency for periodic (default 1.0).
                - phase (bool): Random phase if True (default True).
            n (int): Number of time steps to simulate.
            seed (int): Random seed for reproducibility.
        """
        self.configs = configs
        self.n = n
        self.rng = np.random.default_rng(seed)
        # time index
        self.t = 0
        # history of outputs per process
        self.history = [[] for _ in configs]
        # internal state for stateful processes
        self.states = [0.0] * len(configs)  # List[float]
        # random initial phase for periodic
        self.phases = [0.0] * len(configs)
        self._init_states()

    def _init_states(self) -> None:
        """Initialize states and phases for each process."""
        for i, cfg in enumerate(self.configs):
            kind = cfg.get("kind", "white_noise")
            sigma = cfg.get("sigma", 1.0)
            phi = cfg.get("phi", 0.7)
            if kind in ("random_walk", "unit_root", "ar1"):
                # draw initial noise for stateful processes
                eps0 = self.rng.normal(0, sigma)
                if kind == "ar1":
                    self.states[i] = eps0
                else:
                    self.states[i] = eps0
            if kind == "periodic":
                phase_flag = cfg.get("phase", True)
                self.phases[i] = self.rng.uniform(0, 2 * np.pi) if phase_flag else 0.0
            if kind == "arch":
                # start with unconditional var = sigma^2
                self.states[i] = 0.0  # last ε_t
                self.variances = getattr(self, "variances", {})
                self.variances[i] = sigma**2  # last σ²_t
            if kind == "garch":
                # same: track last residual and variance
                self.states[i] = 0.0
                self.variances = getattr(self, "variances", {})
                self.variances[i] = sigma**2
            if kind == "tv_ar1":
                # state holds last residual x_{t-1}
                self.states[i] = 0.0

    def step(self) -> list[float]:
        """Advance one time step, generate and record values for all processes.

        Returns:
            list[float]: Values at the current time for each process.
        """
        values: list[float] = []
        for i, cfg in enumerate(self.configs):
            kind = cfg.get("kind", "white_noise")
            sigma = float(cfg.get("sigma", 1.0))
            phi = float(cfg.get("phi", 0.7))

            if kind == "white_noise":
                y = self.rng.normal(0, sigma)

            elif kind == "random_walk":
                eps = self.rng.normal(0, sigma)
                y = self.states[i] + eps
                self.states[i] = y

            elif kind == "ar1":
                eps = self.rng.normal(0, sigma)
                y = phi * self.states[i] + eps
                self.states[i] = y

            elif kind == "periodic":
                omega = float(cfg.get("omega", 1.0))
                t = self.t
                phase0 = self.phases[i]
                y = np.cos(omega * t + phase0)

            elif kind == "heteroskedastic":
                t = self.t + 1
                eps = self.rng.normal(0, 1)
                y = sigma * np.sqrt(t) * eps

            elif kind == "arch":
                ω = float(cfg.get("omega", 0.1 * sigma**2))
                α = float(cfg.get("alpha", 0.8))
                ε_prev = self.states[i]
                σ2_prev = self.variances[i]
                σ2_t = ω + α * (ε_prev**2)
                ε_t = self.rng.normal(0, np.sqrt(σ2_t))
                y = ε_t
                self.states[i] = ε_t
                self.variances[i] = σ2_t

            elif kind == "garch":
                ω = float(cfg.get("omega", 0.1 * sigma**2))
                α = float(cfg.get("alpha", 0.1))
                β = float(cfg.get("beta", 0.8))
                ε_prev = self.states[i]
                σ2_prev = self.variances[i]
                σ2_t = ω + α * (ε_prev**2) + β * σ2_prev
                ε_t = self.rng.normal(0, np.sqrt(σ2_t))
                y = ε_t
                self.states[i] = ε_t
                self.variances[i] = σ2_t

            elif kind == "tv_ar1":
                # user must supply two callables in cfg:
                phi_trend: Callable[[float], float] = cfg["phi_trend"]
                mu_trend: Callable[[int], float] = cfg["mu_trend"]
                φ_t = phi_trend(self.t / self.n)
                μ_t = mu_trend(self.t)
                x_prev = self.states[i]
                eps = self.rng.normal(0, sigma)
                x_t = φ_t * x_prev + eps
                y = μ_t + x_t
                self.states[i] = x_t

            else:
                raise ValueError(f"Unknown kind: {kind!r}")

            self.history[i].append(y)
            values.append(y)

        self.t += 1
        return values

    def generate(self) -> np.ndarray:
        """Simulate all n time steps and return array of shape (n, num_processes)."""
        self.t = 0
        self.history = [[] for _ in self.configs]
        self._init_states()
        output = np.zeros((self.n, len(self.configs)))
        for ti in range(self.n):
            vals = self.step()
            output[ti] = vals
        return output

    def get_history(self) -> list:
        """Retrieve the list of recorded values for each process."""
        return [np.array(hist) for hist in self.history]


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    w_val: int,
    w_test: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time-series data into train, validation, and test sets.

    Splits:
      - Train: indices [0, n - w_val - w_test)
      - Val:   indices [n - w_val - w_test, n - w_test)
      - Test:  indices [n - w_test, n)

    Args:
        X (ndarray): Feature matrix of shape (n, d).
        y (ndarray): Target array of length n.
        w_val (int): Number of points to hold out for validation.
        w_test (int): Number of points to hold out for final test.

    Returns:
        Tuple containing:
          - X_train, y_train: Training data
          - X_val,   y_val:   Validation data
          - X_test,  y_test:  Test data
    """
    n = len(y)
    idx_train_end = n - w_val - w_test
    idx_val_end = n - w_test

    X_train = X[:idx_train_end]
    y_train = y[:idx_train_end]

    X_val = X[idx_train_end:idx_val_end]
    y_val = y[idx_train_end:idx_val_end]

    X_test = X[idx_val_end:]
    y_test = y[idx_val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test
