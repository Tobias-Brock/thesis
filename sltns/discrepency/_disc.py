import numpy as np
import scipy.linalg as la


class DiscrepancyEstimator:
    """Compute per-sample discrepancies d_i = sup_{w:||w||<=1} |a(w) - l_i(w)|.

    Provides two computation modes:
      - TRS (exact via trust-region eigenvalues)
      - DCP (approximate via Convex–Concave Procedure)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        trs_plus_only: bool = False,
    ):
        """Initialize estimator with data and prior weights.

        Args:
            X (np.ndarray): Feature matrix of shape (n, d).
            y (np.ndarray): Target vector of shape (n,).
            p (np.ndarray): Prior weights of shape (n,), summing to 1.
            trs_plus_only (bool):
                If True, in the TRS method only return the largest eigenvalue
                of A_s - x_i x_i^T (S_plus) instead of max(S_plus, S_minus).
        """
        self.X = X
        self.y = y
        self.p = p
        self.n, self.d = X.shape
        self.A_s = X.T @ (p[:, None] * X)
        self.b_s = X.T @ (p * y)
        self.trs_plus_only = trs_plus_only

    def di_trs(self, i: int) -> float:
        """Compute d_i exactly via trust-region eigenvalue solver.

        Args:
            i (int): Index of the sample.

        Returns:
            float: Exact discrepancy d_i.
        """
        xi = self.X[i : i + 1]  # 1×d
        yi = self.y[i]

        M = self.A_s - xi.T @ xi  # d×d
        c = (-2 * self.b_s) + 2 * yi * xi.flatten()  # length-d

        top = np.hstack([M, (c / 2)[:, None]])
        bottom = np.hstack([(c / 2)[None, :], np.zeros((1, 1))])
        K = np.vstack([top, bottom])

        ev = la.eigvalsh(K)
        λ_plus = ev[-1]  # sup    f(w)
        if self.trs_plus_only:
            return float(λ_plus)
        λ_minus = -ev[0]  # sup -f(w)  = - inf f(w)
        return float(max(λ_plus, λ_minus))

    def all_di_trs(self) -> np.ndarray:
        """Compute all d_i via exact TRS method.

        Returns:
            np.ndarray: Array of shape (n,) of d_i values.
        """
        return np.array([self.di_trs(i) for i in range(self.n)])

    def all_di_dcp(
        self,
        outer_iters: int = 20,
        inner_iters: int = 100,
        lr: float = 0.1,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Compute all d_i via DCP approximation.

        Args:
            outer_iters (int): DCP outer iterations.
            inner_iters (int): Gradient steps per surrogate.
            lr (float): Learning rate for inner ascent.
            tol (float): Convergence tolerance.

        Returns:
            np.ndarray: Array of shape (n,) of d_i values.
        """
        return np.array(
            [
                self.di_dcp(
                    i, outer_iters=outer_iters, inner_iters=inner_iters, lr=lr, tol=tol
                )
                for i in range(self.n)
            ]
        )

    def di_dcp(
        self,
        i: int,
        outer_iters: int = 20,
        inner_iters: int = 100,
        lr: float = 0.1,
        tol: float = 1e-6,
    ) -> float:
        """Approximate d_i via DC programming (Convex–Concave Procedure).

        Args:
            i (int): Index of the sample.
            outer_iters (int): DCP outer iterations.
            inner_iters (int): Gradient steps per surrogate.
            lr (float): Learning rate for inner ascent.
            tol (float): Convergence tolerance.

        Returns:
            float: Approximated discrepancy d_i.
        """
        xi = self.X[i : i + 1]
        yi = self.y[i]
        # Initialize w on unit sphere
        w = np.random.default_rng(0).normal(size=(self.d, 1))
        w /= np.linalg.norm(w)

        def g_grad(w):
            return 2 * self.A_s @ w

        def h_grad(w):
            return 2 * xi.T @ (xi @ w - yi)

        prev_phi = -np.inf
        for _ in range(outer_iters):
            grad_hi = h_grad(w)
            for _ in range(inner_iters):
                w += lr * (g_grad(w) - grad_hi)
                w /= max(1.0, np.linalg.norm(w))
            phi = float((w.T @ self.A_s @ w) - ((xi @ w - yi) ** 2))
            if abs(phi - prev_phi) < tol:
                break
            prev_phi = phi
        return abs(prev_phi)
