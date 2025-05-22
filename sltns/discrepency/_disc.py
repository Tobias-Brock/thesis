from typing import List, Sequence

import numpy as np
from numpy.typing import NDArray


def compute_disc(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    xt: NDArray[np.float64],
    yt: NDArray[np.float64],
    rng: np.random.Generator,
    outer_iters: int = 100,
    inner_iters: int = 1000,
    tol: float = 1e-5,
    lr: float = 0.1,
) -> float:
    """Compute a discrepancy measure.

    Compute discrepency measure between source domain (xs, ys) a target domain (xt, yt)
    via difference-of-convex (DC) programming.

    This function attempts to find a linear mapping that maximizes the
    difference between the mean-squared error on the source data and that on the
    target data, thus serving as an approximate discrepancy measure.

    Args:
        xs (NDArray[np.float_]): Source features of shape (m, d).
        ys (NDArray[np.float_]): Source labels of shape (m, 1).
        xt (NDArray[np.float_]): Target features of shape (n, d).
        yt (NDArray[np.float_]): Target labels of shape (n, 1).
        rng (np.random.Generator): Numpy random number generator.
        outer_iters (int): Number of outer iterations for DC programming.
        inner_iters (int): Number of gradient steps in each outer iteration.
        tol (float): Tolerance for improvement in objective value.
        lr (float): Learning rate for the inner gradient steps.

    Returns:
        float: The computed discrepancy estimate between source and target.
    """
    m = xs.shape[0]
    n = xt.shape[0]
    d = xs.shape[1]

    # Randomly initialize a weight vector w with norm 1
    w = rng.normal(size=(d, 1))
    w /= np.linalg.norm(w)

    outer_obj_val = np.inf
    loss_values: List[float] = []

    for _ in range(outer_iters):
        w0 = w
        w = rng.normal(size=(d, 1))
        w /= np.linalg.norm(w)

        ypred_s = xs @ w0
        ypred_t = xt @ w0

        curr_obj_val = (
            np.linalg.norm(ypred_s - ys) ** 2 / m
            - np.linalg.norm(ypred_t - yt) ** 2 / n
        )
        loss_values.append(curr_obj_val)

        if abs(curr_obj_val - outer_obj_val) <= tol:
            return curr_obj_val
        outer_obj_val = curr_obj_val

        # Inner optimization for w
        inner_obj_val = np.inf
        for _ in range(inner_iters):
            residual_s = (xs @ w - ys).squeeze()
            M_s = np.matmul(np.diag(residual_s), xs)
            grad_s = np.sum(M_s, axis=0) / m
            grad_s = grad_s.reshape(-1, 1)

            residual_t = (xt @ w0 - yt).squeeze()
            M_t = np.matmul(np.diag(residual_t), xt)
            grad_t = np.sum(M_t, axis=0) / n
            grad_t = grad_t.reshape(-1, 1)

            grad = grad_s - grad_t
            w -= lr * grad

            # Project w back to unit norm if needed
            norm_w = np.linalg.norm(w)
            if norm_w > 1:
                w /= norm_w

            # Evaluate new objective
            ypred_s_new = xs @ w
            ypred_t_old = xt @ w0
            ypred_t_new = xt @ w

            curr_inner_val = (
                0.5 * np.linalg.norm(ypred_s_new - ys) ** 2 / m
                - np.sum(ypred_t_old * ypred_t_new) / n
            )
            if abs(curr_inner_val - inner_obj_val) <= tol:
                break
            inner_obj_val = curr_inner_val

    return -outer_obj_val


def get_dbars(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    xt: NDArray[np.float64],
    yt: NDArray[np.float64],
    ms: Sequence[int],
    rng: np.random.Generator,
) -> List[float]:
    """Compute discrepancy values.

    Compute discrepancy values between multiple source chunks and single target domain.

    Slices the combined sources (xs, ys) into sub-sources according to the sizes
    in ms, and for each sub-source, calls compute_disc(...) against the target
    (xt, yt).

    Args:
        xs (NDArray[np.float_]): Source features with total shape (M, d),
            where M is the sum of all source chunk sizes.
        ys (NDArray[np.float_]): Source labels with total shape (M, 1).
        xt (NDArray[np.float_]): Target features of shape (N, d).
        yt (NDArray[np.float_]): Target labels of shape (N, 1).
        ms (Sequence[int]): Sizes of each source domain chunk, followed by the
            target size. The last element typically corresponds to the size of
            the target data.
        rng (np.random.Generator): Numpy random number generator.

    Returns:
        List[float]: A list of absolute discrepancy values for each source
        chunk, compared to the target.
    """
    dbars_drift: List[float] = []
    # len(ms) - 1 because the last item of ms is typically the target chunk
    for t in range(len(ms) - 1):
        n_t = int(np.sum(ms[:t]))
        x_slice = xs[n_t : n_t + ms[t]]
        y_slice = ys[n_t : n_t + ms[t]]
        t_disc = compute_disc(xs=x_slice, ys=y_slice, xt=xt, yt=yt, rng=rng)
        dbars_drift.append(abs(t_disc))
    return dbars_drift
