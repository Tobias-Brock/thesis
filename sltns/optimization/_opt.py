from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


def altmin(
    xs: NDArray[np.float64],
    xt: NDArray[np.float64],
    ys: NDArray[np.float64],
    yt: NDArray[np.float64],
    x_trgt_test: NDArray[np.float64],
    y_trgt_test: NDArray[np.float64],
    x_trgt_val: NDArray[np.float64],
    y_trgt_val: NDArray[np.float64],
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
    dbar: List[float],
    p0: NDArray[np.float64],
    lr: float,
    ms: List[int],
    maxiters: int = 100,
    niters: int = 1000,
    q_init: Optional[NDArray[np.float64]] = None,
    tol: float = 1e-3,
    hnorm_mode: str = "original",
) -> Tuple[float, NDArray[np.float64], List[float]]:
    """Alternate minimization procedure.

    Run an alternating-minimization algorithm to learn a model for target data
    based on multiple sources and their discrepancy to the target. Reports the
    resulting test error.

    Args:
        xs (NDArray[np.float64]): Source feature data, shape (m, d).
        xt (NDArray[np.float64]): Target feature data, shape (n, d).
        ys (NDArray[np.float64]): Source label data, shape (m, 1).
        yt (NDArray[np.float64]): Target label data, shape (n, 1).
        x_trgt_test (NDArray[np.float64]): Target feature test data.
        y_trgt_test (NDArray[np.float64]): Target label test data.
        x_trgt_val (NDArray[np.float64]): Target feature validation data.
        y_trgt_val (NDArray[np.float64]): Target label validation data.
        lambda_1 (float): Regularizer term for the infinity norm of q.
        lambda_2 (float): Regularizer term for the L1 distance from p0.
        lambda_3 (float): Regularizer term for the squared L2 norm of q.
        dbar (List[float]): Discrepancy values between each source and target.
        p0 (NDArray[np.float64]): Starting distribution over all data points.
        lr (float): Learning rate for gradient descent updates on q.
        ms (List[int]): Sizes of each source domain plus target domain.
        maxiters (int): Maximum number of alternating-minimization iterations.
        niters (int): Maximum number of q-updates per outer iteration.
        q_init (Optional[NDArray[np.float64]]): Initial q distribution.
        tol (float): Tolerance on objective value changes for stopping.
        hnorm_mode (str): {'original','force_qmax=1','omit_hnorm'}:
        - 'original': add lambda_1 * hnorm * np.max(q)
        - 'force_qmax=1': add lambda_1 * hnorm * 1
        - 'omit_hnorm': skip that term

    Returns:
        Tuple[float, NDArray[np.float64]]:
            - test_error: Final test error on (x_trgt_test, y_trgt_test).
            - q: The final distribution over source+target data.
            - loss_values: List of objective values during optimization.
    """
    print("Calling altmin")

    m = xs.shape[0]
    n = xt.shape[0]
    x = np.vstack((xs, xt)).astype(np.float64)
    y = np.vstack((ys, yt)).astype(np.float64)

    # Initialize q
    if q_init is None:
        rng = np.random.default_rng(seed=42)
        q_rand = rng.uniform(size=m + n).astype(np.float64)
        q = -np.log(q_rand)
        q /= np.sum(q)
    else:
        q = q_init.astype(np.float64)

    loss_values: List[float] = []
    prev_obj = np.inf
    T_counter = 0
    best_h = []

    # Outer loop of alt-min
    for i in range(maxiters):
        # Fit a ridge model with current distribution q
        indices = [j for j in range(m + n) if q[j] > 0]
        if hnorm_mode == "original":
            lambda_ridge = lambda_1 * float(np.max(q))
        elif hnorm_mode == "force_qmax=1":
            lambda_ridge = lambda_1 * 1.0
        elif hnorm_mode == "omit_hnorm":
            lambda_ridge = lambda_1 * 0
        clf = Ridge(alpha=lambda_ridge, solver="cholesky")
        clf.fit(x[indices, :], y[indices], sample_weight=q[indices])

        curr_h = clf.coef_
        ypred = clf.predict(x)

        # Compute objective
        curr_obj = compute_obj_val(
            xs=xs,
            xt=xt,
            ys=ys,
            yt=yt,
            ypred=ypred,
            q=q,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            p0=p0,
            dbars=dbar,
            hnorm=float(np.linalg.norm(curr_h) ** 2),
            ms=ms,
            hnorm_mode=hnorm_mode,
        )
        print(f"Current objective value: {curr_obj}")

        # Convergence check
        if abs(curr_obj - prev_obj) <= tol:
            best_h.append(clf)
            T_counter += 1
        else:
            T_counter = 0

        if i == maxiters - 1:
            best_h.append(clf)

        prev_obj = curr_obj
        if T_counter == 5:
            break

        loss_values.append(curr_obj)

        # Update q via gradient descent
        init_lr = lr
        best_inner_obj = curr_obj
        best_q = q
        # iloss: per-point squared error, used in gradient
        iloss = (y - ypred.reshape(-1, 1)) ** 2

        for j in range(niters):
            grad = get_gradient(
                xs=xs,
                xt=xt,
                q=q,
                loss=iloss.squeeze(),
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                p0=p0,
                hnorm=float(np.linalg.norm(curr_h) ** 2),
                dbars=dbar,
                ms=ms,
                hnorm_mode=hnorm_mode,
            )
            norm_grad = np.linalg.norm(grad)
            if norm_grad > 0:
                grad /= norm_grad

            q = q - init_lr * grad / (j + 1)
            q = project_simplex(q)
            obj = compute_obj_val(
                xs=xs,
                xt=xt,
                ys=ys,
                yt=yt,
                ypred=ypred,
                q=q,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                lambda_3=lambda_3,
                p0=p0,
                dbars=dbar,
                hnorm=float(np.linalg.norm(curr_h) ** 2),
                ms=ms,
                hnorm_mode=hnorm_mode,
            )
            if obj < best_inner_obj:
                best_inner_obj = obj
                best_q = q

        q = best_q

    # Final refit with best distribution
    indices = [j for j in range(m + n) if q[j] > 0]
    clf = Ridge(solver="cholesky")
    clf.fit(x[indices, :], y[indices], sample_weight=q[indices])
    best_h.append(clf)

    # Validation-based model selection
    best_error = np.inf
    best_model = None
    for candidate_clf in best_h:
        ypred_val = candidate_clf.predict(x_trgt_val)
        error_val = np.linalg.norm(y_trgt_val - ypred_val) / len(ypred_val)
        if error_val < best_error:
            best_error = error_val
            best_model = candidate_clf

    if best_model is None:  # fallback if no model found
        best_model = clf

    # Final test error
    ypred_test = best_model.predict(x_trgt_test)
    test_error = (np.linalg.norm(y_trgt_test - ypred_test) ** 2) / len(ypred_test)
    print(f"test error = {test_error}")

    return test_error, q, loss_values


def compute_obj_val(
    xs: NDArray[np.float64],
    xt: NDArray[np.float64],
    ys: NDArray[np.float64],
    yt: NDArray[np.float64],
    ypred: NDArray[np.float64],
    q: NDArray[np.float64],
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
    p0: NDArray[np.float64],
    dbars: List[float],
    hnorm: float,
    ms: List[int],
    hnorm_mode: str = "original",
) -> float:
    """Objective value computation.

    Compute the value of the custom objective given a distribution q, a
    regression predictor (ypred), and regularization terms.

    Args:
        xs (NDArray[np.float64]): Source features (m, d).
        xt (NDArray[np.float64]): Target features (n, d).
        ys (NDArray[np.float64]): Source labels (m, 1).
        yt (NDArray[np.float64]): Target labels (n, 1).
        ypred (NDArray[np.float64]): Model predictions on all data (m+n,).
        q (NDArray[np.float64]): Current distribution over data points.
        lambda_1 (float): Infinity-norm regularizer scale.
        lambda_2 (float): Distance-from-p0 regularizer scale.
        lambda_3 (float): L2 norm of q regularizer scale.
        p0 (NDArray[np.float64]): Initial distribution over data.
        dbars (List[float]): Discrepancy values between each source and target.
        hnorm (float): Squared L2 norm of the regression predictor.
        ms (List[int]): Sizes of each source domain plus target domain.
        hnorm_mode (str): {'original','force_qmax=1','omit_hnorm'}:
        - 'original': add lambda_1 * hnorm * np.max(q)
        - 'force_qmax=1': add lambda_1 * hnorm * 1
        - 'omit_hnorm': skip that term

    Returns:
        float: The value of the objective function at the given parameters.
    """
    m = xs.shape[0]
    n = xt.shape[0]
    M = m + n

    y = np.vstack((ys, yt)).astype(np.float64)
    total_loss = 0.0

    for i in range(M):
        diff = y[i] - ypred[i]
        total_loss += (diff**2) * q[i]

    # Add regularizers
    total_loss += lambda_3 * float(np.linalg.norm(q) ** 2)
    total_loss += lambda_2 * float(np.linalg.norm(q - p0, ord=1))

    if hnorm_mode == "original":
        total_loss += lambda_1 * hnorm * float(np.max(q))
    elif hnorm_mode == "force_qmax=1":
        total_loss += lambda_1 * hnorm * 1.0
    elif hnorm_mode == "omit_hnorm":
        pass  # skip

    # Add discrepancy-based penalty
    for t in range(len(ms) - 1):
        n_t = int(np.sum(ms[:t]))
        q_t_bar = np.sum(q[n_t : n_t + ms[t]])
        total_loss += dbars[t] * q_t_bar

    return total_loss


def get_gradient(
    xs: NDArray[np.float64],
    xt: NDArray[np.float64],
    q: NDArray[np.float64],
    loss: NDArray[np.float64],
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
    p0: NDArray[np.float64],
    hnorm: float,
    dbars: List[float],
    ms: List[int],
    hnorm_mode: str = "original",
) -> NDArray[np.float64]:
    """Compute the gradient of the objective with respect to q.

    Args:
        xs (NDArray[np.float64]): Source feature data (m, d).
        xt (NDArray[np.float64]): Target feature data (n, d).
        q (NDArray[np.float64]): Current distribution over data points (m+n,).
        loss (NDArray[np.float64]): Per-point squared error, shape (m+n,).
        lambda_1 (float): Infinity-norm regularizer scale.
        lambda_2 (float): L1 distance-from-p0 regularizer scale.
        lambda_3 (float): L2 norm of q regularizer scale.
        p0 (NDArray[np.float64]): The initial distribution (m+n,).
        hnorm (float): Squared L2 norm of the regression predictor.
        dbars (List[float]): Discrepancy values for each source chunk.
        ms (List[int]): Source domain sizes plus target domain size.
        ms (List[int]): Sizes of each source domain plus target domain.
        hnorm_mode (str): {'original','force_qmax=1','omit_hnorm'}:
        - 'original': add lambda_1 * hnorm * np.max(q)
        - 'force_qmax=1': add lambda_1 * hnorm * 1
        - 'omit_hnorm': skip that term

    Returns:
        NDArray[np.float64]: The gradient of the objective with respect to q.
    """
    m = xs.shape[0]
    n = xt.shape[0]
    M = m + n

    # Loss part of gradient
    loss_grad = loss

    # dbar part of gradient
    dbar_grad = np.zeros(M, dtype=np.float64)
    for t in range(len(ms) - 1):
        n_t = int(np.sum(ms[:t]))
        dbar_grad[n_t : n_t + ms[t]] = dbars[t]

    # lambda_1 part of gradient
    lambda_1_grad = np.zeros(M, dtype=np.float64)
    if hnorm_mode == "original":
        i1 = np.argmax(q)
        lambda_1_grad[i1] = lambda_1 * hnorm
    elif hnorm_mode in ["force_qmax=1", "omit_hnorm"]:
        # gradient is 0 because it's a constant wrt q
        pass

    # lambda_2 part: L1 distance from p0
    lambda_2_grad = lambda_2 * np.sign(q - p0)

    # lambda_3 part: L2 norm of q
    lambda_3_grad = 2.0 * lambda_3 * q

    grad = loss_grad + dbar_grad + lambda_1_grad + lambda_2_grad + lambda_3_grad
    return grad.astype(np.float64)


def project_simplex(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project a vector q onto the probability simplex.

    Args:
        q (NDArray[np.float64]): A vector to be projected.

    Returns:
        NDArray[np.float64]: The projection of q onto the simplex.
    """
    d = q.shape[0]
    # Sort in descending order
    qprime = -np.sort(-q)

    K = 1
    sum_list = [qprime[K - 1]]
    for i in range(2, d + 1):
        sum_list.append(sum_list[-1] + qprime[i - 1])
        if (sum_list[-1] - 1.0) / i < qprime[i - 1]:
            K = i

    tau = (sum_list[K - 1] - 1.0) / K
    q_proj = q - tau
    q_proj = np.maximum(q_proj, 0.0)
    q_proj /= np.sum(q_proj)
    return q_proj
