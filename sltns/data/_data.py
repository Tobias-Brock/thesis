from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def generate_simulated_data(
    d: int,
    ms: List[int],
    eps1: List[float],
    eps2: List[float],
    rng: np.random.Generator,
    alpha: float = 0.5,
    sigma: float = 0.1,
    random_w: bool = False,
    normalize_w_t: bool = True,
    feature_means: Optional[List[float]] = None,
    beta_mixing: bool = False,
    beta_value: float = 0.0,
    weight_loc: float = 0.0,
    noise_loc: float = 0.0,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
]:
    """Simulation setup.

    Data simulation for D_1..D_{T+1} Gaussian distributions, with optional:
      - Different feature means,
      - Simple AR(1)-like beta mixing across time steps,
      - Mean shift for weight vectors,
      - Mean shift for label noise.

    Args:
      d (int):
        Dimension of each data point.
      ms (List[int]):
        Sample sizes for T+1 distributions (first T are sources, last is target).
      eps1 (List[float]):
        Std devs for feature generation of each distribution (length T+1).
      eps2 (List[float]):
        Magnitudes for how each source's w differs from the target's w (length T).
      rng (np.random.Generator):
        Random generator for reproducibility.
      alpha (float):
        Fraction of each D_i replaced by heavy noise from index alpha*m_i onward.
      sigma (float):
        Std dev of label noise (but possibly with mean = noise_loc).
      random_w (bool):
        If True, re-draw base_w for each source; otherwise reuse one base_w.
      normalize_w_t (bool):
        If True, re-normalize w_i after w_i = w_target + eps2[i] * w_base.
      feature_means (Optional[List[float]]):
        Per-distribution means for the features (length T+1). Defaults to 0.0 each.
      beta_mixing (bool):
        If True, apply AR(1)-like correlation across time steps in the features.
      beta_value (float):
        Beta parameter in [0,1) for the AR(1)-like correlation if beta_mixing=True.
      weight_loc (float):
        Mean shift for weight-vector draws (both target and base). By default 0.0.
      noise_loc (float):
        Mean shift for label noise draws. By default 0.0.

    Returns:
      (x_d1T_train, y_d1T_train,
       x_trgt_train, y_trgt_train,
       x_trgt_test, y_trgt_test,
       x_trgt_val, y_trgt_val,
       x_d1T_list, y_d1T_list)

      - x_d1T_train, y_d1T_train:
          Flattened source features/labels over T distributions.
      - x_trgt_train, y_trgt_train:
          Target train data/labels.
      - x_trgt_test, y_trgt_test:
          Target test data/labels.
      - x_trgt_val, y_trgt_val:
          Target validation data/labels.
      - x_d1T_list, y_d1T_list:
          Per-source-chunk features/labels (lists of arrays),
          shape (ms[i], d) and (ms[i],1) for each i in [0..T-1].
    """
    T = len(ms) - 1
    m1T_sum = np.sum(ms[:T])

    if feature_means is None:
        feature_means = [0.0] * (T + 1)
    else:
        if len(feature_means) != (T + 1):
            raise ValueError("feature_means must have length T+1 or be None.")

    # 2) Generate feature data for sources D_1..D_T
    x_d1T_list: List[NDArray[np.float64]] = []
    X_prev: Optional[NDArray[np.float64]] = None

    for i in range(T):
        x_i = rng.normal(loc=0.0, scale=eps1[i], size=(ms[i], d)).astype(np.float64)
        if feature_means[i] != 0.0:
            x_i += feature_means[i]  # shift entire distribution by feature_means[i]

        # If beta_mixing => incorporate fraction of previous distribution
        if beta_mixing and i > 0 and X_prev is not None:
            x_i = beta_value * X_prev + np.sqrt(1 - beta_value**2) * x_i

        x_d1T_list.append(x_i)
        X_prev = x_i

    # 3) Generate target train, test, val sets
    x_trgt_train = rng.normal(loc=0.0, scale=eps1[T], size=(ms[T], d)).astype(
        np.float64
    )
    x_trgt_test = rng.normal(
        loc=0.0, scale=eps1[T], size=(int(10 * m1T_sum), d)
    ).astype(np.float64)
    x_trgt_val = rng.normal(loc=0.0, scale=[T], size=(100, d)).astype(np.float64)

    if feature_means[T] != 0.0:
        x_trgt_train += feature_means[T]
        x_trgt_test += feature_means[T]
        x_trgt_val += feature_means[T]

    # 4) Construct base and target weight vectors
    w_trgt = rng.normal(loc=weight_loc, scale=1.0, size=(d, 1))
    w_trgt /= np.linalg.norm(w_trgt)

    w_base = rng.normal(loc=weight_loc, scale=1.0, size=(d, 1))
    w_base /= np.linalg.norm(w_base)

    ws = [w_trgt]
    for i in range(T):
        if random_w:
            # re-draw base with same loc
            w_base = rng.normal(loc=weight_loc, scale=1.0, size=(d, 1))
            w_base /= np.linalg.norm(w_base)
        w_i = w_trgt + eps2[i] * w_base
        if normalize_w_t:
            w_i /= np.linalg.norm(w_i)
        ws.insert(-1, w_i)

    # 5) Generate labels for D_1..D_T
    y_d1T_list: List[NDArray[np.float64]] = []
    for i in range(T):
        noise_i = rng.normal(loc=noise_loc, scale=sigma, size=(ms[i], 1))
        y_temp = x_d1T_list[i] @ ws[i] + noise_i
        y_temp = y_temp.astype(np.float64)
        y_d1T_list.append(y_temp)

    # 6) Generate labels for D_{T+1}: train, test, val
    noise_train = rng.normal(loc=noise_loc, scale=sigma, size=(ms[T], 1))
    y_trgt_train = (x_trgt_train @ ws[T] + noise_train).astype(np.float64)

    noise_test = rng.normal(loc=noise_loc, scale=sigma, size=(10 * m1T_sum, 1))
    y_trgt_test = (x_trgt_test @ ws[T] + noise_test).astype(np.float64)

    noise_val = rng.normal(loc=noise_loc, scale=sigma, size=(ms[T], 1))
    y_trgt_val = (x_trgt_val @ ws[T] + noise_val).astype(np.float64)

    # 7) Introduce heavy noise in D_1..D_T from alpha*ms[i] onward
    for i in range(T):
        start_idx = int(alpha * ms[i])
        for j in range(start_idx, ms[i]):
            y_d1T_list[i][j, 0] = np.linalg.norm(ws[i]) ** 2
            x_d1T_list[i][j, :] = -100.0 * ws[i].ravel()

    # 8) Flatten the source data
    x_d1T_train, y_d1T_train = np.vstack(x_d1T_list), np.vstack(y_d1T_list)

    return (
        x_d1T_train,
        y_d1T_train,
        x_trgt_train,
        y_trgt_train,
        x_trgt_test,
        y_trgt_test,
        x_trgt_val,
        y_trgt_val,
        x_d1T_list,
        y_d1T_list,
    )
