from itertools import product
import random
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from sltns.data._data import TimeSeriesSimulator, split_data
from sltns.discrepency._disc import DiscrepancyEstimator
from sltns.optimization._opt import MDRIFTTuner

warnings.simplefilter("ignore", ConvergenceWarning)

warnings.filterwarnings(
    "ignore",
    message="Non-stationary starting autoregressive parameters found",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Non-invertible starting MA parameters found",
    category=UserWarning,
)


class SingleRunResult(NamedTuple):
    seed: int
    series: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    tuner: MDRIFTTuner
    best_params: dict
    h_mdr: Any
    q_full: np.ndarray
    trace: list[float]
    metrics: Dict[str, float]
    verbose: bool


class ExperimentRunner:
    """Initialize the experiment runner.

    Args:
        param_grid (Dict[str, Any]):
            Hyperparameter grid for MDRIFTTuner.
        fixed_kwargs (Dict[str, Any]):
            Fixed arguments passed to MDRIFT on every run.
        arima_orders (Optional[List[Tuple[int,int,int]]]):
            List of (p,d,q) tuples to search for ARIMA tuning. If None,
            defaults to all combinations in range(3).
        verbose (bool):
            If True, print progress and ARIMA tuning info.
    """

    def __init__(
        self,
        param_grid: Dict[str, Any],
        fixed_kwargs: Dict[str, Any],
        arima_orders: Optional[List[Tuple[int, int, int]]] = None,
        verbose: bool = False,
    ):
        self.param_grid = param_grid
        self.fixed_kwargs = fixed_kwargs
        self.arima_orders = (
            arima_orders
            if arima_orders is not None
            else list(product(range(3), repeat=3))
        )
        self.verbose = verbose

    def generate_seeds(
        self, n_seeds: int, master_seed: Optional[int] = None
    ) -> List[int]:
        """Produce a list of distinct 32-bit integer seeds.

        Args:
            n_seeds (int):
                Number of seeds to generate.
            master_seed (Optional[int]):
                If provided, seeds the RNG for reproducibility.

        Returns:
            List[int]: A list of `n_seeds` unique integers in [0, 2**31-1].
        """
        rng = random.Random(master_seed)
        return rng.sample(range(2**31), k=n_seeds)

    def evaluate_arima(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Tuple[str, float]:
        """Select and evaluate an ARIMA model via validation and full refit.

        Args:
            y_train (np.ndarray):
                Training targets of length n_train.
            y_val (np.ndarray):
                Validation targets of length n_val.
            y_test (np.ndarray):
                Test targets of length n_test.

        Returns:
            Tuple[str, float]:
                A key of the form 'ARIMA(p,d,q)' (or 'ARIMA' if none fit)
                and the corresponding test RMSE (or np.nan on failure).
        """
        best_order: Optional[Tuple[int, int, int]] = None
        best_val_rmse = float("inf")

        for order in self.arima_orders:
            try:
                model = ARIMA(y_train, order=order).fit()
                y_pred_val = model.forecast(steps=len(y_val))
                rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
                if rmse_val < best_val_rmse:
                    best_val_rmse = rmse_val
                    best_order = order
            except Exception:
                continue

        if best_order is None:
            if self.verbose:
                print(
                    "  Warning: no ARIMA model could be fit on validation; skipping ARIMA."
                )
            return "ARIMA", float("nan")

        try:
            full_series = np.concatenate([y_train, y_val])
            arima_full = ARIMA(full_series, order=best_order).fit()
            y_pred_test = arima_full.forecast(steps=len(y_test))
            test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            key = f"ARIMA{best_order}"
            if self.verbose:
                print(
                    f"  Best ARIMA order: {best_order} "
                    f"(val RMSE={best_val_rmse:.4f}, test RMSE={test_rmse:.4f})"
                )
            return key, test_rmse
        except Exception as e:
            if self.verbose:
                print(
                    f"  Warning: full‐data ARIMA fit failed for order {best_order}: {e}"
                )
            return f"ARIMA{best_order}", float("nan")

    def evaluate_models(
        self,
        h_model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Compute test RMSE for MDRIFT and three baselines.

        Args:
            h_model (Any):
                Fitted MDRIFT model (trained on train+val).
            X_train (np.ndarray):
                Training features of shape (n_train, d).
            y_train (np.ndarray):
                Training targets of length n_train.
            X_val (np.ndarray):
                Validation features of shape (n_val, d).
            y_val (np.ndarray):
                Validation targets of length n_val.
            X_test (np.ndarray):
                Test features of shape (n_test, d).
            y_test (np.ndarray):
                Test targets of length n_test.

        Returns:
            Dict[str, float]:
                RMSEs for keys 'MDRIFT', 'OLS', 'Mean', and 'ARIMA'.
        """
        results: Dict[str, float] = {}
        y_pred = h_model.predict(X_test)
        results["MDRIFT"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        X_trval = np.vstack([X_train, X_val])
        y_trval = np.concatenate([y_train, y_val])
        ols = LinearRegression().fit(X_trval, y_trval)
        results["OLS"] = float(np.sqrt(mean_squared_error(y_test, ols.predict(X_test))))

        m = float(y_trval.mean())
        results["Mean"] = float(
            np.sqrt(mean_squared_error(y_test, np.full_like(y_test, m)))
        )

        arima_key, arima_rmse = self.evaluate_arima(
            y_train=y_train, y_val=y_val, y_test=y_test
        )
        results[arima_key] = arima_rmse

        return results

    def simulate_and_split(
        self,
        process_cfg: dict,
        seed: int,
        series_length: int,
        w_val: int,
        w_test: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Simulate a time series and split into train/val/test sets.

        Args:
            process_cfg (dict):
                Single-process configuration for TimeSeriesSimulator.
            seed (int):
                Random seed for simulation.
            series_length (int):
                Total length of the simulated series.
            w_val (int):
                Number of points held out for validation.
            w_test (int):
                Number of points held out for final test.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - series: the full simulated series of length `series_length`
                - X_train, y_train: training split
                - X_val,   y_val:   validation split
                - X_test,  y_test:  test split
        """
        sim = TimeSeriesSimulator(configs=[process_cfg], n=series_length, seed=seed)
        series = sim.generate().ravel()
        X = np.arange(series_length).reshape(-1, 1)
        y = series

        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X, y, w_val=w_val, w_test=w_test
        )
        return series, X_train, y_train, X_val, y_val, X_test, y_test

    def run_single(
        self,
        process_cfg: dict,
        seed: int,
        w_val: int,
        w_test: int,
        n_iter: int,
        series_length: int = 500,
        trs_plus_only: bool = False,
        full_refit: bool = True,
        slice_prior: bool = False,
    ) -> SingleRunResult:
        """Simulate, tune, refit, and evaluate a single random‐seed run.

        Args:
            process_cfg (dict):
                Single-process configuration for TimeSeriesSimulator.
            seed (int):
                Random seed for simulation, tuning, and RNG.
            w_val (int):
                Number of points held out for validation.
            w_test (int):
                Number of points held out for final test.
            n_iter (int):
                Number of iterations for MDRIFTTuner.random_search.
            series_length (int):
                Total length of the simulated series.
            trs_plus_only (bool, optional):
                If True, use discrepancies without absolute values (trs+ = False).
                Defaults to False.
            full_refit (bool, optional):
                If True, refit the final MDRIFT model on the combined train+val set;
                if False, use early stopping on validation. Defaults to True.
        slice_prior (bool, optional): Slice prior from validation set if True.
            Defaults to False.

        Returns:
            SingleRunResult:
                NamedTuple containing:
                - seed: the random seed used
                - series: the full simulated series
                - X_train, y_train, X_val, y_val, X_test, y_test: data splits
                - tuner: the MDRIFTTuner instance
                - best_params: dict of selected hyperparameters
                - h_mdr: the final fitted MDRIFT model
                - q_full: learned sampling distribution
                - trace: list of Φ values over iterations
                - metrics: dict of test RMSEs for all models
                - verbose: whether verbose logging was enabled
        """
        series, X_tr, y_tr, X_va, y_va, X_te, y_te = self.simulate_and_split(
            process_cfg, seed, series_length, w_val, w_test
        )

        tuner = MDRIFTTuner(
            param_grid=self.param_grid,
            fixed_kwargs=self.fixed_kwargs,
            full_n=len(X_tr) + len(X_va),
            random_state=seed,
            trs_plus_only=trs_plus_only,
            full_refit=full_refit,
            slice_prior=slice_prior,
        )
        _, best_params, _ = tuner.random_search(
            X=X_tr,
            y=y_tr,
            x_val=X_va,
            y_val=y_va,
            n_iter=n_iter,
        )

        h_mdr, q_full, trace = tuner.refit_on_full(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            best_params=best_params,
        )

        metrics = self.evaluate_models(
            h_model=h_mdr,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            X_test=X_te,
            y_test=y_te,
        )

        return SingleRunResult(
            seed=seed,
            series=series,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            X_test=X_te,
            y_test=y_te,
            tuner=tuner,
            best_params=best_params,
            h_mdr=h_mdr,
            q_full=q_full,
            trace=trace,
            metrics=metrics,
            verbose=self.verbose,
        )

    def run_experiments(
        self,
        process_cfg: dict,
        seeds: List[int],
        w_val: int,
        w_test: int,
        n_iter: int,
        series_length: int = 500,
        trs_plus_only: bool = False,
        full_refit: bool = True,
        slice_prior: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run multiple seeds and aggregate per‐model test RMSEs.

        Args:
            process_cfg (dict):
                Single-process configuration for simulation.
            seeds (List[int]):
                List of random seeds to iterate over.
            w_val (int):
                Validation set size.
            w_test (int):
                Test set size.
            n_iter (int):
                MDRIFTTuner.random_search iteration count.
            series_length (int):
                Total length of each simulated series.
            trs_plus_only (bool, optional):
                If True, use discrepancies without absolute values (trs+ = False).
                Defaults to False.
            full_refit (bool, optional):
                If True, refit the final MDRIFT model on the combined train+val set;
                if False, use early stopping on validation. Defaults to True.
            slice_prior (bool, optional): Slice prior from validation set if True.
                Defaults to False.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - DataFrame with columns ['seed','model','test_rmse'].
                - Summary DataFrame with mean±std RMSE per model.
        """
        records = []
        for seed in seeds:
            res = self.run_single(
                process_cfg=process_cfg,
                seed=seed,
                w_val=w_val,
                w_test=w_test,
                n_iter=n_iter,
                series_length=series_length,
                trs_plus_only=trs_plus_only,
                full_refit=full_refit,
                slice_prior=slice_prior,
            )
            # flatten the per‐model metrics
            for model, rmse in res.metrics.items():
                records.append({"seed": seed, "model": model, "test_rmse": rmse})

        df = pd.DataFrame.from_records(records)
        df["model"] = df["model"].apply(
            lambda m: "ARIMA" if m.startswith("ARIMA") else m
        )

        summary = (
            df.groupby("model")["test_rmse"]
            .agg(["mean", "std"])
            .rename(columns={"mean": "rmse_mean", "std": "rmse_std"})
        )
        return df, summary

    def compare_abs_vs_regular(
        self,
        process_cfg: dict,
        seed: int,
        w_val: int,
        w_test: int,
        n_iter: int,
        series_length: int = 500,
        full_refit: bool = True,
        slice_prior: bool = False,
    ) -> Tuple[
        np.ndarray,  # series
        np.ndarray,  # d_abs
        np.ndarray,  # d
        np.ndarray,  # q_abs
        np.ndarray,  # q
        np.ndarray,  # p0_abs
        np.ndarray,  # p0
        Dict[str, float],  # metrics_abs
        Dict[str, float],  # metrics
    ]:
        """Run two MDRIFT fits (discrepancies with and without abs).

        Args:
            process_cfg (dict):
                Single-process configuration for TimeSeriesSimulator.
            seed (int):
                Random seed for simulation, tuning, and RNG.
            w_val (int):
                Number of points held out for validation.
            w_test (int):
                Number of points held out for final test.
            n_iter (int):
                Number of iterations for MDRIFTTuner.random_search.
            series_length (int, optional):
                Total length of the simulated series. Defaults to 500.
            full_refit (bool, optional):
                If True, both runs are refit on train+val (no early stopping);
                if False, both use validation-based early stopping. Defaults to True.
            slice_prior (bool, optional): Slice prior from validation set if True.
                Defaults to False.

        Returns:
            series (ndarray):
                The full simulated series (length = series_length).
            d_abs (ndarray):
                Discrepancy vector from the abs-only run
                (length = n_source).
            d (ndarray):
                Discrepancy vector from the TRS run
                (length = n_source).
            q_abs (ndarray):
                Learned q-vector from abs-only run (length = n_source).
            q (ndarray):
                Learned q-vector from TRS run (length = n_source).
            p0_abs (ndarray):
                Prior used in the abs run (length = n_source).
            p0 (ndarray):
                Prior used in the TRS run (length = n_source).
            metrics_abs (dict):
                Evaluation metrics from the abs-only run.
            metrics (dict):
                Evaluation metrics from the TRS run.
        """
        single_abs = self.run_single(
            process_cfg=process_cfg,
            seed=seed,
            w_val=w_val,
            w_test=w_test,
            n_iter=n_iter,
            series_length=series_length,
            trs_plus_only=False,
            full_refit=full_refit,
            slice_prior=slice_prior,
        )
        single = self.run_single(
            process_cfg=process_cfg,
            seed=seed,
            w_val=w_val,
            w_test=w_test,
            n_iter=n_iter,
            series_length=series_length,
            trs_plus_only=True,
            full_refit=full_refit,
            slice_prior=slice_prior,
        )

        if full_refit:
            X_source = np.vstack([single_abs.X_train, single_abs.X_val])
            y_source = np.concatenate([single_abs.y_train, single_abs.y_val])
        else:
            X_source = single_abs.X_train
            y_source = single_abs.y_train

        n_source = X_source.shape[0]
        prior_len_abs = int(single_abs.best_params["prior_len"])
        prior_len = int(single.best_params["prior_len"])

        p0_abs = single_abs.tuner._build_prior(n_source, prior_len_abs)
        d_abs = DiscrepancyEstimator(
            X=X_source, y=y_source, p=p0_abs, trs_plus_only=True
        ).all_di_trs()[:n_source]
        q_abs = single_abs.q_full[:n_source]

        p0 = single.tuner._build_prior(n_source, prior_len)
        d = DiscrepancyEstimator(
            X=X_source, y=y_source, p=p0, trs_plus_only=False
        ).all_di_trs()[:n_source]
        q = single.q_full[:n_source]

        return (
            single_abs.series,
            d_abs,
            d,
            q_abs,
            q,
            p0_abs,
            p0,
            single_abs.metrics,
            single.metrics,
        )

    def run_experiment_grid(
        self,
        process_cfgs: list[dict],
        seeds: list[int],
        w_val: int,
        w_test: int,
        n_iter: int,
        series_length: int = 500,
        full_refit: bool = False,
        slice_prior: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run MDRIFT experiments for multiple processes and both trs_plus_only settings.

        For each cfg in `process_cfgs` and for trs_plus_only in [False, True],
        calls `run_experiments`, tags the results, and returns:

        - df_all:   per‐seed/per‐model RMSEs
        - summary_all: per‐model mean±std RMSEs with kind & trs flag
        - summary_simple: same as summary_all but with a single `model` label
            ("MDRIFT abs" or "MDRIFT") and no `trs_plus_only` column.

        Args:
            process_cfgs (list of dict): e.g. [{"kind":"random_walk","sigma":1}, ...]
            seeds (list of int):           RNG seeds (e.g. runner.generate_seeds(50))
            w_val, w_test, n_iter, series_length, full_refit: passed to run_experiments
            slice_prior (bool, optional): Slice prior from validation set if True.
                Defaults to False.

        Returns:
            df_all (pd.DataFrame):
                Columns = ["seed","model","test_rmse","kind","trs_plus_only"]
            summary_all (pd.DataFrame):
                Columns = ["model","rmse_mean","rmse_std","kind","trs_plus_only"]
            summary_simple (pd.DataFrame):
                Columns = ["model","kind","rmse_mean","rmse_std"]
                where model is one of
                - "MDRIFT abs"  (trs_plus_only=False)
                - "MDRIFT"      (trs_plus_only=True)
                - "OLS", "Mean", "ARIMA"
        """
        dfs = []
        summaries = []
        for cfg in process_cfgs:
            for trs in (False, True):
                df, summary = self.run_experiments(
                    process_cfg=cfg,
                    seeds=seeds,
                    w_val=w_val,
                    w_test=w_test,
                    n_iter=n_iter,
                    series_length=series_length,
                    trs_plus_only=trs,
                    full_refit=full_refit,
                    slice_prior=slice_prior,
                )
                df = df.copy()
                df["kind"] = cfg["kind"]
                df["trs_plus_only"] = trs
                dfs.append(df)

                sum_df = summary.reset_index().rename(columns={"index": "model"})
                sum_df["kind"] = cfg["kind"]
                sum_df["trs_plus_only"] = trs
                summaries.append(sum_df)

        df_all = pd.concat(dfs, ignore_index=True)
        summary_all = pd.concat(summaries, ignore_index=True)

        summary_simple = summary_all.copy()

        def _label(row):
            if row["model"] == "MDRIFT":
                return "MDRIFT abs" if not row["trs_plus_only"] else "MDRIFT"
            return row["model"]

        summary_simple["model"] = summary_simple.apply(_label, axis=1)

        mdr = summary_simple[summary_simple["model"].str.startswith("MDRIFT")]
        baselines = summary_simple[~summary_simple["model"].str.startswith("MDRIFT")]
        baselines = baselines.drop_duplicates(subset=["kind", "model"], keep="first")
        summary_simple = pd.concat([baselines, mdr], ignore_index=True)

        order = ["ARIMA", "MDRIFT abs", "MDRIFT", "Mean", "OLS"]
        summary_simple["model"] = pd.Categorical(
            summary_simple["model"], categories=order, ordered=True
        )
        summary_simple = summary_simple.sort_values(["kind", "model"]).reset_index(
            drop=True
        )

        summary_simple = summary_simple[["kind", "model", "rmse_mean", "rmse_std"]]

        return df_all, summary_all, summary_simple
