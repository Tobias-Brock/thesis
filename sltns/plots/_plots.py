from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

mapping = {
    "white_noise": "White noise",
    "random_walk": "Random walk",
    "ar1": "AR(1)",
    "unit_root": "Unit-Root AR(1)",
    "periodic": "Periodic",
    "heteroskedastic": "Heteroskedastic",
    "arch": "ARCH(1)",
    "garch": "GARCH(1,1)",
    "tv_ar1": "TV-AR(1)",
}


class TimeSeriesPlotter:
    """Plotter class extending TimeSeriesSimulator.

    Inherits simulation of various processes and adds
    customizable plotting and save functionality.
    """

    def __init__(
        self,
        series_list: list[np.ndarray],
        configs: list[dict],
        gridsize: Tuple[int, int] = (1, 2),
        figsize: Optional[Tuple[float, float]] = None,
        dpi: int = 300,
        style: str = "science",
        grid: bool = False,
        sharey: bool = False,
        use_tex: bool = False,
    ):
        """Plotter init.

        Args:
            series_list: list of 1D arrays, each the same length n.
            configs:     one dict per series, for titles & styling.
            gridsize (tuple): Size of grid for plotting.
            figsize:     figure size.
            dpi:         resolution.
            style:       matplotlib style.
            grid:        show grid on each subplot.
            sharey:      share y-axis across all subplots.
            use_tex:     render text with LaTeX.
        """
        assert len(series_list) == len(configs), "Need one config per series"
        self.series_list = series_list
        self.configs = configs
        self.n = series_list[0].shape[0]
        self.gridsize = gridsize
        self.dpi = dpi
        self.grid = grid
        self.sharey = sharey
        self.use_tex = use_tex
        self.style = style

        if figsize is None:
            nrows, ncols = gridsize
            self.figsize = (5.0 * ncols, 3.0 * nrows)
        else:
            self.figsize = figsize

        self._apply_style()

    def _apply_style(self) -> None:
        """Apply plotting style and LaTeX/math settings."""
        if self.style.lower() == "science":
            try:
                plt.style.use(["science", "no-latex"])
            except Exception:
                plt.style.use("classic")
        else:
            plt.style.use(self.style)

        if self.use_tex:
            plt.rcParams.update(
                {
                    "mathtext.fontset": "cm",
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                }
            )

        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "axes.edgecolor": "black",
                "axes.labelcolor": "black",
                "axes.titlecolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": "lightgray",
                "axes.grid": self.grid,
                "grid.linestyle": "--",
                "grid.alpha": 0.5,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
            }
        )

    def _draw(self):
        """Exactly your old `_draw`, but pulling from self.series_list."""
        nrows, ncols = self.gridsize
        total = nrows * ncols

        all_series = []
        for series, cfg in zip(self.series_list, self.configs, strict=False):
            y = series.copy()
            if cfg.get("normalize", False):
                y = (y - y.mean()) / y.std()
            all_series.append(y)

        if self.sharey and all_series:
            y_min = min(y.min() for y in all_series)
            y_max = max(y.max() for y in all_series)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=self.figsize,
            dpi=self.dpi,
            sharey=self.sharey,
            squeeze=False,
        )

        kinds = [cfg.get("kind") for cfg in self.configs]
        all_periodic_1xN = nrows == 1 and all(k == "periodic" for k in kinds)

        if all_periodic_1xN:
            fig.subplots_adjust(top=0.82)
            fig.text(0.5, 0.95, "Periodic", ha="center", va="bottom", fontsize=16)

        for idx, (cfg, y) in enumerate(zip(self.configs, all_series, strict=False)):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            kind = cfg.get("kind", "white_noise")
            sigma = cfg.get("sigma")
            omega = cfg.get("omega")
            alpha = cfg.get("alpha")
            beta = cfg.get("beta")
            phi = cfg.get("phi")

            # line color
            color = (
                "tab:red"
                if kind in ("random_walk", "unit_root", "heteroskedastic")
                else "tab:blue"
            )
            ax.plot(y, color=color, linewidth=1.5)

            ax.set_xlabel(r"$i$")
            if col == 0:
                ax.set_ylabel(r"$X_i$")

            # SPECIAL: pure-periodic 1×N → only draw ω under the shared header
            if all_periodic_1xN:
                ax.set_title("")  # clear any default
                ax.text(
                    0.5,
                    1.02,
                    rf"$\omega={omega}$",
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.set_xlim(0, self.n - 1)
                continue

            if row == 0:
                # build top_line / bot_line for *every* kind
                if kind == "white_noise":
                    top_line = "White noise"
                    bot_line = rf"$\sigma={sigma}$"

                elif kind == "heteroskedastic":
                    top_line = "Heteroskedastic noise"
                    expr = r"\sqrt{i}" if sigma == 1 else rf"{sigma}\sqrt{{i}}"
                    bot_line = rf"$\sigma_i={expr}$"

                elif kind == "random_walk":
                    top_line = "Random walk"
                    bot_line = rf"$\sigma={sigma}$"

                elif kind == "ar1":
                    top_line = rf"AR(1) ($\phi={phi}$)"
                    bot_line = rf"$\sigma={sigma}$"

                elif kind == "unit_root":
                    top_line = "Unit-Root AR(1)"
                    bot_line = rf"$\sigma={sigma}$"

                elif kind == "periodic":
                    top_line = "Periodic"
                    bot_line = rf"$\omega={omega}$"

                elif kind == "arch":
                    top_line = "ARCH(1)"
                    bot_line = rf"$\omega={omega},\,\alpha={alpha}$"

                elif kind == "garch":
                    top_line = "GARCH(1,1)"
                    bot_line = rf"$\omega={omega},\,\alpha={alpha},\,\beta={beta}$"

                elif kind == "tv_ar1":
                    top_line = "TV-AR(1)"
                    bot_line = r"$\phi_t,\ \mu_t\text{ varies}$"

                else:
                    top_line = mapping.get(kind, kind.replace("_", " ").title())
                    bot_line = rf"$\sigma={sigma}$"

                # draw the two fixed‐height lines
                ax.text(
                    0.5,
                    1.13,
                    top_line,
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=14,
                )
                ax.text(
                    0.5,
                    1.02,
                    bot_line,
                    ha="center",
                    va="bottom",
                    transform=ax.transAxes,
                    fontsize=14,
                )

            else:
                # lower‐row: just show the parameter line as before
                if kind == "periodic":
                    subtitle = rf"$\omega={omega}$"
                elif kind == "heteroskedastic":
                    expr = r"\sqrt{i}" if sigma == 1 else rf"{sigma}\sqrt{{i}}"
                    subtitle = rf"$\sigma_i={expr}$"
                elif kind == "arch":
                    subtitle = rf"$\alpha={alpha}$"
                elif kind == "garch":
                    subtitle = rf"$\alpha={alpha},\,\beta={beta}$"
                elif kind == "tv_ar1":
                    subtitle = r"$\phi_t,\mu_t\text{ varies}$"
                else:
                    subtitle = rf"$\sigma={sigma}$"

                ax.set_title(subtitle)

            ax.set_xlim(0, self.n - 1)

        for idx in range(len(self.series_list), total):
            row, col = divmod(idx, ncols)
            axes[row][col].axis("off")

        if self.sharey and all_series:
            for row_axes in axes:
                for a in row_axes:
                    a.set_ylim(y_min, y_max)

        plt.tight_layout()
        return fig, axes

    def plot(self) -> None:
        """Display the generated time series plots in an interactive window."""
        fig, _ = self._draw()
        plt.show()

    def plot_with_discrepancy(
        self,
        series: np.ndarray,
        d: np.ndarray,
        q: np.ndarray,
        prior: np.ndarray,
        window: int = 20,
        plot_raw: bool = True,
        plot_weighted: bool = True,
    ) -> None:
        """Plot the raw series alongside raw and weighted discrepancies.

        This produces a 1×2 panel:
        - Left:  the series with the hold-out window shaded.
        - Right: the discrepancy curve(s), with prior range highlighted.

        Args:
            series (ndarray):
                The full time series to plot (length ≥ n_source + window).
            d (ndarray):
                Discrepancy vector of length n_source.
            q (ndarray):
                Learned sampling weights of length n_source.
            prior (ndarray):
                Prior distribution vector of length n_source.
            window (int, optional):
                Number of points beyond the source range to shade in the series plot.
                Defaults to 20.
            plot_raw (bool, optional):
                If True, draw the raw discrepancy curve. Defaults to True.
            plot_weighted (bool, optional):
                If True, draw the weighted discrepancy curve. Defaults to True.

        Returns:
            None
        """
        kind = self.configs[0].get("kind", "series")
        label = mapping.get(kind, kind.replace("_", " ").title())

        n_dq = min(len(d), len(q), len(prior))
        d = d[:n_dq]
        q = q[:n_dq]
        prior = prior[:n_dq]

        n_source = len(d)
        L = min(len(series), n_source + window)
        series_plot = series[:L]
        mask = prior > 0
        if mask.any():
            p0_start = mask.argmax()
            p0_end = n_source - mask[::-1].argmax()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=self.dpi)

        axes[0].plot(np.arange(L), series_plot, linewidth=1.5)
        axes[0].axvspan(n_source, n_source + window, color="gray", alpha=0.2)
        axes[0].axvline(n_source, color="black", linestyle="--", alpha=0.7)
        axes[0].set_xlabel(r"$i$")
        axes[0].set_ylabel(r"$X_i$")
        axes[0].set_title(
            self.configs[0].get("kind", "Series").replace("_", " ").title()
        )
        if self.grid:
            axes[0].grid(True)

        # — Discrepancy Plot —
        idx = np.arange(n_source)
        if plot_raw:
            axes[1].plot(
                idx, d, label=r"$d_i$", linestyle="-", marker="o", markersize=4
            )
        if plot_weighted:
            axes[1].plot(
                idx, q * d, label=r"$q_i d_i$", linestyle="--", marker="s", markersize=4
            )

        if mask.any():
            axes[1].axvspan(p0_start, p0_end, color="gray", alpha=0.2)
            axes[1].set_title(f"Discrepancy (prior on {p0_start}:{p0_end-1})")
        else:
            axes[1].axvspan(n_source - window, n_source, color="gray", alpha=0.2)
            axes[1].set_title(f"Discrepancy (last {window})")

        axes[1].set_xlabel(r"$i$")
        axes[1].set_ylabel("Discrepancy")
        axes[1].legend()
        if self.grid:
            axes[1].grid(True)

        axes[0].set_xlim(0, len(series) - 1)
        axes[1].set_xlim(0, n_source - 1)
        plt.tight_layout()
        plt.show()

    def plot_compare_abs_vs_regular(
        self,
        d_abs: np.ndarray,
        d: np.ndarray,
        q_abs: np.ndarray,
        q: np.ndarray,
        p0_abs: np.ndarray,
        p0: np.ndarray,
        window: int = 20,
        figsize_triplet: tuple | None = None,
        figsize_2x2: tuple | None = None,
        return_figs: bool = False,
    ) -> Optional[Tuple[plt.Figure, plt.Figure]]:
        """Compare absolute‐only vs. regular discrepancies and weights.

        Produces two figures:
        1. Raw series with discrepancy curves for `d` and `d_abs`.
        2. Weighted discrepancy and weight vs prior plots for both runs.

        Args:
            d_abs (ndarray):
                Absolute discrepancy vector (length = n_source).
            d (ndarray):
                Regular discrepancy vector (length = n_source).
            q_abs (ndarray):
                Learned weights for the absolute run.
            q (ndarray):
                Learned weights for the regular run.
            p0_abs (ndarray):
                Prior distribution used in the absolute run.
            p0 (ndarray):
                Prior distribution used in the regular run.
            window (int, optional):
                Number of points beyond the source range to show in the series plot.
            figsize_triplet (tuple, optional):
                Figure size for the 1×3 panel; defaults to (12, 3) if None.
            figsize_2x2 (tuple, optional):
                Figure size for the 2×2 panel; defaults to (12, 6) if None.
            return_figs (bool, optional):
                If True, return the two Figure objects instead of displaying them.

        Returns:
            Tuple[plt.Figure, plt.Figure] or None:
                The 1×3 and 2×2 figures, if `return_figs` is True; otherwise None.
        """
        n_source = len(d_abs)
        L = min(self.n, n_source + window)  # total points to show
        series = self.series_list[0][:L]  # assume first series drives titles

        if figsize_triplet is None:
            figsize_triplet = (12, 3)
        if figsize_2x2 is None:
            figsize_2x2 = (12, 6)

        fig1, axes = plt.subplots(
            1, 3, figsize=figsize_triplet, dpi=self.dpi, sharey=False
        )

        cfg = self.configs[0]
        kind_key = cfg.get("kind", "series")
        kind_name = mapping.get(kind_key, kind_key.replace("_", " ").title())

        sigma = cfg.get("sigma", None)
        if sigma is not None:
            title = f"{kind_name} ($\\sigma$={sigma})"
        else:
            title = kind_name

        axes[0].set_title(title)

        idx_s = np.arange(L)
        axes[0].plot(idx_s, series, lw=1.5, color="grey")
        axes[0].set_xlabel(r"$i$")
        axes[0].set_ylabel(r"$X_i$")
        axes[0].set_title(title)
        if self.grid:
            axes[0].grid(True)
        axes[0].set_xlim(0, L - 1)

        idx_d = np.arange(n_source)
        axes[1].plot(idx_d, d, "--s", ms=2)  # next color in cycle
        axes[1].set_xlabel(r"$i$")
        axes[1].set_ylabel(r"$d_i'$")
        axes[1].set_title(r"$d_i'$")
        if self.grid:
            axes[1].grid(True)

        axes[2].plot(idx_d, d_abs, "-o", ms=2, color="red")  # default color: blue
        axes[2].set_xlabel(r"$i$")
        axes[2].set_ylabel(r"$d_i$")
        axes[2].set_title(r"$d_i$")
        if self.grid:
            axes[2].grid(True)

        for ax in axes[1:]:
            ax.set_xlim(0, n_source - 1)

        plt.tight_layout()
        plt.show()
        mask_abs = p0_abs > 0
        s_abs = int(mask_abs.sum())

        mask_full = p0 > 0
        s_full = int(mask_full.sum())

        fig2, axes = plt.subplots(
            2, 2, figsize=figsize_2x2, dpi=self.dpi, sharex="col", sharey=False
        )

        axes[0, 0].plot(idx_d, q * d, "--s", ms=4)
        axes[0, 0].set_title(r"$q_i d_i'$")

        axes[0, 1].plot(idx_d, q_abs * d_abs, "-o", ms=4, color="red")
        axes[0, 1].set_title(r"$q_i d_i$")

        axes[1, 0].plot(idx_d, q, "-o", ms=4, label=r"$q_i$")
        axes[1, 0].plot(idx_d, p0, "--", lw=2, label=r"$p_0$", color="black")
        axes[1, 0].set_title(f"Weights vs prior (s={s_full})")
        axes[1, 0].legend()

        axes[1, 1].plot(idx_d, q_abs, "-o", ms=4, label=r"$q_i$", color="red")
        axes[1, 1].plot(idx_d, p0_abs, "--", lw=2, label=r"$p_0$", color="black")
        axes[1, 1].set_title(f"Weights vs prior (s={s_abs})")
        axes[1, 1].legend()

        for ax in axes.flatten():
            ax.set_xlabel(r"$i$")
            if self.grid:
                ax.grid(True)
        for ax in axes[0, :]:
            ax.set_xlim(0, n_source - 1)
        for ax in axes[1, :]:
            ax.set_xlim(0, n_source - 1)

        plt.tight_layout()
        plt.show()
        if return_figs:
            return fig1, fig2
        return None

    def plot_weights(self, q: np.ndarray, prior: np.ndarray) -> None:
        """Plot final learned MDRIFT source-point weights against the prior.

        Args:
            q (n_source,): Learned weights on source points.
            prior (n_source,): Prior distribution over source points.
        """
        n_source = len(q)
        idx = np.arange(n_source)
        p_s = prior[:n_source]

        fig, ax = plt.subplots(figsize=(12, 4), dpi=self.dpi)

        ax.plot(idx, q, "-o", color="C0", label=r"$q_i$", markersize=5)
        ax.plot(idx, p_s, "--", color="C1", label=r"$p_0$ (prior)", linewidth=2)

        ax.set_xlabel("Source index $i$")
        ax.set_ylabel("Weight value")
        ax.set_title("Final MDRIFT Weights vs. Prior")
        ax.set_xlim(0, n_source - 1)

        fig.subplots_adjust(bottom=0.10)
        fig.legend(
            loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1)
        )

        plt.show()

    def save(
        self,
        filename: str,
        fig: Optional[plt.Figure] = None,
        dpi: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        format: str = "png",
    ) -> None:
        """Save a figure to disk, using the plotter’s styling defaults.

        If `fig` is not provided, the default drawing method (`_draw`) is used.

        Args:
            filename (str):
                Path (including extension) where the figure will be saved.
            fig (plt.Figure, optional):
                The Matplotlib Figure to save. If None, `self._draw()` is called
                to produce a new figure.
            dpi (int, optional):
                Resolution in dots per inch. If provided, overrides the plotter’s
                `self.dpi` setting.
            figsize (tuple[float, float], optional):
                Figure size in inches (width, height). If provided, overrides
                the plotter’s `self.figsize`.
            format (str, optional):
                File format, e.g. "png", "pdf", "svg". Defaults to "png".

        Returns:
            None
        """
        if dpi is not None:
            self.dpi = dpi
        if figsize is not None:
            self.figsize = figsize

        if fig is None:
            fig, _ = self._draw()
        fig.savefig(filename, dpi=self.dpi, format=format, bbox_inches="tight")
        plt.close(fig)
