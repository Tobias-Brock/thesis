from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401


class TimeSeriesPlotter:
    """Plotter class.

    Generate and plot various time series (white noise, random walk,
    AR(1), unit-root AR(1), periodic, heteroskedastic) in a customizable
    grid layout, with shared axes, color coding, LaTeX-like math via SciencePlots,
    and save functionality.

    Methods:
        plot(): Display the generated plots interactively.
        save(filename, dpi=None, figsize=None, format='png'): Save plots to file.
    """

    def __init__(
        self,
        configs: list,
        n: int = 200,
        seed: int = 42,
        gridsize: tuple = (1, 2),
        figsize: tuple = None,
        dpi: int = 300,
        style: str = "science",
        grid: bool = False,
        sharey: bool = False,
        use_tex: bool = False,
    ):
        """Initialize a TimeSeriesPlotter instance.

        Args:
            configs (List[dict]): List of series configurations. Each dict may include:
                - kind (str): "white_noise", "random_walk", "ar1",
                  "unit_root", "periodic", or "heteroskedastic".
                - sigma (float): Standard deviation of innovations (default 1.0).
                - phi (float): AR(1) coefficient for "ar1" (default 0.7).
                - normalize (bool): Normalize series to zero mean and unit variance.
                - omega (float): Angular frequency for "periodic" (default 1.0).
                - phase (bool): If True, random initial phase for "periodic" (default True).
                - title (str): Custom title for the first-row plots (can include LaTeX math).
            n (int): Length of each generated series (default 200).
            seed (int): Random seed for reproducibility (default 42).
            gridsize (tuple): Number of rows and columns (nrows, ncols) for subplots.
            figsize (tuple): Figure size in inches (width, height).
                If None, 5×ncols by 3×nrows.
            dpi (int): Resolution of the figure in dots per inch (default 300).
            style (str): Matplotlib style name or "science" for SciencePlots
                (fallback to "classic").
            grid (bool): Whether to draw gridlines on subplots.
            sharey (bool): If True, all subplots share the same y-axis limits.
            use_tex (bool): If True, enable real LaTeX rendering (requires TeX install).
        """
        self.configs = configs
        self.n = n
        self.seed = seed
        self.gridsize = gridsize
        self.dpi = dpi
        self.grid = grid
        self.sharey = sharey
        self.use_tex = use_tex
        if figsize is None:
            self.figsize = (5 * gridsize[1], 3 * gridsize[0])
        else:
            self.figsize = figsize
        self._init_random()
        self._apply_style(style)

    def _init_random(self) -> None:
        """Seed the NumPy random number generator."""
        np.random.seed(self.seed)

    def _apply_style(self, style: str) -> None:
        """Apply plotting style, math support, and override colors.

        Args:
            style (str): Matplotlib style or "science" for SciencePlots.
        """
        if style.lower() == "science":
            try:
                plt.style.use(["science", "no-latex"])
            except Exception:
                plt.style.use("classic")
        else:
            plt.style.use(style)
        if self.use_tex:
            plt.rcParams.update({
                "mathtext.fontset": "cm",
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
            })
        plt.rcParams.update({
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
            "mathtext.fontset": "cm",
            "axes.labelsize": 12,
            "axes.titlesize": 14,
        })

    def _generate(self, kind: str, sigma: float = 1.0, phi: float = 0.7) -> np.ndarray:
        """Generate a single time series (excluding periodic, heteroskedastic).

        Raises:
            ValueError: If `kind` not recognized or should be handled in _draw.
        """
        eps = np.random.normal(0, sigma, self.n)
        if kind == "white_noise":
            return eps
        elif kind == "random_walk":
            return np.cumsum(eps)
        elif kind == "ar1":
            y = np.zeros(self.n)
            y[0] = eps[0]
            for t in range(1, self.n):
                y[t] = phi * y[t - 1] + eps[t]
            return y
        elif kind == "unit_root":
            y = np.zeros(self.n)
            y[0] = eps[0]
            for t in range(1, self.n):
                y[t] = y[t - 1] + eps[t]
            return y
        else:
            raise ValueError(f"_generate cannot handle kind={kind!r}; use _draw for others.")

    def _draw(self):
        """Internal helper to generate and plot all subplots.

        Returns:
            tuple: (fig, axes) of created Figure and Axes array.
        """
        mapping = {
            "white_noise": "White Noise",
            "random_walk": "Random Walk",
            "ar1": "AR(1)",
            "unit_root": "Unit-Root AR(1)",
            "periodic": "Periodic",
            "heteroskedastic": "Heteroskedastic",
        }
        nrows, ncols = self.gridsize
        total = nrows * ncols
        all_series = []
        for cfg in self.configs:
            kind = cfg.get("kind", "white_noise")
            if kind == "periodic":
                omega = cfg.get("omega", 1.0)
                phase_flag = cfg.get("phase", True)
                phi0 = np.random.uniform(0, 2*np.pi) if phase_flag else 0.0
                tvals = np.arange(self.n)
                y = np.cos(omega * tvals + phi0)
            elif kind == "heteroskedastic":
                sigma = cfg.get("sigma", 1.0)
                tvals = np.arange(1, self.n + 1)
                eps = np.random.normal(0, 1, self.n)
                y = sigma * np.sqrt(tvals) * eps
            else:
                y = self._generate(
                    kind=kind,
                    sigma=cfg.get("sigma", 1.0),
                    phi=cfg.get("phi", 0.7)
                )
            if cfg.get("normalize", False):  # apply normalize to all kinds
                y = (y - y.mean()) / y.std()
            all_series.append(y)
        if self.sharey and all_series:
            y_min = min(y.min() for y in all_series)
            y_max = max(y.max() for y in all_series)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=self.figsize,
            dpi=self.dpi,
            sharey=self.sharey,
            squeeze=False,
        )
        for idx, (cfg, y) in enumerate(zip(self.configs, all_series)):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            kind = cfg.get("kind", "white_noise")
            color = "tab:red" if kind in ("random_walk", "unit_root", "heteroskedastic") else "tab:blue"
            ax.plot(y, color=color, linewidth=1.5)
            ax.set_xlabel(r"$i$")
            if col == 0:
                ax.set_ylabel(r"$y$")
            sigma = cfg.get("sigma")
            omega = cfg.get("omega")
            if row == 0:
                if "title" in cfg:
                    title = str(cfg["title"]).strip('"').strip("'")
                else:
                    desc = mapping.get(kind, kind.replace("_", " ").title())
                    if kind == "periodic":
                        title = rf"{desc} ($\omega={omega}$)"
                    elif kind == "heteroskedastic":
                        title = rf"{desc} ($\mathrm{{Var}}(X_i)=i\,\sigma^2$)"
                    else:
                        title = rf"{desc} ($\sigma={sigma}$)"
                ax.set_title(title)
            else:
                if kind == "periodic":
                    ax.set_title(rf"$\omega={omega}$")
                elif kind == "heteroskedastic":
                    ax.set_title(r"$\sigma_i=\sigma\sqrt{i}$")
                else:
                    ax.set_title(rf"$\sigma={sigma}$")
            ax.set_xlim(0, self.n - 1)
        for idx in range(len(self.configs), total):
            row, col = divmod(idx, ncols)
            axes[row][col].axis("off")
        if self.sharey and all_series:
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_ylim(y_min, y_max)
        plt.tight_layout()
        return fig, axes

    def plot(self) -> None:
        """Display the generated time series plots in an interactive window."""
        fig, _ = self._draw()
        plt.show()

    def save(self, filename: str, dpi: int = None, figsize: tuple = None, format: str = "png") -> None:
        """Save the generated time series plots to a file.

        Args:
            filename (str): Path to the output file.
            dpi (int, optional): Resolution in dots per inch. Defaults to instance dpi.
            figsize (tuple, optional): Figure size (width, height) in inches.
                Defaults to instance figsize.
            format (str): File format (e.g., 'png', 'pdf', 'svg').
        """
        old_dpi = self.dpi
        old_figsize = self.figsize
        if dpi is not None:
            self.dpi = dpi
        if figsize is not None:
            self.figsize = figsize
        fig, _ = self._draw()
        fig.savefig(filename, dpi=self.dpi, bbox_inches="tight", format=format)
        plt.close(fig)
        self.dpi = old_dpi
        self.figsize = old_figsize


def plot_label_distributions_over_time(
    y_chunks: List[np.ndarray], target_chunk: Optional[np.ndarray] = None
) -> None:
    """Label plot.

    Plot how the label distributions (y) change over T time steps,
    with an optional final chunk for the target domain in a different color.

    Args:
        y_chunks (List[np.ndarray]): list of shape (m_i, 1) arrays for each source.
        target_chunk (Optional[np.ndarray]): if provided, shape (m_T, 1) for the
            target domain, to be plotted at time step T in a different color.
    """
    all_chunks = list(y_chunks)
    if target_chunk is not None:
        all_chunks.append(target_chunk)

    time_steps = len(all_chunks)
    means = []
    stds = []

    for i in range(time_steps):
        data_i = all_chunks[i].ravel()
        mu = data_i.mean()
        sd = data_i.std()
        means.append(mu)
        stds.append(sd)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    x_vals = np.arange(time_steps)
    ax1.plot(x_vals, means, color="b", marker="o", label="Mean")
    ax1.fill_between(
        x_vals,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        color="b",
        alpha=0.2,
        label="± std",
    )

    if target_chunk is not None:
        # last index = time_steps - 1
        t_ind = time_steps - 1
        ax1.plot(t_ind, means[t_ind], marker="o", color="red", label="Target domain")
        ax1.fill_between(
            [t_ind, t_ind],
            [means[t_ind] - stds[t_ind]],
            [means[t_ind] + stds[t_ind]],
            color="red",
            alpha=0.2,
        )

    ax1.set_title("Mean ± Std of Label Distributions over Time")
    ax1.set_xlabel("Time step index")
    ax1.set_ylabel("Label value")
    ax1.legend()

    ax2.bar(x_vals, means, color="orange", alpha=0.7)
    if target_chunk is not None:
        # color the last bar differently
        ax2.bar(x_vals[-1], means[-1], color="red", alpha=0.7)
    ax2.set_title("Mean of Label Distributions by Time Step")
    ax2.set_xlabel("Time step index")
    ax2.set_ylabel("Mean label value")

    plt.tight_layout()
    plt.show()


def plot_discrepancy_and_means(
    dbars: List[float], means: List[float], title: str = "True Means vs. Discrepancies"
) -> None:
    """Creates a combined 1x2 plot grid.

    Left plot: Step plot of the per-chunk discrepancy values.
    Right plot: True means and discrepancies over time steps.

    Args:
        dbars (List[float]): Discrepancy values for chunks 0..T-1 in order.
        means (List[float]): The domain means at each time step (length T).
        title (str): Title for the second plot (true means vs discrepancies).

    Raises:
        ValueError: If `means` and `dbars` have different lengths.
    """
    if len(means) != len(dbars):
        raise ValueError("means and dbars must have the same length.")

    xvals = np.arange(len(dbars))

    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Left plot: Step plot of discrepancies
    ax1 = axes[0]
    ax1.step(xvals, dbars, where="post", label="Discrepancy", color="tab:blue")
    ax1.plot(xvals, dbars, marker="o", linestyle="none", color="red")
    ax1.set_xlabel("Time segment index (source chunk)")
    ax1.set_ylabel("Absolute discrepancy")
    ax1.set_title("Step Plot of Discrepancies (Source Chunks vs. Target)")
    ax1.grid(True)
    ax1.legend()

    # Right plot: True means vs discrepancies
    ax2 = axes[1]
    ax2.set_title(title)
    color1 = "tab:blue"
    ax2.set_xlabel("Time step index")

    # Plot means on left y-axis
    ax2.set_ylabel("Mean label", color=color1)
    ax2.plot(xvals, means, marker="o", color=color1, label="Mean label")
    ax2.tick_params(axis="y", labelcolor=color1)

    # Create a twin y-axis for the discrepancy
    ax3 = ax2.twinx()
    color2 = "tab:red"
    ax3.set_ylabel("Discrepancy", color=color2)
    ax3.plot(xvals, dbars, marker="x", color=color2, label="Discrepancy")
    ax3.tick_params(axis="y", labelcolor=color2)

    # Add a legend for the second plot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Adjust layout for clarity
    plt.tight_layout()
    plt.show()
