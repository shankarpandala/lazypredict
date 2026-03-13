"""Time series forecasting visualization functions.

Provides plotting utilities for forecast analysis, model comparison,
residual diagnostics, and error distribution. All functions require
matplotlib (optional dependency).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("lazypredict")

try:
    import matplotlib
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    """Raise ImportError if matplotlib is not installed."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install lazypredict[viz]"
        )


def plot_forecast(
    y_train: np.ndarray,
    y_test: np.ndarray,
    predictions: Union[np.ndarray, Dict[str, np.ndarray]],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    ax: Optional[Any] = None,
) -> Any:
    """Plot actual series with forecast overlay.

    Parameters
    ----------
    y_train : np.ndarray
        Training series shown as historical context.
    y_test : np.ndarray
        Actual test values.
    predictions : np.ndarray or dict
        Single prediction array or ``{model_name: y_pred}`` dict.
    title : str or None
        Plot title.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes or None
        Axes to plot on. Created if None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_train = len(y_train)
    n_test = len(y_test)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_train + n_test)

    ax.plot(train_idx, y_train, color="steelblue", label="Train", alpha=0.7)
    ax.plot(test_idx, y_test, color="black", label="Actual", linewidth=2)

    if isinstance(predictions, dict):
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(predictions), 1)))
        for i, (name, y_pred) in enumerate(predictions.items()):
            ax.plot(
                test_idx[: len(y_pred)],
                y_pred,
                label=name,
                color=colors[i % len(colors)],
                linestyle="--",
                linewidth=1.5,
            )
    else:
        ax.plot(
            test_idx[: len(predictions)],
            predictions,
            label="Forecast",
            color="red",
            linestyle="--",
            linewidth=1.5,
        )

    ax.axvline(x=n_train, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title or "Forecast vs Actual")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_model_comparison(
    scores_df: pd.DataFrame,
    metric: str = "RMSE",
    top_k: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[Any] = None,
) -> Any:
    """Horizontal bar chart comparing models by a metric.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Scores table from ``LazyForecaster.fit()`` (index = model names).
    metric : str
        Metric column to plot.
    top_k : int
        Show only the top k models.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()

    if metric not in scores_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available: {list(scores_df.columns)}"
        )

    data = scores_df[metric].dropna().head(top_k)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(data)))
    ax.barh(range(len(data)), data.values, color=colors)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison by {metric}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: Optional[str] = None,
    seasonal_period: int = 1,
    figsize: Tuple[int, int] = (12, 10),
) -> Any:
    """Four-panel residual diagnostic plot.

    Panels:
        1. Residuals over time (scatter with zero line)
        2. Histogram of residuals
        3. QQ plot against normal distribution
        4. ACF of residuals with confidence bands

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted values.
    model_name : str or None
        Model name for the title.
    seasonal_period : int
        Seasonal period for ACF display.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    title = f"Residual Diagnostics — {model_name}" if model_name else "Residual Diagnostics"
    fig.suptitle(title, fontsize=14)

    # Panel 1: Residuals over time
    ax1 = axes[0, 0]
    ax1.scatter(range(len(residuals)), residuals, s=20, alpha=0.7, color="steelblue")
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Residual")
    ax1.set_title("Residuals Over Time")

    # Panel 2: Histogram
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins="auto", density=True, alpha=0.7, color="steelblue", edgecolor="white")
    # Overlay normal curve
    mu, sigma = np.mean(residuals), np.std(residuals)
    if sigma > 0:
        x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        ax2.plot(x_norm, y_norm, color="red", linewidth=1.5, label="Normal")
        ax2.legend(fontsize=8)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution")

    # Panel 3: QQ Plot
    ax3 = axes[1, 0]
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = np.array([
        _norm_ppf((i - 0.5) / n) for i in range(1, n + 1)
    ])
    ax3.scatter(theoretical_quantiles, sorted_residuals, s=20, alpha=0.7, color="steelblue")
    # Reference line
    q1_idx, q3_idx = max(0, n // 4 - 1), min(n - 1, 3 * n // 4 - 1)
    if q3_idx > q1_idx:
        tq1, tq3 = theoretical_quantiles[q1_idx], theoretical_quantiles[q3_idx]
        sq1, sq3 = sorted_residuals[q1_idx], sorted_residuals[q3_idx]
        slope = (sq3 - sq1) / (tq3 - tq1) if tq3 != tq1 else 1
        intercept = sq1 - slope * tq1
        line_x = np.array([theoretical_quantiles[0], theoretical_quantiles[-1]])
        ax3.plot(line_x, slope * line_x + intercept, color="red", linestyle="--", linewidth=1)
    ax3.set_xlabel("Theoretical Quantiles")
    ax3.set_ylabel("Sample Quantiles")
    ax3.set_title("Q-Q Plot")

    # Panel 4: ACF
    ax4 = axes[1, 1]
    max_lag = min(len(residuals) - 1, max(20, 2 * seasonal_period))
    acf_values = _compute_acf(residuals, max_lag)
    conf_bound = 1.96 / np.sqrt(len(residuals))
    ax4.bar(range(len(acf_values)), acf_values, width=0.3, color="steelblue")
    ax4.axhline(y=conf_bound, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.axhline(y=-conf_bound, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.axhline(y=0, color="black", linewidth=0.5)
    ax4.set_xlabel("Lag")
    ax4.set_ylabel("ACF")
    ax4.set_title("Autocorrelation of Residuals")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 6),
    ax: Optional[Any] = None,
) -> Any:
    """Box plot of absolute errors across models.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    predictions : dict
        ``{model_name: y_pred}`` dict.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()

    y_true = np.asarray(y_true, dtype=float)
    if not predictions:
        raise ValueError("No predictions provided.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    error_data = []
    labels = []
    for name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred, dtype=float)
        abs_errors = np.abs(y_true[: len(y_pred)] - y_pred)
        error_data.append(abs_errors)
        labels.append(name)

    bp = ax.boxplot(error_data, patch_artist=True, tick_labels=labels)
    colors = plt.cm.Set2(np.linspace(0, 1, len(error_data)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Absolute Error")
    ax.set_title("Error Distribution by Model")
    if len(labels) > 5:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_metrics_heatmap(
    scores_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    ax: Optional[Any] = None,
) -> Any:
    """Heatmap of normalized metrics across models.

    Metrics are min-max normalized to [0, 1] for visual comparison.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Scores table from ``LazyForecaster.fit()``.
    metrics : list of str or None
        Metric columns to include. Defaults to MAE, RMSE, MAPE, SMAPE, MASE.
    figsize : tuple
        Figure size.
    ax : matplotlib Axes or None
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _check_matplotlib()

    if metrics is None:
        metrics = [m for m in ["MAE", "RMSE", "MAPE", "SMAPE", "MASE"] if m in scores_df.columns]
    if not metrics:
        raise ValueError("No valid metrics found in scores DataFrame.")

    data = scores_df[metrics].dropna()
    if data.empty:
        raise ValueError("No data to plot after dropping NaN values.")

    # Min-max normalize each column
    normalized = data.copy()
    for col in metrics:
        col_min, col_max = data[col].min(), data[col].max()
        if col_max > col_min:
            normalized[col] = (data[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(normalized.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(normalized)))
    ax.set_yticklabels(normalized.index, fontsize=9)

    # Annotate with original values
    for i in range(len(normalized)):
        for j in range(len(metrics)):
            val = data.iloc[i, j]
            text_color = "white" if normalized.iloc[i, j] > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_title("Metrics Heatmap (normalized)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized (lower=better)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation function up to max_lag."""
    x = x - np.mean(x)
    n = len(x)
    var = np.sum(x ** 2) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.array([
        np.sum(x[: n - k] * x[k:]) / (n * var)
        for k in range(max_lag + 1)
    ])
    return acf


def _norm_ppf(p: float) -> float:
    """Approximate normal quantile function (inverse CDF).

    Uses the rational approximation by Abramowitz and Stegun.
    """
    if p <= 0:
        return -6.0
    if p >= 1:
        return 6.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        t = np.sqrt(-2.0 * np.log(p))
    else:
        t = np.sqrt(-2.0 * np.log(1.0 - p))

    # Rational approximation coefficients
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    result = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

    return result if p > 0.5 else -result
