"""
Visualization utilities for the Netflix Recommendation System.

Every public function returns a ``matplotlib.figure.Figure`` so callers
can display, save, or further customise the plot.  An optional
``save_path`` parameter (str | Path) writes the figure to disk when provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("viridis", n_colors=8)
MODEL_PALETTE = {"SVD": "#4C72B0", "NeuMF": "#DD8452", "LightGCN": "#55A868"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: plt.Figure,
    name: str,
    directory: Union[str, Path],
    dpi: int = 300,
    tight: bool = True,
) -> Path:
    """
    Persist a matplotlib figure to *directory/name*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        File name (e.g. ``"rating_dist.png"``).
    directory : str or Path
        Target directory (created if missing).
    dpi : int
        Resolution in dots-per-inch.
    tight : bool
        Whether to apply ``bbox_inches='tight'``.

    Returns
    -------
    Path
        Absolute path of the saved image.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight" if tight else None)
    return out


def _maybe_save(fig: plt.Figure, save_path: Optional[Union[str, Path]]) -> None:
    """Save *fig* if a path is supplied."""
    if save_path is not None:
        p = Path(save_path)
        save_figure(fig, p.name, p.parent)


# ---------------------------------------------------------------------------
# Rating distribution
# ---------------------------------------------------------------------------


def plot_rating_distribution(
    ratings_df: pd.DataFrame,
    *,
    rating_col: str = "rating",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Bar chart showing the count of each discrete rating value (1–5).

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain a column *rating_col* with numeric ratings.
    rating_col : str
        Column name for the rating values.
    save_path : str or Path, optional
        If given, the figure is written to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = ratings_df[rating_col].value_counts().sort_index()
    ax.bar(
        counts.index.astype(str),
        counts.values,
        color=PALETTE[: len(counts)],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Rating Distribution")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    for bar, count in zip(ax.patches, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# User activity
# ---------------------------------------------------------------------------


def plot_user_activity(
    ratings_df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Log-scale histogram of the number of ratings per user.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain *user_col*.
    user_col : str
        Column name identifying users.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ratings_per_user = ratings_df[user_col].value_counts()
    ax.hist(
        ratings_per_user.values,
        bins=np.logspace(0, np.log10(ratings_per_user.max() + 1), 60),
        color=PALETTE[1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Ratings per User (log scale)")
    ax.set_ylabel("Number of Users (log scale)")
    ax.set_title("User Activity Distribution")

    median_val = ratings_per_user.median()
    ax.axvline(median_val, color="red", linestyle="--", linewidth=1.2, label=f"Median = {median_val:.0f}")
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Movie popularity
# ---------------------------------------------------------------------------


def plot_movie_popularity(
    ratings_df: pd.DataFrame,
    *,
    movie_col: str = "movie_id",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Log-scale histogram of the number of ratings per movie.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain *movie_col*.
    movie_col : str
        Column name identifying movies.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ratings_per_movie = ratings_df[movie_col].value_counts()
    ax.hist(
        ratings_per_movie.values,
        bins=np.logspace(0, np.log10(ratings_per_movie.max() + 1), 60),
        color=PALETTE[3],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Ratings per Movie (log scale)")
    ax.set_ylabel("Number of Movies (log scale)")
    ax.set_title("Movie Popularity Distribution")

    median_val = ratings_per_movie.median()
    ax.axvline(median_val, color="red", linestyle="--", linewidth=1.2, label=f"Median = {median_val:.0f}")
    ax.legend()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Temporal trends
# ---------------------------------------------------------------------------


def plot_temporal_trends(
    ratings_df: pd.DataFrame,
    *,
    date_col: str = "date",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Line chart of ratings count aggregated by month.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Must contain *date_col* (datetime or convertible).
    date_col : str
        Column with timestamps.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    df = ratings_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    monthly = df.set_index(date_col).resample("ME").size()
    ax.plot(monthly.index, monthly.values, color=PALETTE[2], linewidth=1.5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.25, color=PALETTE[2])
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Ratings")
    ax.set_title("Ratings Volume Over Time (Monthly)")
    fig.autofmt_xdate()
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Sparsity heatmap
# ---------------------------------------------------------------------------


def plot_sparsity_heatmap(
    ratings_df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    movie_col: str = "movie_id",
    rating_col: str = "rating",
    sample_users: int = 500,
    sample_movies: int = 500,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Heatmap of a sampled user × movie sub-matrix to visualise sparsity.

    Parameters
    ----------
    ratings_df : pd.DataFrame
    sample_users : int
        Number of random users to include.
    sample_movies : int
        Number of random movies to include.
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    rng = np.random.RandomState(42)

    unique_users = ratings_df[user_col].unique()
    unique_movies = ratings_df[movie_col].unique()

    sampled_users = rng.choice(unique_users, min(sample_users, len(unique_users)), replace=False)
    sampled_movies = rng.choice(unique_movies, min(sample_movies, len(unique_movies)), replace=False)

    sub = ratings_df[
        ratings_df[user_col].isin(sampled_users) & ratings_df[movie_col].isin(sampled_movies)
    ]
    pivot = sub.pivot_table(index=user_col, columns=movie_col, values=rating_col)

    sparsity = 1.0 - pivot.notna().sum().sum() / (pivot.shape[0] * pivot.shape[1])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pivot.notna().values.astype(float), aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel(f"Movies (sampled {pivot.shape[1]})")
    ax.set_ylabel(f"Users (sampled {pivot.shape[0]})")
    ax.set_title(f"User–Movie Interaction Matrix  (sparsity = {sparsity:.2%})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def plot_model_comparison(
    results_dict: Dict[str, Dict[str, float]],
    *,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing RMSE and MAP@10 across models.

    Parameters
    ----------
    results_dict : dict
        ``{ "SVD": {"RMSE": 0.91, "MAP@10": 0.08}, ... }``
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    models = list(results_dict.keys())
    metrics = ["RMSE", "MAP@10"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results_dict[m].get(metric, 0) for m in models]
        colors = [MODEL_PALETTE.get(m, PALETTE[idx]) for m in models]
        bars = ax.bar(models, values, color=colors, edgecolor="white", linewidth=0.8)

        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_ylabel(metric)
        lower_better = metric == "RMSE"
        if lower_better:
            ax.set_ylim(0, max(values) * 1.25 if values else 1)
        else:
            ax.set_ylim(0, max(values) * 1.3 if values else 1)

    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Full metric table
# ---------------------------------------------------------------------------


def plot_metric_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    *,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Render a styled table of all evaluation metrics per model.

    Parameters
    ----------
    results_dict : dict
        ``{ "SVD": {"RMSE": 0.91, "MAE": 0.71, ...}, ... }``
    save_path : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame(results_dict).T
    df.index.name = "Model"

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 1.8), 1 + len(df) * 0.6))
    ax.axis("off")

    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns.tolist(),
        rowLabels=df.index.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C72B0")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#fafafa" if row % 2 == 0 else "white")

    ax.set_title("Evaluation Metrics Summary", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig
