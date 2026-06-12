"""
Netflix Prize Data Preprocessor
================================
Encodes raw IDs into contiguous integers, computes dataset statistics,
filters cold-start entities, and produces temporal / random train-val-test
splits suitable for recommendation model training.

Design notes
------------
*   The class stores ``user2idx`` / ``idx2user`` (and movie equivalents)
    mappings as plain Python dicts — they serialise easily and are fast for
    the lookup sizes we encounter (~480 K users, ~18 K movies).
*   Split functions operate **per-user** so every user appears in every
    partition — critical for fair evaluation of personalised models.
*   All public methods accept and return either Polars or Pandas DataFrames
    transparently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

import pandas as pd

logger = logging.getLogger(__name__)

DataFrame = Union["pl.DataFrame", "pd.DataFrame"]


class DataPreprocessor:
    """Preprocess Netflix Prize ratings for model consumption.

    Attributes
    ----------
    user2idx : dict[int, int]
        Mapping from original user IDs to contiguous indices.
    idx2user : dict[int, int]
        Reverse mapping.
    movie2idx : dict[int, int]
        Mapping from original movie IDs to contiguous indices.
    idx2movie : dict[int, int]
        Reverse mapping.
    n_users : int
        Number of unique users after encoding.
    n_movies : int
        Number of unique movies after encoding.
    """

    def __init__(self) -> None:
        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.movie2idx: Dict[int, int] = {}
        self.idx2movie: Dict[int, int] = {}
        self.n_users: int = 0
        self.n_movies: int = 0

    # ------------------------------------------------------------------
    # ID encoding
    # ------------------------------------------------------------------
    def encode_ids(self, df: DataFrame) -> DataFrame:
        """Create contiguous 0-based integer encodings for user and movie IDs.

        Stores the forward and reverse mappings on the instance for later
        use (e.g. decoding predictions back to original IDs).

        Parameters
        ----------
        df : DataFrame
            Must contain ``user_id`` and ``movie_id`` columns.

        Returns
        -------
        DataFrame
            Same DataFrame with ``user_idx`` and ``movie_idx`` columns added.
        """
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)

        if is_polars:
            unique_users = sorted(df["user_id"].unique().to_list())
            unique_movies = sorted(df["movie_id"].unique().to_list())
        else:
            unique_users = sorted(df["user_id"].unique().tolist())
            unique_movies = sorted(df["movie_id"].unique().tolist())

        self.user2idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx2user = {idx: uid for uid, idx in self.user2idx.items()}
        self.movie2idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx2movie = {idx: mid for mid, idx in self.movie2idx.items()}
        self.n_users = len(unique_users)
        self.n_movies = len(unique_movies)

        logger.info(
            "Encoded %s users and %s movies to contiguous indices.",
            f"{self.n_users:,}",
            f"{self.n_movies:,}",
        )

        if is_polars:
            # Vectorized mapping — orders of magnitude faster than map_elements
            # on 100M+ rows.
            return df.with_columns(
                pl.col("user_id").replace(self.user2idx, default=None).cast(pl.Int64).alias("user_idx"),
                pl.col("movie_id").replace(self.movie2idx, default=None).cast(pl.Int64).alias("movie_idx"),
            )
        else:
            df = df.copy()
            df["user_idx"] = df["user_id"].map(self.user2idx).astype(int)
            df["movie_idx"] = df["movie_id"].map(self.movie2idx).astype(int)
            return df

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------
    def temporal_split(
        self,
        df: DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Split ratings chronologically **per user**.

        For each user the ratings are sorted by date, then the first
        ``train_ratio`` fraction becomes training data, the next
        ``val_ratio`` becomes validation, and the rest becomes test.

        Parameters
        ----------
        df : DataFrame
            Must contain ``user_id`` (or ``user_idx``) and ``date`` columns.
        train_ratio, val_ratio, test_ratio : float
            Must sum to 1.0.

        Returns
        -------
        tuple[DataFrame, DataFrame, DataFrame]
            ``(train, val, test)`` DataFrames.
        """
        self._validate_ratios(train_ratio, val_ratio, test_ratio)
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)

        if is_polars:
            pdf = df.to_pandas()
        else:
            pdf = df.copy()

        user_col = "user_idx" if "user_idx" in pdf.columns else "user_id"
        pdf = pdf.sort_values([user_col, "date"]).reset_index(drop=True)

        train_rows, val_rows, test_rows = [], [], []

        for _, group in pdf.groupby(user_col):
            n = len(group)
            train_end = max(1, int(n * train_ratio))
            val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))

            train_rows.append(group.iloc[:train_end])
            val_rows.append(group.iloc[train_end:val_end])
            test_rows.append(group.iloc[val_end:])

        train_pd = pd.concat(train_rows, ignore_index=True)
        val_pd = pd.concat(val_rows, ignore_index=True)
        test_pd = pd.concat(test_rows, ignore_index=True)

        logger.info(
            "Temporal split — train: %s, val: %s, test: %s",
            f"{len(train_pd):,}",
            f"{len(val_pd):,}",
            f"{len(test_pd):,}",
        )

        if is_polars:
            return (
                pl.from_pandas(train_pd),
                pl.from_pandas(val_pd),
                pl.from_pandas(test_pd),
            )
        return train_pd, val_pd, test_pd

    def random_split(
        self,
        df: DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Randomly split ratings into train / val / test sets.

        Parameters
        ----------
        df : DataFrame
            Ratings DataFrame.
        train_ratio, val_ratio, test_ratio : float
            Must sum to 1.0.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        tuple[DataFrame, DataFrame, DataFrame]
            ``(train, val, test)`` DataFrames.
        """
        self._validate_ratios(train_ratio, val_ratio, test_ratio)
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)

        n = len(df)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        if is_polars:
            # Polars uses gather for integer-index selection
            train_df = df[train_idx.tolist()]
            val_df = df[val_idx.tolist()]
            test_df = df[test_idx.tolist()]
        else:
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

        logger.info(
            "Random split — train: %s, val: %s, test: %s",
            f"{len(train_df):,}",
            f"{len(val_df):,}",
            f"{len(test_df):,}",
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_user_stats(self, df: DataFrame) -> DataFrame:
        """Per-user rating statistics.

        Returns
        -------
        DataFrame
            Columns: ``[user_id, n_ratings, avg_rating, std_rating]``.
        """
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)
        user_col = "user_idx" if ("user_idx" in (df.columns if is_polars else df.columns)) else "user_id"

        if is_polars:
            return (
                df.group_by(user_col)
                .agg(
                    pl.count().alias("n_ratings"),
                    pl.col("rating").mean().alias("avg_rating"),
                    pl.col("rating").std().alias("std_rating"),
                )
                .sort(user_col)
            )
        else:
            stats = (
                df.groupby(user_col)["rating"]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            stats.columns = [user_col, "n_ratings", "avg_rating", "std_rating"]
            return stats.sort_values(user_col).reset_index(drop=True)

    def get_movie_stats(self, df: DataFrame) -> DataFrame:
        """Per-movie rating statistics.

        Returns
        -------
        DataFrame
            Columns: ``[movie_id, n_ratings, avg_rating, std_rating]``.
        """
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)
        movie_col = "movie_idx" if ("movie_idx" in (df.columns if is_polars else df.columns)) else "movie_id"

        if is_polars:
            return (
                df.group_by(movie_col)
                .agg(
                    pl.count().alias("n_ratings"),
                    pl.col("rating").mean().alias("avg_rating"),
                    pl.col("rating").std().alias("std_rating"),
                )
                .sort(movie_col)
            )
        else:
            stats = (
                df.groupby(movie_col)["rating"]
                .agg(["count", "mean", "std"])
                .reset_index()
            )
            stats.columns = [movie_col, "n_ratings", "avg_rating", "std_rating"]
            return stats.sort_values(movie_col).reset_index(drop=True)

    def get_sparsity(self, df: DataFrame) -> float:
        """Compute the sparsity of the user-item interaction matrix.

        Sparsity = 1 − (n_ratings / (n_users × n_movies)).

        Parameters
        ----------
        df : DataFrame
            Ratings DataFrame with ``user_id`` (or ``user_idx``) and
            ``movie_id`` (or ``movie_idx``) columns.

        Returns
        -------
        float
            Sparsity in [0, 1].  Higher means sparser.
        """
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)
        user_col = "user_idx" if ("user_idx" in (df.columns if is_polars else df.columns)) else "user_id"
        movie_col = "movie_idx" if ("movie_idx" in (df.columns if is_polars else df.columns)) else "movie_id"

        if is_polars:
            n_users = df[user_col].n_unique()
            n_movies = df[movie_col].n_unique()
        else:
            n_users = df[user_col].nunique()
            n_movies = df[movie_col].nunique()

        n_ratings = len(df)
        sparsity = 1.0 - n_ratings / (n_users * n_movies)
        logger.info(
            "Sparsity: %.4f%% (%s ratings, %s users, %s movies)",
            sparsity * 100,
            f"{n_ratings:,}",
            f"{n_users:,}",
            f"{n_movies:,}",
        )
        return sparsity

    # ------------------------------------------------------------------
    # Cold-start filtering
    # ------------------------------------------------------------------
    def filter_cold_start(
        self,
        df: DataFrame,
        min_user_ratings: int = 5,
        min_movie_ratings: int = 5,
    ) -> DataFrame:
        """Iteratively remove users and movies with fewer than *min* ratings.

        The process repeats until convergence because removing movies may
        cause some users to drop below threshold and vice-versa.

        Parameters
        ----------
        df : DataFrame
            Ratings DataFrame.
        min_user_ratings : int
            Minimum number of ratings a user must have.
        min_movie_ratings : int
            Minimum number of ratings a movie must have.

        Returns
        -------
        DataFrame
            Filtered DataFrame.
        """
        is_polars = _HAS_POLARS and isinstance(df, pl.DataFrame)
        user_col = "user_idx" if ("user_idx" in (df.columns if is_polars else df.columns)) else "user_id"
        movie_col = "movie_idx" if ("movie_idx" in (df.columns if is_polars else df.columns)) else "movie_id"

        prev_len = -1
        iteration = 0

        while len(df) != prev_len:
            prev_len = len(df)
            iteration += 1

            if is_polars:
                # Filter users
                user_counts = df.group_by(user_col).agg(pl.count().alias("cnt"))
                valid_users = user_counts.filter(pl.col("cnt") >= min_user_ratings)[user_col]
                df = df.filter(pl.col(user_col).is_in(valid_users))
                # Filter movies
                movie_counts = df.group_by(movie_col).agg(pl.count().alias("cnt"))
                valid_movies = movie_counts.filter(pl.col("cnt") >= min_movie_ratings)[movie_col]
                df = df.filter(pl.col(movie_col).is_in(valid_movies))
            else:
                user_counts = df[user_col].value_counts()
                valid_users = user_counts[user_counts >= min_user_ratings].index
                df = df[df[user_col].isin(valid_users)].reset_index(drop=True)

                movie_counts = df[movie_col].value_counts()
                valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
                df = df[df[movie_col].isin(valid_movies)].reset_index(drop=True)

            logger.debug(
                "Cold-start filter iteration %d: %s ratings remaining",
                iteration,
                f"{len(df):,}",
            )

        logger.info(
            "Cold-start filtering converged after %d iterations: %s ratings.",
            iteration,
            f"{len(df):,}",
        )
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_splits(
        self,
        train: DataFrame,
        val: DataFrame,
        test: DataFrame,
        output_dir: Union[str, Path],
    ) -> None:
        """Save train/val/test DataFrames as Parquet files and ID mappings as JSON.

        Parameters
        ----------
        train, val, test : DataFrame
            Split DataFrames.
        output_dir : str or Path
            Directory to write files into.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, split_df in [("train", train), ("val", val), ("test", test)]:
            path = output_dir / f"{name}.parquet"
            if _HAS_POLARS and isinstance(split_df, pl.DataFrame):
                split_df.write_parquet(path)
            else:
                split_df.to_parquet(path, index=False, engine="pyarrow")
            logger.info("Saved %s split (%s rows) → %s", name, f"{len(split_df):,}", path)

        # Save ID mappings if available
        if self.user2idx:
            mappings = {
                "user2idx": {str(k): v for k, v in self.user2idx.items()},
                "movie2idx": {str(k): v for k, v in self.movie2idx.items()},
                "n_users": self.n_users,
                "n_movies": self.n_movies,
            }
            mappings_path = output_dir / "id_mappings.json"
            with open(mappings_path, "w", encoding="utf-8") as fh:
                json.dump(mappings, fh)
            logger.info("Saved ID mappings → %s", mappings_path)

    def load_splits(
        self,
        output_dir: Union[str, Path],
        use_polars: Optional[bool] = None,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """Load previously saved train/val/test Parquet splits.

        Also restores ID mappings if ``id_mappings.json`` exists.

        Parameters
        ----------
        output_dir : str or Path
            Directory containing ``train.parquet``, ``val.parquet``,
            ``test.parquet``.
        use_polars : bool, optional
            If *None*, uses Polars when available.

        Returns
        -------
        tuple[DataFrame, DataFrame, DataFrame]
            ``(train, val, test)`` DataFrames.
        """
        output_dir = Path(output_dir)
        if use_polars is None:
            use_polars = _HAS_POLARS

        splits = []
        for name in ("train", "val", "test"):
            path = output_dir / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}")
            if use_polars and _HAS_POLARS:
                splits.append(pl.read_parquet(path))
            else:
                splits.append(pd.read_parquet(path, engine="pyarrow"))
            logger.info("Loaded %s split (%s rows) ← %s", name, f"{len(splits[-1]):,}", path)

        # Restore ID mappings
        mappings_path = output_dir / "id_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, "r", encoding="utf-8") as fh:
                mappings = json.load(fh)
            self.user2idx = {int(k): v for k, v in mappings["user2idx"].items()}
            self.idx2user = {v: k for k, v in self.user2idx.items()}
            self.movie2idx = {int(k): v for k, v in mappings["movie2idx"].items()}
            self.idx2movie = {v: k for k, v in self.movie2idx.items()}
            self.n_users = mappings["n_users"]
            self.n_movies = mappings["n_movies"]
            logger.info("Restored ID mappings (%d users, %d movies).", self.n_users, self.n_movies)

        return tuple(splits)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_ratios(
        train_ratio: float, val_ratio: float, test_ratio: float
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f} "
                f"({train_ratio} + {val_ratio} + {test_ratio})"
            )
