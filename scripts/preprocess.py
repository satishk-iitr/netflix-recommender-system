#!/usr/bin/env python
"""
preprocess.py — Data preprocessing pipeline for the Netflix Prize dataset.

Usage:
    python scripts/preprocess.py --data-dir data/raw --output-dir data/processed
    python scripts/preprocess.py --data-dir data/raw --output-dir data/processed \\
        --split-method random --min-user-ratings 10

Steps:
    1. Load raw Netflix Prize data files
    2. Encode user/movie IDs to contiguous integers
    3. Filter cold-start users and movies
    4. Split into train / validation / test
    5. Save processed DataFrames + ID mappings
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Data Loading
# ======================================================================

def load_netflix_raw(data_dir: Path) -> pd.DataFrame:
    """Load Netflix Prize raw data files (combined_data_*.txt).

    Each file has the structure:
        movie_id:
        user_id,rating,date
        ...

    Returns a DataFrame with columns: user_id, movie_id, rating, date.
    """
    data_dir = Path(data_dir)
    txt_files = sorted(data_dir.glob("combined_data_*.txt"))

    if not txt_files:
        # Fallback: look for a pre-merged CSV
        csv_path = data_dir / "ratings.csv"
        if csv_path.exists():
            logger.info("Loading pre-merged CSV: %s", csv_path)
            df = pd.read_csv(csv_path)
            for col in ("user_id", "movie_id", "rating"):
                if col not in df.columns:
                    raise ValueError(f"CSV missing required column: {col}")
            return df
        raise FileNotFoundError(
            f"No combined_data_*.txt or ratings.csv found in {data_dir}"
        )

    logger.info("Found %d raw data file(s)", len(txt_files))
    rows = []
    current_movie_id: int | None = None

    for fpath in txt_files:
        logger.info("Reading %s …", fpath.name)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                else:
                    parts = line.split(",")
                    user_id = int(parts[0])
                    rating = float(parts[1])
                    date = parts[2] if len(parts) > 2 else None
                    rows.append((user_id, current_movie_id, rating, date))

    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "date"])
    logger.info("Loaded %d ratings", len(df))
    return df


# ======================================================================
# Encoding
# ======================================================================

def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Map raw user/movie IDs to contiguous 0-based integers.

    Returns
    -------
    df : DataFrame with encoded IDs
    user_map : {original_id: encoded_id}
    movie_map : {original_id: encoded_id}
    """
    user_ids = sorted(df["user_id"].unique())
    movie_ids = sorted(df["movie_id"].unique())

    user_map = {orig: idx for idx, orig in enumerate(user_ids)}
    movie_map = {orig: idx for idx, orig in enumerate(movie_ids)}

    df = df.copy()
    df["user_id"] = df["user_id"].map(user_map)
    df["movie_id"] = df["movie_id"].map(movie_map)

    logger.info(
        "Encoded %d users, %d movies", len(user_map), len(movie_map)
    )
    return df, user_map, movie_map


# ======================================================================
# Filtering
# ======================================================================

def filter_cold_start(
    df: pd.DataFrame,
    min_user_ratings: int = 5,
    min_movie_ratings: int = 5,
) -> pd.DataFrame:
    """Iteratively filter users and movies below minimum rating counts."""
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)

        # Filter users
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df["user_id"].isin(valid_users)]

        # Filter movies
        movie_counts = df["movie_id"].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df["movie_id"].isin(valid_movies)]

    logger.info(
        "After cold-start filter (min_user=%d, min_movie=%d): %d ratings, "
        "%d users, %d movies",
        min_user_ratings,
        min_movie_ratings,
        len(df),
        df["user_id"].nunique(),
        df["movie_id"].nunique(),
    )
    return df.reset_index(drop=True)


# ======================================================================
# Splitting
# ======================================================================

def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date: oldest → train, middle → val, newest → test."""
    if "date" not in df.columns or df["date"].isna().all():
        logger.warning("No date column — falling back to random split")
        return random_split(df, val_frac, test_frac)

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        "Temporal split: train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test


def random_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split per user."""
    rng = np.random.default_rng(seed)

    train_rows, val_rows, test_rows = [], [], []

    for _uid, group in df.groupby("user_id"):
        n = len(group)
        indices = rng.permutation(n)

        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        test_rows.append(group.iloc[test_idx])
        val_rows.append(group.iloc[val_idx])
        train_rows.append(group.iloc[train_idx])

    train = pd.concat(train_rows).reset_index(drop=True)
    val = pd.concat(val_rows).reset_index(drop=True)
    test = pd.concat(test_rows).reset_index(drop=True)

    logger.info(
        "Random split: train=%d, val=%d, test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ======================================================================
# Saving
# ======================================================================

def save_processed(
    output_dir: Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    user_map: dict,
    movie_map: dict,
) -> None:
    """Save all processed artefacts to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    # Save mappings as CSV for easy loading
    pd.DataFrame(
        list(user_map.items()), columns=["original_user_id", "encoded_user_id"]
    ).to_csv(output_dir / "user_mapping.csv", index=False)

    pd.DataFrame(
        list(movie_map.items()), columns=["original_movie_id", "encoded_movie_id"]
    ).to_csv(output_dir / "movie_mapping.csv", index=False)

    # Summary stats
    stats = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "n_users": train["user_id"].nunique(),
        "n_movies": train["movie_id"].nunique(),
        "density": len(train)
        / (train["user_id"].nunique() * train["movie_id"].nunique())
        * 100,
    }
    pd.Series(stats).to_json(output_dir / "stats.json")

    logger.info("Saved processed data to %s", output_dir)
    for k, v in stats.items():
        logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw Netflix Prize data for model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing raw data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to write processed train/val/test splits",
    )
    parser.add_argument(
        "--split-method",
        type=str,
        choices=["temporal", "random"],
        default="temporal",
        help="How to split data into train/val/test",
    )
    parser.add_argument(
        "--min-user-ratings",
        type=int,
        default=5,
        help="Minimum ratings per user (cold-start filter)",
    )
    parser.add_argument(
        "--min-movie-ratings",
        type=int,
        default=5,
        help="Minimum ratings per movie (cold-start filter)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of data for validation set",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of data for test set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load
    df = load_netflix_raw(args.data_dir)

    # 2. Encode IDs
    df, user_map, movie_map = encode_ids(df)

    # 3. Cold-start filtering
    df = filter_cold_start(df, args.min_user_ratings, args.min_movie_ratings)

    # 4. Re-encode after filtering (IDs may no longer be contiguous)
    df, user_map2, movie_map2 = encode_ids(df)
    # Compose original → final mapping
    inv_user = {v: k for k, v in user_map.items()}
    inv_movie = {v: k for k, v in movie_map.items()}
    final_user_map = {inv_user[k]: v for k, v in user_map2.items()}
    final_movie_map = {inv_movie[k]: v for k, v in movie_map2.items()}

    # 5. Split
    if args.split_method == "temporal":
        train, val, test = temporal_split(df, args.val_frac, args.test_frac)
    else:
        train, val, test = random_split(df, args.val_frac, args.test_frac)

    # 6. Save
    save_processed(
        args.output_dir, train, val, test, final_user_map, final_movie_map
    )

    logger.info("Preprocessing complete ✓")


if __name__ == "__main__":
    main()
