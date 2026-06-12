#!/usr/bin/env python
"""
train_svd.py — Train matrix-factorisation models (SVD / SVD++ / NMF) using
the Surprise library.

Usage:
    python scripts/train_svd.py --data-dir data/processed --output-dir models/svd
    python scripts/train_svd.py --data-dir data/processed --model-type svdpp \\
        --grid-search --output-dir models/svdpp

Requires:
    pip install scikit-surprise pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

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
# Data helpers
# ======================================================================

def load_surprise_data(data_dir: Path):
    """Load train/val splits as Surprise Dataset objects.

    Supports both Parquet (default, written by preprocess.py) and CSV.
    """
    from surprise import Dataset, Reader

    def _read(name: str) -> pd.DataFrame:
        parquet_path = data_dir / f"{name}.parquet"
        csv_path = data_dir / f"{name}.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path, engine="pyarrow")
        elif csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"Neither {parquet_path} nor {csv_path} found. "
                "Run scripts/preprocess.py first."
            )

    train_df = _read("train")
    val_df = _read("val")

    reader = Reader(rating_scale=(1, 5))

    train_data = Dataset.load_from_df(
        train_df[["user_id", "movie_id", "rating"]], reader
    )
    trainset = train_data.build_full_trainset()

    # Build anti-testset-style testset from val
    val_tuples = list(
        zip(
            val_df["user_id"].astype(str),
            val_df["movie_id"].astype(str),
            val_df["rating"],
        )
    )

    return trainset, val_tuples, train_df, val_df


# ======================================================================
# Model factory
# ======================================================================

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "svd": {
        "n_factors": 100,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02,
        "random_state": 42,
    },
    "svdpp": {
        "n_factors": 50,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02,
        "random_state": 42,
    },
    "nmf": {
        "n_factors": 15,
        "n_epochs": 50,
        "random_state": 42,
    },
}

GRID_SEARCH_PARAMS: Dict[str, Dict[str, list]] = {
    "svd": {
        "n_factors": [50, 100, 200],
        "n_epochs": [20, 30],
        "lr_all": [0.002, 0.005, 0.01],
        "reg_all": [0.02, 0.05, 0.1],
    },
    "svdpp": {
        "n_factors": [20, 50],
        "n_epochs": [20],
        "lr_all": [0.005, 0.007],
        "reg_all": [0.02, 0.05],
    },
    "nmf": {
        "n_factors": [10, 15, 30],
        "n_epochs": [30, 50],
    },
}


def build_model(model_type: str, params: Optional[Dict] = None):
    """Instantiate a Surprise algorithm."""
    from surprise import SVD, SVDpp, NMF

    algo_map = {"svd": SVD, "svdpp": SVDpp, "nmf": NMF}
    if model_type not in algo_map:
        raise ValueError(f"Unknown model_type: {model_type}")

    final_params = DEFAULT_PARAMS.get(model_type, {}).copy()
    if params:
        final_params.update(params)

    logger.info("Building %s with params: %s", model_type.upper(), final_params)
    return algo_map[model_type](**final_params)


# ======================================================================
# Training
# ======================================================================

def train_model(model, trainset) -> float:
    """Fit the model, return wall-clock training time in seconds."""
    t0 = time.time()
    model.fit(trainset)
    elapsed = time.time() - t0
    logger.info("Training completed in %.1f s", elapsed)
    return elapsed


def evaluate_on_val(model, val_tuples) -> Dict[str, float]:
    """Evaluate on validation set, return RMSE and MAE."""
    from surprise import accuracy

    predictions = model.test(val_tuples)
    rmse_val = accuracy.rmse(predictions, verbose=False)
    mae_val = accuracy.mae(predictions, verbose=False)
    logger.info("Validation RMSE=%.4f  MAE=%.4f", rmse_val, mae_val)
    return {"rmse": rmse_val, "mae": mae_val}


def run_grid_search(model_type: str, trainset, val_tuples):
    """Grid search over hyper-parameters; returns best model + results."""
    from surprise import SVD, SVDpp, NMF
    from surprise.model_selection import GridSearchCV
    from surprise import Dataset, Reader

    algo_map = {"svd": SVD, "svdpp": SVDpp, "nmf": NMF}
    param_grid = GRID_SEARCH_PARAMS.get(model_type, {})

    logger.info("Starting grid search for %s …", model_type.upper())
    logger.info("Parameter grid: %s", param_grid)

    # GridSearchCV needs a dataset, not a trainset
    # Reconstruct from trainset
    raw_ratings = []
    for uid, iid, rating in trainset.all_ratings():
        raw_uid = trainset.to_raw_uid(uid)
        raw_iid = trainset.to_raw_iid(iid)
        raw_ratings.append((raw_uid, raw_iid, rating))

    df_temp = pd.DataFrame(raw_ratings, columns=["user_id", "movie_id", "rating"])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_temp, reader)

    gs = GridSearchCV(
        algo_map[model_type],
        param_grid,
        measures=["rmse", "mae"],
        cv=3,
        n_jobs=-1,
    )
    gs.fit(data)

    logger.info("Best RMSE: %.4f", gs.best_score["rmse"])
    logger.info("Best params: %s", gs.best_params["rmse"])

    # Retrain best model on full training set
    best_model = build_model(model_type, gs.best_params["rmse"])
    train_model(best_model, trainset)
    val_metrics = evaluate_on_val(best_model, val_tuples)

    return best_model, {
        "best_cv_rmse": gs.best_score["rmse"],
        "best_params": gs.best_params["rmse"],
        "val_metrics": val_metrics,
    }


# ======================================================================
# Saving
# ======================================================================

def save_model(
    model,
    output_dir: Path,
    model_type: str,
    results: Dict[str, Any],
    train_time: float,
) -> None:
    """Persist model and training metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = output_dir / f"{model_type}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", model_path)

    # Results
    results_out = {
        "model_type": model_type,
        "train_time_s": round(train_time, 2),
        **results,
    }
    results_path = output_dir / f"{model_type}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2, default=str)
    logger.info("Results saved to %s", results_path)


# ======================================================================
# Config loading
# ======================================================================

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML config if provided."""
    if config_path is None:
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed; ignoring --config")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return cfg or {}


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SVD / SVD++ / NMF on processed Netflix data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed train.csv, val.csv",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["svd", "svdpp", "nmf"],
        default="svd",
        help="Matrix factorisation algorithm",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config with hyper-parameters",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run grid search over hyper-parameter space",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save trained model and results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Load data
    trainset, val_tuples, train_df, val_df = load_surprise_data(data_dir)
    logger.info(
        "Data: %d train ratings, %d val ratings",
        trainset.n_ratings,
        len(val_tuples),
    )

    # Optionally load config overrides
    config = load_config(args.config)

    if args.grid_search:
        best_model, gs_results = run_grid_search(
            args.model_type, trainset, val_tuples
        )
        save_model(best_model, output_dir, args.model_type, gs_results, 0)
    else:
        model = build_model(args.model_type, config.get("model_params"))
        train_time = train_model(model, trainset)
        val_metrics = evaluate_on_val(model, val_tuples)
        save_model(
            model,
            output_dir,
            args.model_type,
            {"val_metrics": val_metrics},
            train_time,
        )

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
