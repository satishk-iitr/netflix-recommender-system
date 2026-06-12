#!/usr/bin/env python
"""
evaluate_all.py — Evaluate all trained models on the test set and compare.

Usage:
    python scripts/evaluate_all.py \\
        --data-dir data/processed \\
        --models-dir models \\
        --output-dir results

Steps:
    1. Load test set
    2. Discover and load all trained models (SVD, NeuMF, LightGCN)
    3. Evaluate each on test set (RMSE + MAP@10)
    4. Evaluate a simple ensemble (average)
    5. Print comparison table
    6. Save results to JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import Evaluator
from src.evaluation.metrics import rmse, mae, map_at_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Model Loaders
# ======================================================================

def load_svd_model(model_dir: Path):
    """Load a Surprise SVD/SVDpp/NMF model."""
    for fname in ("svd_model.pkl", "svdpp_model.pkl", "nmf_model.pkl"):
        path = model_dir / fname
        if path.exists():
            with open(path, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded %s", path)
            return model, fname.replace("_model.pkl", "")
    return None, None


def load_torch_model(model_dir: Path, model_type: str, data_dir: Path):
    """Load a PyTorch model (NeuMF or LightGCN).

    Requires the results JSON to know architecture params.
    """
    import torch

    results_path = model_dir / f"{model_type}_results.json"
    model_path = model_dir / f"{model_type}_model.pt"

    if not model_path.exists() or not results_path.exists():
        return None

    with open(results_path, "r") as f:
        meta = json.load(f)

    n_users = meta["n_users"]
    n_items = meta["n_items"]
    config = meta.get("config", {})

    if model_type == "neumf":
        # Import the builder from training script
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from train_neumf import build_neumf_model

        model = build_neumf_model(n_users, n_items, config)
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
    elif model_type == "lightgcn":
        from train_lightgcn import build_lightgcn_model, build_adj_matrix

        model = build_lightgcn_model(n_users, n_items, config)
        model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        # LightGCN needs adjacency matrix
        train_df = pd.read_csv(data_dir / "train.csv")
        adj = build_adj_matrix(train_df, n_users, n_items)
        model.set_adj(adj)
    else:
        return None

    model.eval()
    logger.info("Loaded %s from %s", model_type, model_path)
    return model


# ======================================================================
# Prediction helpers
# ======================================================================

def predict_surprise(model, user_id: int, item_id: int) -> float:
    """Predict rating using a Surprise model."""
    pred = model.predict(str(user_id), str(item_id))
    return pred.est


def predict_torch(model, user_id: int, item_id: int) -> float:
    """Predict rating using a PyTorch model."""
    return model.predict(user_id, item_id)


def generate_predictions(
    model,
    predict_fn,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions for all test rows."""
    preds = []
    for _, row in test_df.iterrows():
        uid, iid = int(row["user_id"]), int(row["movie_id"])
        try:
            score = predict_fn(model, uid, iid)
        except Exception:
            score = 3.0
        preds.append(score)
    result = test_df[["user_id", "movie_id", "rating"]].copy()
    result["predicted_rating"] = preds
    return result


def generate_topk(
    model,
    predict_fn,
    user_ids: List[int],
    all_items: List[int],
    train_df: pd.DataFrame,
    k: int = 10,
) -> Dict[int, List[int]]:
    """Generate top-K recommendations per user."""
    # Build seen-item index
    user_seen: Dict[int, set] = {}
    for uid, group in train_df.groupby("user_id"):
        user_seen[int(uid)] = set(group["movie_id"].tolist())

    user_recs: Dict[int, List[int]] = {}
    for uid in user_ids:
        seen = user_seen.get(uid, set())
        candidates = [iid for iid in all_items if iid not in seen]

        scores = []
        for iid in candidates:
            try:
                s = predict_fn(model, uid, iid)
            except Exception:
                s = 0.0
            scores.append((iid, float(s)))

        scores.sort(key=lambda x: x[1], reverse=True)
        user_recs[uid] = [iid for iid, _ in scores[:k]]

    return user_recs


# ======================================================================
# Ensemble
# ======================================================================

def ensemble_predictions(
    all_predictions: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Simple average ensemble across model predictions."""
    if not all_predictions:
        raise ValueError("No predictions to ensemble")

    dfs = list(all_predictions.values())
    base = dfs[0][["user_id", "movie_id", "rating"]].copy()

    # Average predicted ratings
    pred_cols = []
    for name, df in all_predictions.items():
        col_name = f"pred_{name}"
        base[col_name] = df["predicted_rating"].values
        pred_cols.append(col_name)

    base["predicted_rating"] = base[pred_cols].mean(axis=1)
    return base[["user_id", "movie_id", "rating", "predicted_rating"]]


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models and produce a comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=10, help="Top-K for ranking metrics")
    parser.add_argument(
        "--n-sample-users",
        type=int,
        default=200,
        help="Number of users to sample for ranking evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df = pd.read_csv(data_dir / "train.csv")
    all_items = sorted(
        set(train_df["movie_id"].unique()) | set(test_df["movie_id"].unique())
    )

    evaluator = Evaluator(test_df, relevance_threshold=3.5, k=args.k)

    # Sample users for ranking
    rng = np.random.default_rng(42)
    all_test_users = test_df["user_id"].unique()
    sample_users = rng.choice(
        all_test_users,
        size=min(args.n_sample_users, len(all_test_users)),
        replace=False,
    ).tolist()

    # Discover and evaluate models
    all_results: Dict[str, Dict[str, Any]] = {}
    all_predictions: Dict[str, pd.DataFrame] = {}

    # --- Surprise models ---
    for subdir in models_dir.iterdir():
        if not subdir.is_dir():
            continue

        model, model_name = load_svd_model(subdir)
        if model is not None:
            logger.info("Evaluating %s …", model_name)
            preds_df = generate_predictions(model, predict_surprise, test_df)
            rating_metrics = evaluator.evaluate_rating_prediction(preds_df)

            user_recs = generate_topk(
                model, predict_surprise, sample_users, all_items, train_df, args.k
            )
            ranking_metrics = evaluator.evaluate_ranking(user_recs, args.k)

            all_results[model_name] = {
                "rating_metrics": rating_metrics,
                "ranking_metrics": ranking_metrics,
            }
            all_predictions[model_name] = preds_df

    # --- NeuMF ---
    for subdir in models_dir.iterdir():
        if not subdir.is_dir():
            continue
        neumf = load_torch_model(subdir, "neumf", data_dir)
        if neumf is not None:
            logger.info("Evaluating NeuMF …")
            preds_df = generate_predictions(neumf, predict_torch, test_df)
            rating_metrics = evaluator.evaluate_rating_prediction(preds_df)

            user_recs = generate_topk(
                neumf, predict_torch, sample_users, all_items, train_df, args.k
            )
            ranking_metrics = evaluator.evaluate_ranking(user_recs, args.k)

            all_results["neumf"] = {
                "rating_metrics": rating_metrics,
                "ranking_metrics": ranking_metrics,
            }
            all_predictions["neumf"] = preds_df

    # --- LightGCN ---
    for subdir in models_dir.iterdir():
        if not subdir.is_dir():
            continue
        lgcn = load_torch_model(subdir, "lightgcn", data_dir)
        if lgcn is not None:
            logger.info("Evaluating LightGCN …")
            preds_df = generate_predictions(lgcn, predict_torch, test_df)
            rating_metrics = evaluator.evaluate_rating_prediction(preds_df)

            user_recs = generate_topk(
                lgcn, predict_torch, sample_users, all_items, train_df, args.k
            )
            ranking_metrics = evaluator.evaluate_ranking(user_recs, args.k)

            all_results["lightgcn"] = {
                "rating_metrics": rating_metrics,
                "ranking_metrics": ranking_metrics,
            }
            all_predictions["lightgcn"] = preds_df

    # --- Ensemble ---
    if len(all_predictions) >= 2:
        logger.info("Evaluating ensemble (%d models) …", len(all_predictions))
        ens_df = ensemble_predictions(all_predictions)
        ens_rating = evaluator.evaluate_rating_prediction(ens_df)
        all_results["ensemble_avg"] = {"rating_metrics": ens_rating}
        logger.info("Ensemble RMSE=%.4f", ens_rating["rmse"])

    # --- Print comparison ---
    if all_results:
        comparison = evaluator.compare_models(all_results)
        print("\n" + "=" * 80)
        print("  MODEL COMPARISON")
        print("=" * 80)
        print(comparison.to_string())
        print("=" * 80 + "\n")

        # Save
        evaluator.save_results(all_results, output_dir / "all_results.json")
        comparison.to_csv(output_dir / "model_comparison.csv")
        logger.info("Comparison saved to %s", output_dir)
    else:
        logger.warning("No models found in %s", models_dir)

    logger.info("Evaluation complete ✓")


if __name__ == "__main__":
    main()
