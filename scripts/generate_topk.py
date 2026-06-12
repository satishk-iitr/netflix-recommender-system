#!/usr/bin/env python
"""
generate_topk.py — Generate Top-K recommendations for sample users and
produce a detailed analysis report.

Usage:
    python scripts/generate_topk.py \\
        --model-path models/svd/svd_model.pkl \\
        --data-dir data/processed \\
        --output-dir results/topk \\
        --k 10 --n-users 50

Outputs:
    - recommendations.csv      — per-user ranked lists
    - analysis_report.json     — success / failure analysis
    - sample_recommendations.txt — human-readable display
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.topk import TopKGenerator
from src.evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ======================================================================
# Model loading helpers
# ======================================================================

def load_model(model_path: Path, data_dir: Path):
    """Auto-detect and load the model from its file extension / name."""
    model_path = Path(model_path)

    if model_path.suffix == ".pkl":
        # Surprise model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        predict_fn = lambda m, u, i: m.predict(str(u), str(i)).est
        logger.info("Loaded Surprise model from %s", model_path)
        return model, predict_fn

    elif model_path.suffix == ".pt":
        import torch

        name = model_path.stem.replace("_model", "").replace("_best", "")

        # Load metadata
        results_path = model_path.parent / f"{name}_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing results JSON: {results_path}")

        with open(results_path, "r") as f:
            meta = json.load(f)

        n_users = meta["n_users"]
        n_items = meta["n_items"]
        config = meta.get("config", {})

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

        if "neumf" in name:
            from train_neumf import build_neumf_model

            model = build_neumf_model(n_users, n_items, config)
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
        elif "lightgcn" in name:
            from train_lightgcn import build_lightgcn_model, build_adj_matrix

            model = build_lightgcn_model(n_users, n_items, config)
            model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            train_df = pd.read_csv(data_dir / "train.csv")
            adj = build_adj_matrix(train_df, n_users, n_items)
            model.set_adj(adj)
        else:
            raise ValueError(f"Cannot determine model type from: {model_path.name}")

        model.eval()
        predict_fn = lambda m, u, i: m.predict(u, i)
        logger.info("Loaded PyTorch model (%s) from %s", name, model_path)
        return model, predict_fn

    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Top-K recommendations and analyse quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a trained model file (.pkl or .pt)",
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--n-users",
        type=int,
        default=50,
        help="Number of sample users for recommendation",
    )
    parser.add_argument(
        "--movie-titles",
        type=str,
        default=None,
        help="Path to CSV with movie_id,title columns (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    all_items = sorted(
        set(train_df["movie_id"].unique()) | set(test_df["movie_id"].unique())
    )

    # Movie titles (optional)
    movie_titles_df = None
    if args.movie_titles and Path(args.movie_titles).exists():
        movie_titles_df = pd.read_csv(args.movie_titles)
        logger.info("Loaded %d movie titles", len(movie_titles_df))

    # 2. Load model
    model, predict_fn = load_model(args.model_path, data_dir)

    # 3. Build generator
    generator = TopKGenerator(model, train_df, k=args.k, predict_fn=predict_fn)

    # 4. Sample users
    rng = np.random.default_rng(42)
    test_users = test_df["user_id"].unique()
    n_sample = min(args.n_users, len(test_users))
    sample_users = rng.choice(test_users, size=n_sample, replace=False).tolist()
    logger.info("Generating Top-%d for %d sample users", args.k, n_sample)

    # 5. Generate recommendations
    user_recs = generator.generate_for_users(
        sample_users, all_items, k=args.k, show_progress=True
    )

    # 6. Save recommendations
    generator.save_recommendations(user_recs, output_dir / "recommendations.csv")

    # 7. Analysis
    analysis = generator.analyze_recommendations(
        user_recs, test_df, movie_titles_df
    )
    with open(output_dir / "analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(
        "Analysis: avg_hit_ratio=%.4f, %d successes, %d failures",
        analysis["avg_hit_ratio"],
        len(analysis["success_cases"]),
        len(analysis["failure_cases"]),
    )

    # 8. Human-readable sample
    lines = []
    for uid in sample_users[:5]:
        recs = user_recs.get(uid, [])
        lines.append(generator.format_recommendations(uid, recs, movie_titles_df))
        lines.append("")

    sample_text = "\n".join(lines)
    with open(output_dir / "sample_recommendations.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    print(sample_text)

    # 9. Quick evaluation
    evaluator = Evaluator(test_df, relevance_threshold=3.5, k=args.k)
    recs_for_eval = {uid: [iid for iid, _ in recs] for uid, recs in user_recs.items()}
    ranking_metrics = evaluator.evaluate_ranking(recs_for_eval, args.k)
    evaluator.print_results({"ranking_metrics": ranking_metrics})
    evaluator.save_results(ranking_metrics, output_dir / "ranking_metrics.json")

    logger.info("Top-K generation complete ✓  Output: %s", output_dir)


if __name__ == "__main__":
    main()
