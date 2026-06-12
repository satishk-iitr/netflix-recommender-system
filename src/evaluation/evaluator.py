"""
Unified Evaluator for the Netflix Prize Recommendation System.

Brings together rating-prediction and ranking metrics into a single
evaluation harness that can compare multiple models side-by-side.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    rmse,
    mae,
    precision_at_k,
    recall_at_k,
    ap_at_k,
    map_at_k,
    ndcg_at_k,
    coverage,
    hit_rate,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """End-to-end evaluator for recommendation models.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test split with at least columns ``['user_id', 'movie_id', 'rating']``.
    relevance_threshold : float
        Ratings >= this value mark an item as relevant (default 3.5).
    k : int
        Default cut-off for ranking metrics.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        relevance_threshold: float = 3.5,
        k: int = 10,
    ) -> None:
        required_cols = {"user_id", "movie_id", "rating"}
        missing = required_cols - set(test_df.columns)
        if missing:
            raise ValueError(f"test_df is missing columns: {missing}")

        self.test_df = test_df.copy()
        self.relevance_threshold = relevance_threshold
        self.k = k

        # Pre-compute relevant items per user
        self._relevant_items = self.get_relevant_items(
            self.test_df, self.relevance_threshold
        )
        logger.info(
            "Evaluator initialised — %d test interactions, %d users with "
            "relevant items (threshold=%.1f)",
            len(self.test_df),
            len(self._relevant_items),
            self.relevance_threshold,
        )

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_relevant_items(
        test_df: pd.DataFrame,
        threshold: float = 3.5,
    ) -> Dict[int, Set[int]]:
        """Build a mapping of user → relevant item IDs from the test set.

        Parameters
        ----------
        test_df : pd.DataFrame
            Must contain ``user_id``, ``movie_id``, ``rating``.
        threshold : float
            Minimum rating for relevance.

        Returns
        -------
        dict
            ``{user_id: set(movie_ids)}`` with only users that have ≥1
            relevant item.
        """
        relevant = test_df[test_df["rating"] >= threshold]
        result: Dict[int, Set[int]] = {}
        for user_id, group in relevant.groupby("user_id"):
            result[int(user_id)] = set(group["movie_id"].tolist())
        return result

    # ------------------------------------------------------------------
    # Rating-prediction evaluation
    # ------------------------------------------------------------------

    def evaluate_rating_prediction(
        self,
        predictions_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Evaluate predicted ratings against the test set.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Must contain ``user_id``, ``movie_id``, ``predicted_rating``.

        Returns
        -------
        dict
            ``{'rmse': float, 'mae': float}``
        """
        required = {"user_id", "movie_id", "predicted_rating"}
        missing = required - set(predictions_df.columns)
        if missing:
            raise ValueError(f"predictions_df missing columns: {missing}")

        merged = self.test_df.merge(
            predictions_df[["user_id", "movie_id", "predicted_rating"]],
            on=["user_id", "movie_id"],
            how="inner",
        )
        if merged.empty:
            raise ValueError(
                "No overlapping (user_id, movie_id) pairs between "
                "test_df and predictions_df"
            )

        y_true = merged["rating"].values
        y_pred = merged["predicted_rating"].values

        return {
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n_predictions": len(merged),
        }

    # ------------------------------------------------------------------
    # Ranking evaluation
    # ------------------------------------------------------------------

    def evaluate_ranking(
        self,
        user_recommendations: Dict[int, List[int]],
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate a ranked recommendation list per user.

        Parameters
        ----------
        user_recommendations : dict
            ``{user_id: [item_id, ...]}`` ordered by predicted score
            (descending).
        k : int, optional
            Cut-off; defaults to ``self.k``.

        Returns
        -------
        dict
            Keys: ``map@k``, ``precision@k``, ``recall@k``, ``ndcg@k``,
            ``hit_rate``, ``coverage``, ``n_users_evaluated``.
        """
        k = k or self.k
        relevant_items = self._relevant_items

        precisions: List[float] = []
        recalls: List[float] = []
        ndcgs: List[float] = []

        for user_id, recs in user_recommendations.items():
            rel = relevant_items.get(user_id, set())
            if not rel:
                continue
            precisions.append(precision_at_k(recs, rel, k))
            recalls.append(recall_at_k(recs, rel, k))
            ndcgs.append(ndcg_at_k(recs, rel, k))

        n_users = len(precisions)
        all_items = self.test_df["movie_id"].nunique()

        return {
            f"map@{k}": map_at_k(
                user_recommendations, relevant_items, k,
                relevance_threshold=self.relevance_threshold,
            ),
            f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
            f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
            f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "hit_rate": hit_rate(user_recommendations, relevant_items),
            "coverage": coverage(user_recommendations, all_items),
            "n_users_evaluated": n_users,
        }

    # ------------------------------------------------------------------
    # Full evaluation pipeline
    # ------------------------------------------------------------------

    def full_evaluation(
        self,
        model: Any,
        test_df: pd.DataFrame,
        all_items: List[int],
        k: Optional[int] = None,
        predict_fn: Optional[Callable] = None,
        recommend_fn: Optional[Callable] = None,
        n_sample_users: int = 500,
    ) -> Dict[str, Any]:
        """Run both rating prediction and ranking evaluation.

        Parameters
        ----------
        model : object
            Recommendation model.  Must expose ``predict(user_id, item_id)``
            unless *predict_fn* is provided.
        test_df : pd.DataFrame
            Test split (used for rating prediction).
        all_items : list of int
            Full item catalogue.
        k : int, optional
            Ranking cut-off.
        predict_fn : callable, optional
            ``(model, user_id, item_id) -> float``.  Defaults to
            ``model.predict(user_id, item_id)``.
        recommend_fn : callable, optional
            ``(model, user_id, all_items, k) -> [(item_id, score), ...]``.
            If not provided, scores are generated via *predict_fn*.
        n_sample_users : int
            Number of users to sample for ranking evaluation (for speed).

        Returns
        -------
        dict
            Combined results with keys ``rating_metrics`` and
            ``ranking_metrics``.
        """
        k = k or self.k

        # --- Rating prediction ---
        if predict_fn is None:
            def predict_fn(m, u, i):
                return m.predict(u, i)

        logger.info("Generating rating predictions on test set (%d rows)…", len(test_df))
        preds = []
        for _, row in test_df.iterrows():
            uid, iid = int(row["user_id"]), int(row["movie_id"])
            try:
                score = predict_fn(model, uid, iid)
            except Exception:
                score = 3.0  # safe fallback (global mean-ish)
            preds.append(
                {"user_id": uid, "movie_id": iid, "predicted_rating": float(score)}
            )
        predictions_df = pd.DataFrame(preds)
        rating_results = self.evaluate_rating_prediction(predictions_df)

        # --- Ranking ---
        users = test_df["user_id"].unique()
        if len(users) > n_sample_users:
            rng = np.random.default_rng(42)
            users = rng.choice(users, size=n_sample_users, replace=False)

        logger.info("Generating Top-%d recommendations for %d users…", k, len(users))
        user_recs: Dict[int, List[int]] = {}

        if recommend_fn is not None:
            for uid in users:
                uid = int(uid)
                recs = recommend_fn(model, uid, all_items, k)
                user_recs[uid] = [item_id for item_id, _ in recs]
        else:
            # Fallback: score every item for every user (slow but universal)
            train_items_set = set(all_items)
            for uid in users:
                uid = int(uid)
                scores = []
                for iid in all_items:
                    try:
                        s = predict_fn(model, uid, iid)
                    except Exception:
                        s = 0.0
                    scores.append((iid, float(s)))
                scores.sort(key=lambda x: x[1], reverse=True)
                user_recs[uid] = [iid for iid, _ in scores[:k]]

        ranking_results = self.evaluate_ranking(user_recs, k)

        return {
            "rating_metrics": rating_results,
            "ranking_metrics": ranking_results,
        }

    # ------------------------------------------------------------------
    # Display / persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def print_results(results: Dict[str, Any]) -> None:
        """Pretty-print evaluation results to stdout."""
        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)

        if "rating_metrics" in results:
            print("\n  Rating Prediction Metrics")
            print("  " + "-" * 40)
            for key, val in results["rating_metrics"].items():
                if isinstance(val, float):
                    print(f"    {key:<25s} {val:.6f}")
                else:
                    print(f"    {key:<25s} {val}")

        if "ranking_metrics" in results:
            print("\n  Ranking Metrics")
            print("  " + "-" * 40)
            for key, val in results["ranking_metrics"].items():
                if isinstance(val, float):
                    print(f"    {key:<25s} {val:.6f}")
                else:
                    print(f"    {key:<25s} {val}")

        # Flat dict (e.g. from evaluate_rating_prediction directly)
        flat_keys = set(results.keys()) - {"rating_metrics", "ranking_metrics"}
        if flat_keys:
            for key in sorted(flat_keys):
                val = results[key]
                if isinstance(val, float):
                    print(f"    {key:<25s} {val:.6f}")
                else:
                    print(f"    {key:<25s} {val}")

        print("=" * 60 + "\n")

    @staticmethod
    def save_results(results: Dict[str, Any], path: str | Path) -> None:
        """Persist results to a JSON file.

        Parameters
        ----------
        results : dict
            Evaluation results (must be JSON-serialisable).
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Make numpy types JSON-safe
        def _convert(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, set):
                return sorted(obj)
            return obj

        clean = json.loads(json.dumps(results, default=_convert))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
        logger.info("Results saved to %s", path)

    @staticmethod
    def compare_models(
        results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Build a comparison table across multiple models.

        Parameters
        ----------
        results : dict
            ``{model_name: evaluation_results_dict}`` — each value is the
            dict returned by ``full_evaluation``, ``evaluate_rating_prediction``,
            or ``evaluate_ranking``.

        Returns
        -------
        pd.DataFrame
            One row per model, columns for every metric encountered.
        """
        rows = []
        for model_name, res in results.items():
            flat: Dict[str, Any] = {"model": model_name}

            # Flatten nested structure
            if "rating_metrics" in res:
                for k, v in res["rating_metrics"].items():
                    flat[k] = v
            if "ranking_metrics" in res:
                for k, v in res["ranking_metrics"].items():
                    flat[k] = v

            # Handle already-flat dicts
            for k, v in res.items():
                if k not in ("rating_metrics", "ranking_metrics"):
                    flat.setdefault(k, v)

            rows.append(flat)

        df = pd.DataFrame(rows).set_index("model")
        return df
