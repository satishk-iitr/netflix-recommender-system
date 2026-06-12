"""
Top-K Recommendation Generator for the Netflix Prize System.

Given a trained model, generates ranked recommendation lists per user,
handles seen-item filtering, and provides success/failure analysis tools.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback
    def tqdm(it, **_kwargs):
        return it

logger = logging.getLogger(__name__)


class TopKGenerator:
    """Generate and analyse Top-K recommendations.

    Parameters
    ----------
    model : object
        Trained recommendation model.  Must expose a
        ``predict(user_id, item_id) -> float`` method, **or** supply a
        custom *predict_fn* when calling generation methods.
    train_df : pd.DataFrame
        Training data with at least ``['user_id', 'movie_id']`` columns.
        Used to identify items a user has already interacted with.
    k : int
        Default number of recommendations.
    predict_fn : callable, optional
        ``(model, user_id, item_id) -> float``.  Falls back to
        ``model.predict(user_id, item_id)`` if not provided.
    """

    def __init__(
        self,
        model: Any,
        train_df: pd.DataFrame,
        k: int = 10,
        predict_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.k = k

        # Build user → seen-items index
        self._user_seen: Dict[int, Set[int]] = {}
        for uid, group in train_df.groupby("user_id"):
            self._user_seen[int(uid)] = set(group["movie_id"].tolist())

        if predict_fn is not None:
            self._predict = predict_fn
        else:
            self._predict = lambda m, u, i: m.predict(u, i)

        logger.info(
            "TopKGenerator ready — %d users, default k=%d",
            len(self._user_seen),
            self.k,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_seen_items(self, user_id: int) -> Set[int]:
        """Return the set of item IDs the user has rated in training."""
        return self._user_seen.get(user_id, set())

    def generate_for_user(
        self,
        user_id: int,
        all_items: List[int],
        k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Generate Top-K recommendations for a single user.

        Parameters
        ----------
        user_id : int
        all_items : list of int
            Full item catalogue.
        k : int, optional
            Number of items to return; defaults to ``self.k``.

        Returns
        -------
        list of (item_id, score)
            Sorted in descending order of predicted score.
        """
        k = k or self.k
        seen = self.get_seen_items(user_id)
        candidates = [iid for iid in all_items if iid not in seen]

        scores: List[Tuple[int, float]] = []
        for iid in candidates:
            try:
                score = self._predict(self.model, user_id, iid)
            except Exception:
                score = 0.0
            scores.append((iid, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def generate_for_users(
        self,
        user_ids: List[int],
        all_items: List[int],
        k: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Generate Top-K recommendations for multiple users.

        Parameters
        ----------
        user_ids : list of int
        all_items : list of int
        k : int, optional
        show_progress : bool
            Show a tqdm progress bar (if available).

        Returns
        -------
        dict
            ``{user_id: [(item_id, score), ...]}``
        """
        k = k or self.k
        results: Dict[int, List[Tuple[int, float]]] = {}

        iterator = tqdm(user_ids, desc="Generating Top-K", disable=not show_progress)
        for uid in iterator:
            results[int(uid)] = self.generate_for_user(uid, all_items, k)

        logger.info(
            "Generated Top-%d for %d users", k, len(results)
        )
        return results

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_recommendations(
        self,
        user_recs: Dict[int, List[Tuple[int, float]]],
        test_df: pd.DataFrame,
        movie_titles_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Analyse recommendation quality with success / failure cases.

        Parameters
        ----------
        user_recs : dict
            ``{user_id: [(item_id, score), ...]}``
        test_df : pd.DataFrame
            Test set with ``['user_id', 'movie_id', 'rating']``.
        movie_titles_df : pd.DataFrame, optional
            Must contain ``['movie_id', 'title']`` for readable output.

        Returns
        -------
        dict
            ``{
                'n_users': int,
                'avg_score': float,
                'success_cases': list[dict],
                'failure_cases': list[dict],
            }``
        """
        # Build test lookup: (user, item) → rating
        test_lookup: Dict[Tuple[int, int], float] = {}
        for _, row in test_df.iterrows():
            test_lookup[(int(row["user_id"]), int(row["movie_id"]))] = float(
                row["rating"]
            )

        # Title lookup
        title_map: Dict[int, str] = {}
        if movie_titles_df is not None and "title" in movie_titles_df.columns:
            for _, row in movie_titles_df.iterrows():
                title_map[int(row["movie_id"])] = str(row["title"])

        success_cases: List[Dict[str, Any]] = []
        failure_cases: List[Dict[str, Any]] = []
        all_scores: List[float] = []

        for uid, recs in user_recs.items():
            rec_items = [iid for iid, _ in recs]
            hits = [
                iid
                for iid in rec_items
                if test_lookup.get((uid, iid), 0) >= 3.5
            ]
            misses = [
                iid
                for iid in rec_items
                if (uid, iid) in test_lookup and test_lookup[(uid, iid)] < 3.5
            ]

            case_info = {
                "user_id": uid,
                "n_hits": len(hits),
                "n_misses": len(misses),
                "hit_items": [
                    {"movie_id": iid, "title": title_map.get(iid, "N/A")}
                    for iid in hits[:5]
                ],
                "miss_items": [
                    {
                        "movie_id": iid,
                        "title": title_map.get(iid, "N/A"),
                        "actual_rating": test_lookup.get((uid, iid)),
                    }
                    for iid in misses[:5]
                ],
            }

            if hits:
                success_cases.append(case_info)
                all_scores.append(len(hits) / len(rec_items))
            else:
                failure_cases.append(case_info)
                all_scores.append(0.0)

        # Sort by quality
        success_cases.sort(key=lambda c: c["n_hits"], reverse=True)
        failure_cases.sort(key=lambda c: c["n_misses"], reverse=True)

        return {
            "n_users": len(user_recs),
            "avg_hit_ratio": float(np.mean(all_scores)) if all_scores else 0.0,
            "success_cases": success_cases[:20],
            "failure_cases": failure_cases[:20],
        }

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_recommendations(
        self,
        user_id: int,
        recs: List[Tuple[int, float]],
        movie_titles_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """Return a human-readable string showing recommendations.

        Parameters
        ----------
        user_id : int
        recs : list of (item_id, score)
        movie_titles_df : pd.DataFrame, optional

        Returns
        -------
        str
        """
        title_map: Dict[int, str] = {}
        if movie_titles_df is not None and "title" in movie_titles_df.columns:
            for _, row in movie_titles_df.iterrows():
                title_map[int(row["movie_id"])] = str(row["title"])

        lines = [f"Top-{len(recs)} Recommendations for User {user_id}"]
        lines.append("-" * 55)
        lines.append(f"{'Rank':<6}{'Movie ID':<12}{'Score':<10}{'Title'}")
        lines.append("-" * 55)
        for rank, (iid, score) in enumerate(recs, start=1):
            title = title_map.get(iid, "—")
            lines.append(f"{rank:<6}{iid:<12}{score:<10.4f}{title}")
        lines.append("-" * 55)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_recommendations(
        user_recs: Dict[int, List[Tuple[int, float]]],
        path: str | Path,
    ) -> None:
        """Save recommendations to a CSV file.

        Columns: ``user_id, rank, movie_id, score``

        Parameters
        ----------
        user_recs : dict
            ``{user_id: [(item_id, score), ...]}``
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "rank", "movie_id", "score"])
            for uid in sorted(user_recs.keys()):
                for rank, (iid, score) in enumerate(user_recs[uid], start=1):
                    writer.writerow([uid, rank, iid, f"{score:.6f}"])

        logger.info(
            "Saved %d user recommendations to %s",
            len(user_recs),
            path,
        )
