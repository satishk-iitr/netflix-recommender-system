"""
Evaluation metrics for the Netflix Prize Recommendation System.

Covers both rating-prediction accuracy (RMSE, MAE) and ranking quality
(Precision@K, Recall@K, AP@K, MAP@K, NDCG@K, Hit Rate, Coverage, Diversity).

Competition note:
    The Netflix Prize uses RMSE as its primary metric.  For ranking evaluation
    the relevance threshold defaults to >= 3.5 (ratings on a 1-5 scale).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Rating-prediction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth ratings.
    y_pred : array-like
        Predicted ratings.

    Returns
    -------
    float
        RMSE value (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth ratings.
    y_pred : array-like
        Predicted ratings.

    Returns
    -------
    float
        MAE value (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.mean(np.abs(y_true - y_pred)))


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    """Precision@K — fraction of top-K recommendations that are relevant.

    Parameters
    ----------
    recommended : list of int
        Item IDs ordered by predicted relevance (descending).
    relevant : set of int
        Ground-truth relevant item IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Precision value in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    """Recall@K — fraction of relevant items captured in top-K.

    Parameters
    ----------
    recommended : list of int
        Item IDs ordered by predicted relevance (descending).
    relevant : set of int
        Ground-truth relevant item IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        Recall value in [0, 1].  Returns 0 if there are no relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ap_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    """Average Precision at K.

    AP@K = (1 / min(|relevant|, K)) * Σ_{i=1}^{K} P(i) * rel(i)

    where P(i) is precision at cut-off i and rel(i) is 1 if the item at
    rank i is relevant, 0 otherwise.

    Parameters
    ----------
    recommended : list of int
        Item IDs ordered by predicted relevance (descending).
    relevant : set of int
        Ground-truth relevant item IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        AP@K value in [0, 1].  Returns 0 if there are no relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0

    top_k = list(recommended)[:k]
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            hits += 1
            sum_precision += hits / i  # P(i) at the hit position

    denominator = min(len(relevant), k)
    return sum_precision / denominator


def map_at_k(
    user_recommendations: Dict[int, List[int]],
    user_relevant_items: Dict[int, Set[int]],
    k: int = 10,
    relevance_threshold: float = 3.5,
) -> float:
    """Mean Average Precision at K across all users.

    Parameters
    ----------
    user_recommendations : dict
        ``{user_id: [item_id, ...]}`` ordered by predicted score descending.
    user_relevant_items : dict
        ``{user_id: set(item_ids)}`` where items have rating >= *relevance_threshold*.
    k : int
        Cut-off rank.
    relevance_threshold : float
        Minimum rating to consider an item relevant (default 3.5 per
        Netflix Prize competition rules).  This parameter is kept here for
        documentation; filtering should already be applied when building
        *user_relevant_items*.

    Returns
    -------
    float
        MAP@K in [0, 1].  Users with zero relevant items in the test set
        are **excluded** from the average.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    total_ap = 0.0
    n_valid_users = 0

    for user_id, recs in user_recommendations.items():
        relevant = user_relevant_items.get(user_id, set())
        if not relevant:
            # Exclude users with no relevant items from average
            continue
        total_ap += ap_at_k(recs, relevant, k)
        n_valid_users += 1

    if n_valid_users == 0:
        return 0.0
    return total_ap / n_valid_users


def ndcg_at_k(
    recommended: Sequence[int],
    relevant: Set[int],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance).

    Parameters
    ----------
    recommended : list of int
        Item IDs ordered by predicted relevance (descending).
    relevant : set of int
        Ground-truth relevant item IDs.
    k : int
        Cut-off rank.

    Returns
    -------
    float
        NDCG@K in [0, 1].
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0

    top_k = list(recommended)[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG (all relevant items ranked perfectly)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Beyond-accuracy / system-level metrics
# ---------------------------------------------------------------------------

def coverage(
    all_recommendations: Dict[int, List[int]],
    n_total_items: int,
) -> float:
    """Catalogue coverage — fraction of all items appearing in any recommendation.

    Parameters
    ----------
    all_recommendations : dict
        ``{user_id: [item_id, ...]}`` for every user.
    n_total_items : int
        Total number of unique items in the catalogue.

    Returns
    -------
    float
        Coverage in [0, 1].
    """
    if n_total_items <= 0:
        raise ValueError("n_total_items must be positive")
    recommended_items: Set[int] = set()
    for recs in all_recommendations.values():
        recommended_items.update(recs)
    return len(recommended_items) / n_total_items


def diversity(
    recommendations: List[int],
    item_similarity_matrix: np.ndarray,
) -> float:
    """Intra-list diversity — average pairwise *dissimilarity* of recommended items.

    Parameters
    ----------
    recommendations : list of int
        Item indices (0-based) into *item_similarity_matrix*.
    item_similarity_matrix : ndarray, shape (n_items, n_items)
        Pairwise cosine (or other) similarity matrix with values in [0, 1].

    Returns
    -------
    float
        Diversity in [0, 1].  Higher values mean more diverse lists.
    """
    if len(recommendations) < 2:
        return 0.0

    n_pairs = 0
    total_dissim = 0.0
    recs = list(recommendations)
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            sim = item_similarity_matrix[recs[i], recs[j]]
            total_dissim += 1.0 - sim
            n_pairs += 1

    return total_dissim / n_pairs if n_pairs > 0 else 0.0


def hit_rate(
    user_recommendations: Dict[int, List[int]],
    user_relevant_items: Dict[int, Set[int]],
) -> float:
    """Hit Rate — fraction of users for whom at least one recommendation is relevant.

    Parameters
    ----------
    user_recommendations : dict
        ``{user_id: [item_id, ...]}`` for each user.
    user_relevant_items : dict
        ``{user_id: set(relevant_item_ids)}`` for each user.

    Returns
    -------
    float
        Hit rate in [0, 1].
    """
    if not user_recommendations:
        return 0.0

    hits = 0
    total = 0
    for user_id, recs in user_recommendations.items():
        relevant = user_relevant_items.get(user_id, set())
        if not relevant:
            continue  # skip users with no relevant items
        total += 1
        if any(item in relevant for item in recs):
            hits += 1

    return hits / total if total > 0 else 0.0
