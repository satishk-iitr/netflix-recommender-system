"""
Evaluation module for the Netflix Prize Recommendation System.

Provides metrics, evaluator classes, and utilities for assessing
recommendation quality across rating prediction and ranking tasks.
"""

from src.evaluation.metrics import (
    rmse,
    mae,
    precision_at_k,
    recall_at_k,
    ap_at_k,
    map_at_k,
    ndcg_at_k,
    coverage,
    diversity,
    hit_rate,
)
from src.evaluation.evaluator import Evaluator

__all__ = [
    # Rating prediction metrics
    "rmse",
    "mae",
    # Ranking metrics
    "precision_at_k",
    "recall_at_k",
    "ap_at_k",
    "map_at_k",
    "ndcg_at_k",
    # Beyond-accuracy metrics
    "coverage",
    "diversity",
    "hit_rate",
    # Evaluator
    "Evaluator",
]
