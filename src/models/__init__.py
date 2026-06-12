"""
Netflix Recommendation System - Model Module
=============================================

Exports all model classes for collaborative filtering, neural, 
graph-based, and ensemble recommendation approaches.
"""

from src.models.svd_model import SVDRecommender
from src.models.neumf import NeuMF, NeuMFTrainer
from src.models.lightgcn import LightGCN, LightGCNTrainer
from src.models.ensemble import WeightedEnsemble, StackingEnsemble

__all__ = [
    "SVDRecommender",
    "NeuMF",
    "NeuMFTrainer",
    "LightGCN",
    "LightGCNTrainer",
    "WeightedEnsemble",
    "StackingEnsemble",
]
